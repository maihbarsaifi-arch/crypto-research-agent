import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

try:
    import google.generativeai as genai
except Exception:  # package may not be installed yet
    genai = None


st.set_page_config(
    page_title="Crypto Trading Research Agent",
    page_icon="📊",
    layout="wide",
)


DEFAULT_SYMBOLS = {
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "Solana": "SOL-USD",
    "BNB": "BNB-USD",
    "XRP": "XRP-USD",
    "Cardano": "ADA-USD",
    "Dogecoin": "DOGE-USD",
    "Polygon": "MATIC-USD",
    "Litecoin": "LTC-USD",
    "Chainlink": "LINK-USD",
}

# Yahoo Finance kabhi-kabhi Streamlit Cloud par data block/rate-limit kar deta hai.
# Isliye CoinGecko fallback add hai, taaki app phone/cloud par reliably chale.
COINGECKO_IDS = {
    "BTC-USD": "bitcoin",
    "ETH-USD": "ethereum",
    "SOL-USD": "solana",
    "BNB-USD": "binancecoin",
    "XRP-USD": "ripple",
    "ADA-USD": "cardano",
    "DOGE-USD": "dogecoin",
    "MATIC-USD": "matic-network",
    "LINK-USD": "chainlink",
    "LTC-USD": "litecoin",
}

PERIOD_DAYS = {
    "1mo": 30,
    "3mo": 90,
    "6mo": 180,
    "1y": 365,
    "2y": 730,
    "5y": 1825,
}

RESAMPLE_RULES = {
    "1d": "1D",
    "1h": "1H",
    "30m": "30min",
    "15m": "15min",
}


DISCLAIMER = """
⚠️ **Disclaimer:** Ye app sirf educational/research purpose ke liye hai. Crypto highly volatile hota hai.
Ye financial advice nahi hai. Live trade lene se pehle apna analysis, risk management aur position sizing zaroor karein.
"""


def fetch_yfinance_data(symbol: str, period: str, interval: str) -> pd.DataFrame:
    try:
        data = yf.download(
            symbol,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False,
            threads=True,
        )
    except Exception:
        return pd.DataFrame()

    if data is None or data.empty:
        return pd.DataFrame()

    # yfinance sometimes returns MultiIndex columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]

    data = data.reset_index()
    data = data.rename(columns={"Date": "Datetime"})
    data = data.dropna(subset=["Open", "High", "Low", "Close"])
    return data


def fetch_coingecko_data(symbol: str, period: str, interval: str) -> pd.DataFrame:
    coin_id = COINGECKO_IDS.get(symbol.upper())
    if not coin_id:
        return pd.DataFrame()

    days = PERIOD_DAYS.get(period, 180)
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"

    try:
        response = requests.get(
            url,
            params={"vs_currency": "usd", "days": days},
            timeout=20,
            headers={"accept": "application/json", "user-agent": "crypto-research-agent"},
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return pd.DataFrame()

    prices = payload.get("prices", [])
    volumes = payload.get("total_volumes", [])
    if not prices:
        return pd.DataFrame()

    price_df = pd.DataFrame(prices, columns=["timestamp", "Close"])
    price_df["Datetime"] = pd.to_datetime(price_df["timestamp"], unit="ms", utc=True).dt.tz_convert(None)
    price_df = price_df[["Datetime", "Close"]].sort_values("Datetime")

    if volumes:
        volume_df = pd.DataFrame(volumes, columns=["timestamp", "Volume"])
        volume_df["Datetime"] = pd.to_datetime(volume_df["timestamp"], unit="ms", utc=True).dt.tz_convert(None)
        volume_df = volume_df[["Datetime", "Volume"]].sort_values("Datetime")
        price_df = pd.merge_asof(price_df, volume_df, on="Datetime")
    else:
        price_df["Volume"] = 0

    price_df = price_df.dropna(subset=["Close"])

    # CoinGecko returns hourly data for shorter periods and daily data for longer periods.
    # We convert price series into OHLC candles. For daily-only data, Open is previous close.
    if len(price_df) > days + 10:
        rule = RESAMPLE_RULES.get(interval, "1D")
        ohlc = price_df.set_index("Datetime").resample(rule).agg(
            Open=("Close", "first"),
            High=("Close", "max"),
            Low=("Close", "min"),
            Close=("Close", "last"),
            Volume=("Volume", "sum"),
        )
        data = ohlc.dropna().reset_index()
    else:
        data = price_df.copy()
        data["Open"] = data["Close"].shift(1).fillna(data["Close"])
        data["High"] = data[["Open", "Close"]].max(axis=1)
        data["Low"] = data[["Open", "Close"]].min(axis=1)
        data = data[["Datetime", "Open", "High", "Low", "Close", "Volume"]]

    return data.dropna(subset=["Open", "High", "Low", "Close"])


@st.cache_data(ttl=300, show_spinner=False)
def fetch_market_data(symbol: str, period: str, interval: str) -> pd.DataFrame:
    # First try Yahoo Finance. If it fails/returns too little data, fallback to CoinGecko.
    data = fetch_yfinance_data(symbol, period, interval)
    if data is not None and not data.empty and len(data) >= 25:
        return data

    fallback = fetch_coingecko_data(symbol, period, interval)
    if fallback is not None and not fallback.empty:
        return fallback

    return data if data is not None else pd.DataFrame()


def calculate_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA_200"] = df["Close"].ewm(span=200, adjust=False).mean()

    df["RSI_14"] = calculate_rsi(df["Close"], 14)

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    mid = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()
    df["BB_Mid"] = mid
    df["BB_Upper"] = mid + 2 * std
    df["BB_Lower"] = mid - 2 * std

    df["ATR_14"] = calculate_atr(df, 14)
    df["Volume_SMA_20"] = df["Volume"].rolling(20).mean()
    df["Support_20"] = df["Low"].rolling(20).min()
    df["Resistance_20"] = df["High"].rolling(20).max()

    df["Return_1"] = df["Close"].pct_change() * 100
    df["Return_7"] = df["Close"].pct_change(7) * 100
    df["Return_30"] = df["Close"].pct_change(30) * 100

    return df


def format_money(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    if value >= 1:
        return f"${value:,.2f}"
    return f"${value:,.6f}"


def format_pct(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:+.2f}%"


def score_and_interpret(df: pd.DataFrame) -> dict:
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    price = float(latest["Close"])
    ema20 = float(latest["EMA_20"])
    ema50 = float(latest["EMA_50"])
    ema200 = float(latest["EMA_200"])
    rsi = float(latest["RSI_14"])
    macd = float(latest["MACD"])
    macd_signal = float(latest["MACD_Signal"])
    volume = float(latest.get("Volume", 0))
    volume_sma = float(latest.get("Volume_SMA_20", np.nan))
    support = float(latest.get("Support_20", np.nan))
    resistance = float(latest.get("Resistance_20", np.nan))
    atr = float(latest.get("ATR_14", np.nan))

    score = 0
    reasons = []
    risks = []

    # Trend score
    if price > ema20 > ema50:
        score += 2
        reasons.append("Price 20 EMA aur 50 EMA ke upar hai — short/mid-term trend positive.")
    elif price > ema20 and price > ema50:
        score += 1
        reasons.append("Price important EMAs ke upar hai, lekin EMA alignment perfect nahi hai.")
    elif price < ema20 < ema50:
        score -= 2
        risks.append("Price 20 EMA aur 50 EMA ke neeche hai — trend weak/bearish.")
    elif price < ema20 and price < ema50:
        score -= 1
        risks.append("Price EMAs ke neeche trade kar raha hai — caution.")

    if price > ema200:
        score += 1
        reasons.append("Price 200 EMA ke upar hai — long-term structure relatively stronger.")
    else:
        score -= 1
        risks.append("Price 200 EMA ke neeche hai — long-term trend pressure me ho sakta hai.")

    # Momentum score
    if 50 <= rsi <= 68:
        score += 1
        reasons.append(f"RSI {rsi:.1f} hai — momentum healthy zone me hai.")
    elif rsi > 70:
        score -= 1
        risks.append(f"RSI {rsi:.1f} hai — overbought zone, pullback risk.")
    elif rsi < 35:
        score -= 1
        risks.append(f"RSI {rsi:.1f} hai — momentum weak/oversold, reversal confirmation zaroori.")
    else:
        reasons.append(f"RSI {rsi:.1f} neutral zone me hai.")

    if macd > macd_signal:
        score += 1
        reasons.append("MACD signal line ke upar hai — momentum improving.")
    else:
        score -= 1
        risks.append("MACD signal line ke neeche hai — momentum soft.")

    # Volume score
    if not pd.isna(volume_sma) and volume_sma > 0:
        vol_ratio = volume / volume_sma
        if vol_ratio >= 1.3 and price >= float(prev["Close"]):
            score += 1
            reasons.append(f"Volume 20-period average se {vol_ratio:.1f}x hai — buying interest possible.")
        elif vol_ratio >= 1.3 and price < float(prev["Close"]):
            score -= 1
            risks.append(f"High volume ke saath red candle — selling pressure possible.")
        else:
            reasons.append("Volume normal range me hai.")
    else:
        vol_ratio = np.nan

    # Support/resistance position
    if not pd.isna(support) and not pd.isna(resistance) and resistance > support:
        distance_to_resistance = ((resistance - price) / price) * 100
        distance_to_support = ((price - support) / price) * 100
        if distance_to_resistance < 2:
            risks.append("Price near-term resistance ke bahut paas hai — breakout confirmation ka wait better.")
        if distance_to_support < 2:
            risks.append("Price support ke paas hai — support break hone par downside risk.")
    else:
        distance_to_resistance = np.nan
        distance_to_support = np.nan

    if score >= 4:
        bias = "Bullish / Watchlist Strong"
        action = "Trend positive hai. Breakout/continuation setup ke liye watchlist me rakh sakte ho. Entry confirmation zaroor check karein."
    elif score >= 1:
        bias = "Mild Bullish / Neutral Positive"
        action = "Setup mixed-positive hai. Confirmation, volume aur risk-reward check karke hi decision lein."
    elif score <= -3:
        bias = "Bearish / High Caution"
        action = "Trend weak hai. Fresh long entry risky ho sakti hai; reversal confirmation ka wait better."
    else:
        bias = "Neutral / Sideways"
        action = "Clear edge nahi dikh raha. Support-resistance breakout/breakdown ka wait karna better ho sakta hai."

    stop_loss_zone = None
    if not pd.isna(support):
        stop_loss_zone = support * 0.985
    elif not pd.isna(atr):
        stop_loss_zone = price - 1.5 * atr

    return {
        "price": price,
        "ema20": ema20,
        "ema50": ema50,
        "ema200": ema200,
        "rsi": rsi,
        "macd": macd,
        "macd_signal": macd_signal,
        "volume": volume,
        "volume_sma": volume_sma,
        "vol_ratio": vol_ratio,
        "support": support,
        "resistance": resistance,
        "atr": atr,
        "score": score,
        "bias": bias,
        "action": action,
        "reasons": reasons,
        "risks": risks,
        "distance_to_resistance": distance_to_resistance,
        "distance_to_support": distance_to_support,
        "stop_loss_zone": stop_loss_zone,
        "return_1": float(latest.get("Return_1", np.nan)),
        "return_7": float(latest.get("Return_7", np.nan)),
        "return_30": float(latest.get("Return_30", np.nan)),
    }


def build_rule_based_report(symbol: str, stats: dict) -> str:
    reasons = "\n".join([f"- {x}" for x in stats["reasons"]]) or "- Koi strong positive signal nahi."
    risks = "\n".join([f"- {x}" for x in stats["risks"]]) or "- Koi major technical risk flag nahi, but crypto volatility high hoti hai."

    report = f"""
## 📌 Crypto Research Report: `{symbol}`

### 1. Market Bias
**Bias:** {stats['bias']}  
**Score:** {stats['score']} / 6 approx

### 2. Key Levels
- Current Price: **{format_money(stats['price'])}**
- 20 EMA: **{format_money(stats['ema20'])}**
- 50 EMA: **{format_money(stats['ema50'])}**
- 200 EMA: **{format_money(stats['ema200'])}**
- 20-period Support: **{format_money(stats['support'])}**
- 20-period Resistance: **{format_money(stats['resistance'])}**
- Indicative risk zone / stop area: **{format_money(stats['stop_loss_zone'])}**

### 3. Momentum & Volume
- RSI 14: **{stats['rsi']:.2f}**
- MACD: **{stats['macd']:.4f}**
- MACD Signal: **{stats['macd_signal']:.4f}**
- Volume vs 20-period avg: **{stats['vol_ratio']:.2f}x**

### 4. Positive Points
{reasons}

### 5. Risk Points
{risks}

### 6. Research Conclusion
{stats['action']}

### 7. Risk Management Hint
Agar trade plan banta hai to pehle entry trigger, invalidation level, stop-loss aur risk/reward define karein. Single trade me capital ka small percentage hi risk karna safer hota hai.
"""
    return report


def build_ai_prompt(symbol: str, stats: dict, df_tail: pd.DataFrame) -> str:
    recent_rows = df_tail[["Datetime", "Open", "High", "Low", "Close", "Volume", "RSI_14", "EMA_20", "EMA_50", "MACD", "MACD_Signal"]].tail(15)
    recent_csv = recent_rows.to_csv(index=False)

    return f"""
You are a crypto trading research assistant. Create a clear Hinglish research report for educational purposes only.
Do not guarantee profits. Do not say "sure buy" or "sure sell". Focus on trend, momentum, support/resistance, risk, and a watchlist-style conclusion.

Crypto symbol: {symbol}
Current statistics:
- Current Price: {stats['price']}
- Bias: {stats['bias']}
- Score: {stats['score']}
- RSI 14: {stats['rsi']}
- EMA 20: {stats['ema20']}
- EMA 50: {stats['ema50']}
- EMA 200: {stats['ema200']}
- MACD: {stats['macd']}
- MACD Signal: {stats['macd_signal']}
- Support 20: {stats['support']}
- Resistance 20: {stats['resistance']}
- ATR 14: {stats['atr']}
- 1 period return %: {stats['return_1']}
- 7 period return %: {stats['return_7']}
- 30 period return %: {stats['return_30']}
- Suggested caution/stop zone: {stats['stop_loss_zone']}

Positive observations:
{stats['reasons']}

Risk observations:
{stats['risks']}

Recent OHLCV and indicators:
{recent_csv}

Report format:
1. Quick Summary
2. Trend Analysis
3. Momentum Analysis
4. Support/Resistance
5. Risk Factors
6. Watchlist Conclusion
7. Disclaimer
"""


def generate_gemini_report(api_key: str, model_name: str, prompt: str) -> str:
    if genai is None:
        raise RuntimeError("google-generativeai package installed nahi hai. `pip install google-generativeai` run karein.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    return response.text


def make_candlestick_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df["Datetime"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
        )
    )
    fig.add_trace(go.Scatter(x=df["Datetime"], y=df["EMA_20"], name="EMA 20", line=dict(width=1.4)))
    fig.add_trace(go.Scatter(x=df["Datetime"], y=df["EMA_50"], name="EMA 50", line=dict(width=1.4)))
    fig.add_trace(go.Scatter(x=df["Datetime"], y=df["BB_Upper"], name="BB Upper", line=dict(width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=df["Datetime"], y=df["BB_Lower"], name="BB Lower", line=dict(width=1, dash="dot")))

    fig.update_layout(
        title=f"{symbol} Price Chart",
        height=620,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def make_macd_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    colors = np.where(df["MACD_Hist"] >= 0, "#26a69a", "#ef5350")
    fig.add_trace(go.Bar(x=df["Datetime"], y=df["MACD_Hist"], marker_color=colors, name="Histogram"))
    fig.add_trace(go.Scatter(x=df["Datetime"], y=df["MACD"], name="MACD"))
    fig.add_trace(go.Scatter(x=df["Datetime"], y=df["MACD_Signal"], name="Signal"))
    fig.update_layout(height=300, template="plotly_dark", margin=dict(l=20, r=20, t=30, b=20))
    return fig


def make_rsi_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Datetime"], y=df["RSI_14"], name="RSI 14"))
    fig.add_hline(y=70, line_dash="dash", line_color="red")
    fig.add_hline(y=30, line_dash="dash", line_color="green")
    fig.update_yaxes(range=[0, 100])
    fig.update_layout(height=300, template="plotly_dark", margin=dict(l=20, r=20, t=30, b=20))
    return fig


# ----------------------------- UI -----------------------------

st.title("📊 Crypto Trading Research Agent")
st.caption("Crypto technical research + optional Gemini AI report generation")

with st.sidebar:
    st.header("⚙️ Settings")
    selected_name = st.selectbox("Crypto select karo", list(DEFAULT_SYMBOLS.keys()), index=0)
    custom_symbol = st.text_input("Ya custom Yahoo symbol", placeholder="Example: BTC-USD, ETH-USD")
    symbol = custom_symbol.strip().upper() if custom_symbol.strip() else DEFAULT_SYMBOLS[selected_name]

    period = st.selectbox("Data period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=2)
    interval = st.selectbox("Interval", ["1d", "1h", "30m", "15m"], index=0)

    st.divider()
    st.subheader("🤖 Gemini AI Report")
    use_ai = st.toggle("Gemini se AI report generate karo", value=False)
    api_key = st.text_input(
        "Gemini API Key",
        value=os.getenv("GEMINI_API_KEY", ""),
        type="password",
        help="Key app me save nahi hoti. Environment variable GEMINI_API_KEY bhi use kar sakte ho.",
    )
    model_name = st.text_input("Gemini model", value="gemini-1.5-flash")

    st.divider()
    st.markdown(DISCLAIMER)

if interval in ["15m", "30m", "1h"] and period in ["2y", "5y"]:
    st.warning("Intraday interval ke liye yfinance long period support nahi karta. Period 1mo/3mo try karein.")

with st.spinner(f"Fetching {symbol} data..."):
    raw_df = fetch_market_data(symbol, period, interval)

if raw_df.empty or len(raw_df) < 25:
    st.error("Data nahi mila ya insufficient data hai. Symbol/period/interval change karke try karein.")
    st.info("Tip: Default coins jaise Bitcoin/Ethereum select karke Period 6mo ya 1y aur Interval 1d try karein.")
    st.stop()

df = add_indicators(raw_df).dropna().reset_index(drop=True)

if df.empty or len(df) < 20:
    st.error("Indicators calculate karne ke liye enough data nahi hai. Longer period select karein.")
    st.stop()

stats = score_and_interpret(df)

# Top metrics
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Symbol", symbol)
c2.metric("Current Price", format_money(stats["price"]), format_pct(stats["return_1"]))
c3.metric("Bias", stats["bias"])
c4.metric("RSI", f"{stats['rsi']:.2f}")
c5.metric("Score", f"{stats['score']} / 6")

st.divider()

left, right = st.columns([2.2, 1])
with left:
    st.plotly_chart(make_candlestick_chart(df.tail(180), symbol), use_container_width=True)
with right:
    st.subheader("🔎 Key Levels")
    st.write(f"**Support:** {format_money(stats['support'])}")
    st.write(f"**Resistance:** {format_money(stats['resistance'])}")
    st.write(f"**ATR 14:** {format_money(stats['atr'])}")
    st.write(f"**Indicative stop/risk zone:** {format_money(stats['stop_loss_zone'])}")
    st.write(f"**7-period return:** {format_pct(stats['return_7'])}")
    st.write(f"**30-period return:** {format_pct(stats['return_30'])}")

    st.subheader("✅ Positives")
    for item in stats["reasons"][:5]:
        st.write(f"- {item}")

    st.subheader("⚠️ Risks")
    for item in stats["risks"][:5]:
        st.write(f"- {item}")

m1, m2 = st.columns(2)
with m1:
    st.subheader("RSI")
    st.plotly_chart(make_rsi_chart(df.tail(180)), use_container_width=True)
with m2:
    st.subheader("MACD")
    st.plotly_chart(make_macd_chart(df.tail(180)), use_container_width=True)

st.divider()
st.subheader("🧠 Research Report")

rule_report = build_rule_based_report(symbol, stats)

if use_ai:
    if not api_key:
        st.warning("Gemini API key daalo, warna rule-based report show hogi.")
        st.markdown(rule_report)
    else:
        prompt = build_ai_prompt(symbol, stats, df)
        try:
            with st.spinner("Gemini AI report generate kar raha hai..."):
                ai_report = generate_gemini_report(api_key, model_name, prompt)
            st.markdown(ai_report)
        except Exception as exc:
            st.error(f"Gemini report generate nahi ho payi: {exc}")
            st.markdown("### Fallback rule-based report")
            st.markdown(rule_report)
else:
    st.markdown(rule_report)

with st.expander("📄 Raw recent data"):
    st.dataframe(df.tail(50), use_container_width=True)

st.caption(f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
