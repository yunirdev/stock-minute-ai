import asyncio
import os
import threading
from typing import List
import time
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import duckdb
import plotly.graph_objects as go
from ingest.alpaca_stream import AlpacaBarStreamer, Bar


def parse_symbols(raw: str) -> List[str]:
    return [s.strip().upper() for s in raw.split(",") if s.strip()]

def load_recent_bars(db_path: str, symbol: str, limit: int = 200) -> pd.DataFrame:
    last_err = None
    for _ in range(5):
        try:
            con = duckdb.connect(db_path)
            df = con.execute(
                """
                SELECT symbol, ts as timestamp, open, high, low, close, volume
                FROM minute_bars
                WHERE symbol = ?
                ORDER BY ts DESC
                LIMIT ?
                """,
                [symbol, limit],
            ).df()
            con.close()
            if df.empty:
                return df
            return df.sort_values("timestamp")
        except Exception as e:
            last_err = e
            time.sleep(0.05)
    raise last_err

def make_hint(bar: Bar) -> str:
    # 先用规则提示（MVP）：后面你再接 ML/LLM
    direction = "上涨" if bar.close >= bar.open else "下跌"
    rng = (bar.high - bar.low) if (bar.high and bar.low) else 0.0
    return f"{bar.symbol} {direction}，本分钟振幅 {rng:.4f}，成交量 {bar.volume:.0f}"


@st.cache_resource
def start_streamer(symbols: List[str], feed: str):
    streamer = AlpacaBarStreamer(symbols=symbols, feed=feed)

    # 在后台线程跑 asyncio
    def _runner():
        asyncio.run(streamer.run())

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    return streamer


st.set_page_config(page_title="US Stocks Minute Bars (Local)", layout="wide")
st.title("美股分钟数据（本地实时）")

load_dotenv()
default_symbols = os.getenv("SYMBOLS", "AAPL,MSFT")
default_feed = os.getenv("ALPACA_FEED", "iex")

with st.sidebar:
    symbols_text = st.text_input("Symbols (comma separated)", default_symbols)
    options = ["iex", "sip", "test"]
    idx = options.index(default_feed) if default_feed in options else 0
    feed = st.selectbox("Alpaca Feed", options=options, index=idx)
    st.caption("如果订阅权限不够，sip 可能会报错，先用 iex。")

symbols = parse_symbols(symbols_text)

# Alpaca 也有 test stream（v2/test），但它需要改 URL（本示例先用 iex/sip 的常规）
streamer = start_streamer(symbols, feed)

# 自动刷新
st.caption("页面每 2 秒刷新一次。")


latest = streamer.latest
rows = []
hints = []
for sym in symbols:
    bar = latest.get(sym)
    if bar:
        rows.append(
            {
                "symbol": bar.symbol,
                "timestamp": bar.timestamp,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
        )
        hints.append(make_hint(bar))

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("最新分钟K线")
    if rows:
        df = pd.DataFrame(rows).sort_values(["symbol"])
        st.dataframe(df, width="stretch")
    else:
        st.info("还没收到数据（可能市场休市 / Key 不对 / feed 权限不够）。")

with col2:
    st.subheader("提示")
    if hints:
        for h in hints:
            st.write("• " + h)
    else:
        st.write("等待数据中…")

st.subheader("K线图（分钟）")

symbol_for_chart = st.selectbox("选择股票", options=symbols, index=0 if symbols else 0)
bars_count = st.slider("显示最近多少根K线", min_value=50, max_value=500, value=200, step=50)

df_hist = load_recent_bars("data.duckdb", symbol_for_chart, bars_count)

if df_hist.empty:
    st.info("还没有历史数据可画（先等 WebSocket 进来几分钟）。")
else:
    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df_hist["timestamp"],
            open=df_hist["open"],
            high=df_hist["high"],
            low=df_hist["low"],
            close=df_hist["close"],
            name="OHLC",
        )
    )

    fig.add_trace(
        go.Bar(
            x=df_hist["timestamp"],
            y=df_hist["volume"],
            name="Volume",
            opacity=0.3,
            yaxis="y2",
        )
    )

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=600,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis=dict(title="Price"),
        yaxis2=dict(
            title="Volume",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        legend=dict(orientation="h"),
    )

    st.plotly_chart(fig, width="stretch")

time.sleep(2)
st.rerun()