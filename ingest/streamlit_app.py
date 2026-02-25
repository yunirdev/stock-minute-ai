import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import asyncio
import os
import threading
import time
from typing import List
from datetime import datetime, timedelta, timezone

import duckdb
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from ingest.alpaca_stream import AlpacaBarStreamer, Bar


def parse_symbols(raw: str) -> List[str]:
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


def make_hint(bar: Bar) -> str:
    direction = "上涨" if bar.close >= bar.open else "下跌"
    rng = (bar.high - bar.low) if (bar.high is not None and bar.low is not None) else 0.0
    return f"{bar.symbol} {direction}，本分钟振幅 {rng:.4f}，成交量 {bar.volume:.0f}"


def ensure_schema(db_path: str) -> None:
    con = duckdb.connect(db_path)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS minute_bars (
            symbol TEXT,
            ts TIMESTAMPTZ,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume BIGINT,
            source TEXT,
            PRIMARY KEY(symbol, ts, source)
        );
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS daily_bars (
            symbol TEXT,
            dt DATE,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume BIGINT,
            source TEXT,
            PRIMARY KEY(symbol, dt, source)
        );
        """
    )
    con.close()


@st.cache_resource
def start_streamer(symbols: List[str], feed: str, db_path: str):
    streamer = AlpacaBarStreamer(symbols=symbols, feed=feed, db_path=db_path)

    def _runner():
        asyncio.run(streamer.run())

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    return streamer


def get_max_history_days(db_path: str, symbol: str, timeframe: str) -> int:
    con = duckdb.connect(db_path)
    if timeframe == "1d":
        result = con.execute(
            "SELECT min(dt), max(dt) FROM daily_bars WHERE symbol = ?",
            [symbol],
        ).fetchone()
        con.close()
        if result is None:
            return 1
        mn, mx = result
        if mn is None or mx is None:
            return 1
        return max((mx - mn).days + 1, 1)

    result = con.execute(
        "SELECT min(ts), max(ts) FROM minute_bars WHERE symbol = ?",
        [symbol],
    ).fetchone()
    con.close()
    if result is None:
        return 1
    mn, mx = result
    if mn is None or mx is None:
        return 1
    if getattr(mn, "tzinfo", None) is None:
        mn = mn.replace(tzinfo=timezone.utc)
    if getattr(mx, "tzinfo", None) is None:
        mx = mx.replace(tzinfo=timezone.utc)
    return max((mx - mn).days + 1, 1)

@st.cache_data(ttl=2)
def load_bars_days(db_path: str, symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    tf_seconds = {"1m": 60, "5m": 300, "30m": 1800, "1h": 3600}
    con = duckdb.connect(db_path)

    if timeframe == "1d":
        cutoff_dt = (datetime.now(timezone.utc) - timedelta(days=days)).date()
        df = con.execute(
            """
            SELECT
              symbol,
              dt::TIMESTAMP AS timestamp,
              open, high, low, close, volume
            FROM daily_bars
            WHERE symbol = ?
              AND dt >= ?
            ORDER BY dt ASC
            """,
            [symbol, cutoff_dt],
        ).df()
        con.close()
        return df

    sec = tf_seconds.get(timeframe, 60)
    bucket_expr = (
        f"to_timestamp(floor(epoch(ts AT TIME ZONE 'America/New_York')/{sec})*{sec}) "
        f"AT TIME ZONE 'America/New_York'"
    )
    cutoff_ts = datetime.now(timezone.utc) - timedelta(days=days)

    if timeframe == "1m":
        df = con.execute(
            """
            SELECT symbol, ts as timestamp, open, high, low, close, volume
            FROM minute_bars
            WHERE symbol = ?
              AND ts >= ?
            ORDER BY ts ASC
            """,
            [symbol, cutoff_ts],
        ).df()
        con.close()
        return df

    df = con.execute(
        f"""
        WITH b AS (
          SELECT
            symbol,
            {bucket_expr} AS bucket_ts,
            ts, open, high, low, close, volume
          FROM minute_bars
          WHERE symbol = ?
            AND ts >= ?
        )
        SELECT
          symbol,
          bucket_ts AS timestamp,
          arg_min(open, ts)  AS open,
          max(high)          AS high,
          min(low)           AS low,
          arg_max(close, ts) AS close,
          sum(volume)        AS volume
        FROM b
        GROUP BY symbol, bucket_ts
        ORDER BY bucket_ts ASC
        """,
        [symbol, cutoff_ts],
    ).df()
    con.close()
    return df


st.set_page_config(page_title="US Stocks Bars (DuckDB)", layout="wide")
st.title("美股K线（DuckDB + Alpaca 实时）")

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

db_path = os.getenv("DB_PATH", "market.duckdb")
ensure_schema(db_path)

default_symbols = os.getenv("SYMBOLS", "AAPL,MSFT")
default_feed = os.getenv("ALPACA_FEED", "iex")

timeframe = st.selectbox("周期", options=["1m", "5m", "30m", "1h", "1d"], index=4)
st.subheader(f"K线图（{timeframe}）")

with st.sidebar:
    st.caption(f"DB: {db_path}")
    auto_refresh = st.checkbox("自动刷新", value=True)
    refresh_sec = st.slider("刷新间隔(秒)", 1, 15, 2)
    symbols_text = st.text_input("Symbols (comma separated)", default_symbols)
    options = ["iex", "sip", "test"]
    idx = options.index(default_feed) if default_feed in options else 0
    feed = st.selectbox("Alpaca Feed", options=options, index=idx)
    st.caption("如果订阅权限不够，sip 可能会报错，先用 iex。")
    load_all = st.checkbox("加载全量数据（可能很慢）", value=True)

symbols = parse_symbols(symbols_text)
if not symbols:
    st.warning("请在左侧输入至少 1 个股票代码（例如 AAPL）。")
    st.stop()

streamer = start_streamer(symbols, feed, db_path)

latest = streamer.latest
rows, hints = [], []
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
    st.subheader("最新分钟bar（Alpaca stream）")
    if rows:
        st.dataframe(pd.DataFrame(rows).sort_values(["symbol"]), use_container_width=True)
    else:
        st.info("还没收到数据（可能休市/Key不对/feed权限不足）。")

with col2:
    st.subheader("提示")
    if hints:
        for h in hints:
            st.write("• " + h)
    else:
        st.write("等待数据中…")

symbol_for_chart = st.selectbox("选择股票", options=symbols, index=0)
only_completed = st.checkbox("只显示已完成K线（更稳定）", value=False)

max_days = get_max_history_days(db_path, symbol_for_chart, timeframe)
if not load_all and timeframe in ["1m", "5m", "30m", "1h"]:
    max_days = min(max_days, 90)

df_hist = load_bars_days(db_path, symbol_for_chart, timeframe, max_days)

st.sidebar.caption(f"rows: {len(df_hist):,}")
if not df_hist.empty:
    st.sidebar.caption(f"range: {df_hist['timestamp'].min()} → {df_hist['timestamp'].max()}")

if only_completed and (not df_hist.empty) and timeframe != "1m":
    df_hist = df_hist.iloc[:-1]

if df_hist.empty:
    st.info("没有可画的数据：确认 history.py 已写入 daily_bars / minute_bars，并且 DB_PATH 一致。")
else:
    xmin = df_hist["timestamp"].min()
    xmax = df_hist["timestamp"].max()

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
        xaxis=dict(
            uirevision=f"{symbol_for_chart}-{timeframe}",
            type="date",
            rangeslider=dict(visible=True),
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(step="all", label="ALL"),
                ]
            ),
        ),
        height=650,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis=dict(title="Price"),
        yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h"),
    )

    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

if auto_refresh:
    time.sleep(refresh_sec)
    st.rerun()