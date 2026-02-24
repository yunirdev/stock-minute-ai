import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import asyncio
import os
import threading
from typing import List
import time
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


@st.cache_resource
def start_streamer(symbols: List[str], feed: str, db_path: str):
    streamer = AlpacaBarStreamer(symbols=symbols, feed=feed, db_path=db_path)

    def _runner():
        asyncio.run(streamer.run())

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    return streamer


def is_crypto_symbol(symbol: str) -> bool:
    s = symbol.upper()
    return s.endswith("-USD") or (s.endswith("USD") and "-" in s)


def get_data_range(db_path: str, symbol: str, timeframe: str):
    con = duckdb.connect(db_path)
    if timeframe == "1d":
        mn, mx = con.execute(
            "SELECT min(dt), max(dt) FROM daily_bars WHERE symbol = ?",
            [symbol],
        ).fetchone()
        con.close()
        if mn is None or mx is None:
            return None, None
        return pd.Timestamp(mn), pd.Timestamp(mx)
    mn, mx = con.execute(
        "SELECT min(ts), max(ts) FROM minute_bars WHERE symbol = ?",
        [symbol],
    ).fetchone()
    con.close()
    if mn is None or mx is None:
        return None, None
    mn = pd.Timestamp(mn)
    mx = pd.Timestamp(mx)
    if mn.tzinfo is None:
        mn = mn.tz_localize("UTC")
    if mx.tzinfo is None:
        mx = mx.tz_localize("UTC")
    return mn, mx


def load_all_bars(db_path: str, symbol: str, timeframe: str) -> pd.DataFrame:
    tf_seconds = {"1m": 60, "5m": 300, "30m": 1800, "1h": 3600}
    con = duckdb.connect(db_path)

    if timeframe == "1d":
        df = con.execute(
            """
            SELECT
              symbol,
              dt::TIMESTAMP AS timestamp,
              open, high, low, close, volume
            FROM daily_bars
            WHERE symbol = ?
            ORDER BY dt ASC
            """,
            [symbol],
        ).df()
        con.close()
        return df

    sec = tf_seconds.get(timeframe, 60)
    bucket_expr = (
        f"to_timestamp(floor(epoch(ts AT TIME ZONE 'America/New_York')/{sec})*{sec}) "
        f"AT TIME ZONE 'America/New_York'"
    )

    if timeframe == "1m":
        df = con.execute(
            """
            SELECT symbol, ts as timestamp, open, high, low, close, volume
            FROM minute_bars
            WHERE symbol = ?
            ORDER BY ts ASC
            """,
            [symbol],
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
        [symbol],
    ).df()
    con.close()
    return df


def _build_x_labels(df_hist: pd.DataFrame, timeframe: str) -> pd.Series:
    ts = pd.to_datetime(df_hist["timestamp"], errors="coerce")
    if getattr(ts.dt, "tz", None) is not None:
        ts_local = ts.dt.tz_convert("America/New_York")
    else:
        ts_local = ts
    fmt = "%Y-%m-%d" if timeframe == "1d" else "%Y-%m-%d %H:%M"
    return ts_local.dt.strftime(fmt)


def _day_start_xs(df_hist: pd.DataFrame, timeframe: str) -> List[str]:
    if timeframe == "1d":
        return []
    ts = pd.to_datetime(df_hist["timestamp"], errors="coerce")
    if getattr(ts.dt, "tz", None) is not None:
        ts_local = ts.dt.tz_convert("America/New_York")
    else:
        ts_local = ts
    dates = ts_local.dt.date
    start_idx = df_hist.index[dates.ne(dates.shift())].tolist()
    x_labels = _build_x_labels(df_hist, timeframe)
    return [x_labels.loc[i] for i in start_idx]


st.set_page_config(page_title="US Stocks Minute Bars (Local)", layout="wide")
st.title("美股分钟数据（本地实时）")

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")
db_path = os.getenv("DB_PATH", "market.duckdb")

default_symbols = os.getenv("SYMBOLS", "AAPL,MSFT")
default_feed = os.getenv("ALPACA_FEED", "iex")

with st.sidebar:
    auto_refresh = st.checkbox("自动刷新", value=True)
    refresh_sec = st.slider("刷新间隔(秒)", 1, 15, 15)
    symbols_text = st.text_input("Symbols (comma separated)", default_symbols)
    options = ["iex", "sip", "test"]
    idx = options.index(default_feed) if default_feed in options else 0
    feed = st.selectbox("Alpaca Feed", options=options, index=idx)
    timeframe = st.selectbox("周期", options=["1m", "5m", "30m", "1h", "1d"], index=0)
    no_gap = st.checkbox("无 gap（按数据顺序显示，忽略所有缺失日期）", value=True)
    show_day_separators = st.checkbox("非日线：按天分隔虚线", value=True)
    st.caption("如果订阅权限不够，sip 可能会报错，先用 iex。")
    st.caption(f"DB: {db_path}")

symbols = parse_symbols(symbols_text)
if not symbols:
    st.warning("请在左侧输入至少 1 个股票代码（例如 AAPL）。")
    time.sleep(1)
    st.rerun()

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
    st.subheader("最新分钟K线")
    if rows:
        df_latest = pd.DataFrame(rows).sort_values(["symbol"])
        st.dataframe(df_latest, use_container_width=True)
    else:
        st.info("还没收到数据（可能市场休市 / Key 不对 / feed 权限不够）。")

with col2:
    st.subheader("提示")
    if hints:
        for h in hints:
            st.write("• " + h)
    else:
        st.write("等待数据中…")

st.subheader(f"K线图（{timeframe}）")
symbol_for_chart = st.selectbox("选择股票", options=symbols, index=0)
only_completed = st.checkbox("只显示已完成K线（更稳定）", value=False)

df_hist = load_all_bars(db_path, symbol_for_chart, timeframe)

if only_completed and (not df_hist.empty) and timeframe != "1m":
    df_hist = df_hist.iloc[:-1]

if df_hist.empty:
    st.info("还没有历史数据可画（确认 history.py 已写入，并且 Streamlit 使用同一个 DB_PATH）。")
else:
    df_hist = df_hist.dropna(subset=["timestamp", "open", "high", "low", "close"]).reset_index(drop=True)

    is_daily = (timeframe == "1d")
    is_crypto = is_crypto_symbol(symbol_for_chart)

    fig = go.Figure()

    if no_gap:
        x_vals = _build_x_labels(df_hist, timeframe).tolist()

        fig.add_trace(
            go.Candlestick(
                x=x_vals,
                open=df_hist["open"],
                high=df_hist["high"],
                low=df_hist["low"],
                close=df_hist["close"],
                name="OHLC",
            )
        )
        fig.add_trace(
            go.Bar(
                x=x_vals,
                y=df_hist["volume"],
                name="Volume",
                opacity=0.3,
                yaxis="y2",
            )
        )

        if (not is_daily) and show_day_separators:
            for x in _day_start_xs(df_hist, timeframe):
                fig.add_vline(x=x, line_width=1, line_dash="dot", line_color="gray", opacity=0.35)

        fig.update_layout(
            xaxis=dict(
                type="category",
                categoryorder="array",
                categoryarray=x_vals,
                rangeslider=dict(visible=False),
            ),
            height=650,
            margin=dict(l=10, r=10, t=30, b=10),
            yaxis=dict(title="Price"),
            yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False),
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "displaylogo": False})

    else:
        rangebreaks = []
        if not is_crypto:
            if is_daily:
                rangebreaks = [dict(bounds=["sat", "mon"])]
            else:
                rangebreaks = [
                    dict(bounds=["sat", "mon"]),
                    dict(bounds=[16, 9.5], pattern="hour"),
                ]

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

        if (not is_daily) and show_day_separators:
            ts = pd.to_datetime(df_hist["timestamp"])
            if getattr(ts.dt, "tz", None) is not None:
                dates = ts.dt.tz_convert("America/New_York").dt.date
            else:
                dates = ts.dt.date
            starts = df_hist.loc[dates.ne(dates.shift()), "timestamp"].tolist()
            for x in starts:
                fig.add_vline(x=x, line_width=1, line_dash="dot", line_color="gray", opacity=0.35)

        mn, mx = get_data_range(db_path, symbol_for_chart, timeframe)
        x_range = [mn, mx] if (mn is not None and mx is not None) else None

        fig.update_layout(
            xaxis=dict(
                type="date",
                rangebreaks=rangebreaks,
                rangeslider=dict(visible=True),
                rangeselector=dict(
                    buttons=[
                        dict(count=1, label="1D", step="day", stepmode="backward"),
                        dict(count=5, label="5D", step="day", stepmode="backward"),
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(step="all", label="ALL"),
                    ]
                ),
                range=x_range,
            ),
            height=650,
            margin=dict(l=10, r=10, t=30, b=10),
            yaxis=dict(title="Price"),
            yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False),
            legend=dict(orientation="h"),
        )

        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "displaylogo": False})

if auto_refresh:
    time.sleep(refresh_sec)
    st.rerun()