import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import asyncio
import os
import threading
import time
from typing import List, Tuple
from datetime import timezone

import numpy as np
import duckdb
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from ingest.alpaca_stream import AlpacaBarStreamer, Bar


# -----------------------------
# Helpers
# -----------------------------
def parse_symbols(raw: str) -> List[str]:
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


def make_hint(bar: Bar) -> str:
    direction = "上涨" if bar.close >= bar.open else "下跌"
    rng = (bar.high - bar.low) if (bar.high is not None and bar.low is not None) else 0.0
    return f"{bar.symbol} {direction}，本分钟振幅 {rng:.4f}，成交量 {bar.volume:.0f}"


def is_crypto_symbol(symbol: str) -> bool:
    s = symbol.upper()
    return s.endswith("-USD") or (s.endswith("USD") and "-" in s)


def _local_ts(ts: pd.Series) -> pd.Series:
    ts = pd.to_datetime(ts, errors="coerce")
    if getattr(ts.dt, "tz", None) is not None:
        return ts.dt.tz_convert("America/New_York")
    return ts


def _rangebreaks_for_us_market(timeframe: str, is_crypto: bool):
    if is_crypto:
        return []
    if timeframe == "1d":
        return [dict(bounds=["sat", "mon"])]
    return [
        dict(bounds=["sat", "mon"]),
        dict(bounds=[16, 9.5], pattern="hour"),
    ]


def _build_time_strings(df: pd.DataFrame, timeframe: str) -> pd.Series:
    ts_local = _local_ts(df["timestamp"])
    fmt = "%Y-%m-%d" if timeframe == "1d" else "%Y-%m-%d %H:%M"
    return ts_local.dt.strftime(fmt)


def _day_start_positions_idx(df: pd.DataFrame, timeframe: str) -> List[int]:
    if timeframe == "1d":
        return []
    ts_local = _local_ts(df["timestamp"])
    dates = ts_local.dt.date
    start_mask = dates.ne(dates.shift())
    return df.index[start_mask].tolist()


def _tick_for_nogap(df: pd.DataFrame, timeframe: str) -> Tuple[List[int], List[str]]:
    n = len(df)
    ts_local = _local_ts(df["timestamp"])
    if n == 0:
        return [], []

    if timeframe == "1d":
        step = max(n // 12, 1)
        tick_idx = list(range(0, n, step))
        tick_text = [ts_local.dt.strftime("%Y-%m-%d").iloc[i] for i in tick_idx]
        return tick_idx, tick_text

    day_change = ts_local.dt.date.ne(ts_local.dt.date.shift())
    tick_idx = df.index[day_change].tolist()
    if len(tick_idx) == 0:
        tick_idx = list(range(0, n, max(n // 10, 1)))
    tick_text = [ts_local.dt.strftime("%m-%d").iloc[i] for i in tick_idx]
    return tick_idx, tick_text


def _compute_indicators(
    df: pd.DataFrame,
    ind_list: List[str],
    sma_n: int,
    ema_fast: int,
    ema_slow: int,
    bb_n: int,
    bb_k: float,
    rsi_n: int,
    macd_fast: int,
    macd_slow: int,
    macd_signal: int,
    atr_n: int,
) -> pd.DataFrame:
    out = df.copy()

    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)

    if "SMA" in ind_list:
        out[f"sma_{sma_n}"] = close.rolling(int(sma_n)).mean()

    if "EMA" in ind_list:
        out[f"ema_{ema_fast}"] = close.ewm(span=int(ema_fast), adjust=False).mean()
        out[f"ema_{ema_slow}"] = close.ewm(span=int(ema_slow), adjust=False).mean()

    if "BBANDS" in ind_list:
        mid = close.rolling(int(bb_n)).mean()
        sd = close.rolling(int(bb_n)).std(ddof=0)
        out["bb_mid"] = mid
        out["bb_up"] = mid + float(bb_k) * sd
        out["bb_dn"] = mid - float(bb_k) * sd

    # BBI (TOS Average() == SMA)
    # BBI = (MA3 + MA6 + MA12 + MA24) / 4
    if "BBI" in ind_list:
        ma3 = close.rolling(3).mean()
        ma6 = close.rolling(6).mean()
        ma12 = close.rolling(12).mean()
        ma24 = close.rolling(24).mean()
        out["bbi"] = (ma3 + ma6 + ma12 + ma24) / 4

    if "RSI" in ind_list:
        n = int(rsi_n)
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
        avg_loss = loss.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
        rs = avg_gain / avg_loss.replace(0, pd.NA)
        out["rsi"] = 100 - (100 / (1 + rs))

    if "MACD" in ind_list:
        ef = close.ewm(span=int(macd_fast), adjust=False).mean()
        es = close.ewm(span=int(macd_slow), adjust=False).mean()
        macd = ef - es
        sig = macd.ewm(span=int(macd_signal), adjust=False).mean()
        hist = macd - sig
        out["macd"] = macd
        out["macd_signal"] = sig
        out["macd_hist"] = hist

    if "ATR" in ind_list:
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        out["atr"] = tr.rolling(int(atr_n)).mean()

    return out


def _build_signals(
    df: pd.DataFrame,
    use_signal_model: bool,
    signal_mode: str,
    ind_list: List[str],
    ema_fast: int,
    ema_slow: int,
    rsi_buy: float,
    rsi_sell: float,
) -> pd.DataFrame:
    out = df.copy()
    out["signal"] = 0

    if not use_signal_model:
        return out

    trend_ok_long = pd.Series(True, index=out.index)
    if "EMA" in ind_list and f"ema_{ema_fast}" in out.columns and f"ema_{ema_slow}" in out.columns:
        trend_ok_long = out[f"ema_{ema_fast}"] > out[f"ema_{ema_slow}"]

    if signal_mode == "Rule-based":
        entry = pd.Series(False, index=out.index)
        exit_ = pd.Series(False, index=out.index)

        if "RSI" in ind_list and "rsi" in out.columns:
            rsi = out["rsi"]
            entry |= (rsi.shift(1) < rsi_buy) & (rsi >= rsi_buy)
            exit_ |= (rsi.shift(1) > rsi_sell) & (rsi <= rsi_sell)

        if "MACD" in ind_list and "macd_hist" in out.columns:
            h = out["macd_hist"]
            entry |= (h.shift(1) < 0) & (h >= 0)
            exit_ |= (h.shift(1) > 0) & (h <= 0)

        out.loc[entry & trend_ok_long, "signal"] = 1
        out.loc[exit_, "signal"] = -1
        return out

    score = pd.Series(0, index=out.index)

    if "EMA" in ind_list and f"ema_{ema_fast}" in out.columns and f"ema_{ema_slow}" in out.columns:
        score += (out[f"ema_{ema_fast}"] > out[f"ema_{ema_slow}"]).astype(int) * 2 - 1

    if "MACD" in ind_list and "macd_hist" in out.columns:
        score += (out["macd_hist"] > 0).astype(int) * 2 - 1

    if "RSI" in ind_list and "rsi" in out.columns:
        score += (out["rsi"] < rsi_buy).astype(int)
        score -= (out["rsi"] > rsi_sell).astype(int)

    out.loc[score >= 2, "signal"] = 1
    out.loc[score <= -2, "signal"] = -1
    return out


_STRAT_BUY_OFFSET = 0.997
_STRAT_SELL_OFFSET = 1.003

STRATEGY_OPTIONS = [
    "无 (None)",
    "5/20均线金叉死叉",
    "10/30均线双线波段",
    "20/60均线长线趋势",
    "MACD零轴战法",
    "MACD信号线红绿柱",
    "RSI震荡战法(60买40卖)",
    "KDJ极值反转(J线探底)",
    "布林带突破(上下轨)",
    "布林带均值回归(探底回升)",
    "CCI顺势指标(±100突破)",
    "天量突破(2倍量上涨)",
    "唐奇安通道(20周突破)",
    "上周高低点(周K突破)",
    "三阳买两阴卖(形态跟踪)",
    "半仓定投(月投2万)",
    "半仓小网格(5%间距)",
    "半仓大网格(10%间距)",
    "半仓高频投(周投5k)",
    "移动止损(回撤20%)",
]


def _cross_above(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a.shift(1) <= b.shift(1)) & (a > b)


def _cross_below(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a.shift(1) >= b.shift(1)) & (a < b)


def _apply_grid_signals(out: pd.DataFrame, close: pd.Series, grid_pct: float) -> None:
    if len(close) == 0:
        return
    ref_price = float(close.iloc[0])
    next_buy = ref_price * (1.0 - grid_pct)
    next_sell = ref_price * (1.0 + grid_pct)
    sig_col = out.columns.get_loc("strat_signal")
    for i in range(1, len(close)):
        price = float(close.iloc[i])
        if price <= next_buy:
            out.iloc[i, sig_col] = 1
            ref_price = price
            next_buy = ref_price * (1.0 - grid_pct)
            next_sell = ref_price * (1.0 + grid_pct)
        elif price >= next_sell:
            out.iloc[i, sig_col] = -1
            ref_price = price
            next_buy = ref_price * (1.0 - grid_pct)
            next_sell = ref_price * (1.0 + grid_pct)


def _build_strategy_signals(df: pd.DataFrame, strategy: str, **kwargs) -> pd.DataFrame:
    out = df.copy()
    out["strat_signal"] = 0

    if not strategy or strategy == "无 (None)":
        return out

    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    open_ = out["open"].astype(float)
    volume = out["volume"].astype(float)

    if strategy == "5/20均线金叉死叉":
        ma5 = close.rolling(5).mean()
        ma20 = close.rolling(20).mean()
        out.loc[_cross_above(ma5, ma20), "strat_signal"] = 1
        out.loc[_cross_below(ma5, ma20), "strat_signal"] = -1

    elif strategy == "10/30均线双线波段":
        ma10 = close.rolling(10).mean()
        ma30 = close.rolling(30).mean()
        out.loc[_cross_above(ma10, ma30), "strat_signal"] = 1
        out.loc[_cross_below(ma10, ma30), "strat_signal"] = -1

    elif strategy == "20/60均线长线趋势":
        ma20 = close.rolling(20).mean()
        ma60 = close.rolling(60).mean()
        out.loc[_cross_above(ma20, ma60), "strat_signal"] = 1
        out.loc[_cross_below(ma20, ma60), "strat_signal"] = -1

    elif strategy == "MACD零轴战法":
        mf = kwargs.get("macd_fast", 12)
        ms = kwargs.get("macd_slow", 26)
        ef = close.ewm(span=int(mf), adjust=False).mean()
        es = close.ewm(span=int(ms), adjust=False).mean()
        macd = ef - es
        zero = pd.Series(0.0, index=macd.index)
        out.loc[_cross_above(macd, zero), "strat_signal"] = 1
        out.loc[_cross_below(macd, zero), "strat_signal"] = -1

    elif strategy == "MACD信号线红绿柱":
        mf = kwargs.get("macd_fast", 12)
        ms = kwargs.get("macd_slow", 26)
        msig = kwargs.get("macd_signal", 9)
        ef = close.ewm(span=int(mf), adjust=False).mean()
        es = close.ewm(span=int(ms), adjust=False).mean()
        macd = ef - es
        sig = macd.ewm(span=int(msig), adjust=False).mean()
        hist = macd - sig
        zero = pd.Series(0.0, index=hist.index)
        out.loc[_cross_above(hist, zero), "strat_signal"] = 1
        out.loc[_cross_below(hist, zero), "strat_signal"] = -1

    elif strategy == "RSI震荡战法(60买40卖)":
        rsi_n = int(kwargs.get("rsi_n", 14))
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.rolling(rsi_n).mean()
        avg_loss = loss.rolling(rsi_n).mean()
        rs = avg_gain / avg_loss.replace(0, pd.NA)
        rsi = 100 - (100 / (1 + rs))
        lvl60 = pd.Series(60.0, index=rsi.index)
        lvl40 = pd.Series(40.0, index=rsi.index)
        out.loc[_cross_above(rsi, lvl60), "strat_signal"] = 1
        out.loc[_cross_below(rsi, lvl40), "strat_signal"] = -1

    elif strategy == "KDJ极值反转(J线探底)":
        kdj_n = int(kwargs.get("kdj_n", 9))
        low_n = low.rolling(kdj_n).min()
        high_n = high.rolling(kdj_n).max()
        denom = (high_n - low_n).replace(0, pd.NA)
        rsv = (close - low_n) / denom * 100
        rsv = rsv.fillna(50.0)
        K = rsv.ewm(com=2, adjust=False).mean()
        D = K.ewm(com=2, adjust=False).mean()
        J = 3 * K - 2 * D
        zero = pd.Series(0.0, index=J.index)
        hundred = pd.Series(100.0, index=J.index)
        out.loc[_cross_above(J, zero), "strat_signal"] = 1
        out.loc[_cross_below(J, hundred), "strat_signal"] = -1

    elif strategy == "布林带突破(上下轨)":
        bb_n = int(kwargs.get("bb_n", 20))
        bb_k = float(kwargs.get("bb_k", 2.0))
        mid = close.rolling(bb_n).mean()
        sd = close.rolling(bb_n).std(ddof=0)
        bb_up = mid + bb_k * sd
        bb_dn = mid - bb_k * sd
        out.loc[_cross_above(close, bb_up), "strat_signal"] = 1
        out.loc[_cross_below(close, bb_dn), "strat_signal"] = -1

    elif strategy == "布林带均值回归(探底回升)":
        bb_n = int(kwargs.get("bb_n", 20))
        bb_k = float(kwargs.get("bb_k", 2.0))
        mid = close.rolling(bb_n).mean()
        sd = close.rolling(bb_n).std(ddof=0)
        bb_dn = mid - bb_k * sd
        out.loc[_cross_above(close, bb_dn), "strat_signal"] = 1
        out.loc[_cross_below(close, mid), "strat_signal"] = -1

    elif strategy == "CCI顺势指标(±100突破)":
        cci_n = int(kwargs.get("cci_n", 20))
        tp = (high + low + close) / 3
        tp_ma = tp.rolling(cci_n).mean()
        mean_dev = tp.rolling(cci_n).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True
        )
        cci = (tp - tp_ma) / (0.015 * mean_dev.replace(0, pd.NA))
        p100 = pd.Series(100.0, index=cci.index)
        n100 = pd.Series(-100.0, index=cci.index)
        out.loc[_cross_above(cci, p100), "strat_signal"] = 1
        out.loc[_cross_below(cci, n100), "strat_signal"] = -1

    elif strategy == "天量突破(2倍量上涨)":
        vol_ma_n = int(kwargs.get("vol_ma_n", 20))
        vol_ma = volume.rolling(vol_ma_n).mean()
        surge = (volume > 2 * vol_ma) & (close > open_)
        out.loc[surge, "strat_signal"] = 1

    elif strategy == "唐奇安通道(20周突破)":
        dc_n = int(kwargs.get("donchian_n", 100))
        dc_upper = high.rolling(dc_n).max().shift(1)
        dc_lower = low.rolling(dc_n).min().shift(1)
        out.loc[close > dc_upper, "strat_signal"] = 1
        out.loc[close < dc_lower, "strat_signal"] = -1

    elif strategy == "上周高低点(周K突破)":
        week_n = int(kwargs.get("week_n", 5))
        prev_high = high.rolling(week_n).max().shift(week_n)
        prev_low = low.rolling(week_n).min().shift(week_n)
        out.loc[close > prev_high, "strat_signal"] = 1
        out.loc[close < prev_low, "strat_signal"] = -1

    elif strategy == "三阳买两阴卖(形态跟踪)":
        green = (close > open_).astype(int)
        red = (close < open_).astype(int)
        three_yang = (green == 1) & (green.shift(1) == 1) & (green.shift(2) == 1)
        two_yin = (red == 1) & (red.shift(1) == 1)
        out.loc[three_yang, "strat_signal"] = 1
        out.loc[two_yin, "strat_signal"] = -1

    elif strategy == "半仓定投(月投2万)":
        period = int(kwargs.get("dca_period", 20))
        buy_idx = list(range(0, len(out), period))
        sig_col = out.columns.get_loc("strat_signal")
        for i in buy_idx:
            out.iloc[i, sig_col] = 1

    elif strategy == "半仓小网格(5%间距)":
        _apply_grid_signals(out, close, 0.05)

    elif strategy == "半仓大网格(10%间距)":
        _apply_grid_signals(out, close, 0.10)

    elif strategy == "半仓高频投(周投5k)":
        period = int(kwargs.get("dca_period", 5))
        buy_idx = list(range(0, len(out), period))
        sig_col = out.columns.get_loc("strat_signal")
        for i in buy_idx:
            out.iloc[i, sig_col] = 1

    elif strategy == "移动止损(回撤20%)":
        trail_pct = float(kwargs.get("trail_pct", 0.20))
        peak = close.cummax()
        drawdown = (close - peak) / peak
        out.loc[drawdown < -trail_pct, "strat_signal"] = -1

    return out


@st.cache_data(ttl=2)
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
            SELECT
              symbol,
              ts AS timestamp,
              open, high, low, close, volume
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


def _make_shapes_day_separators_nogap(day_starts: List[int]) -> List[dict]:
    shapes = []
    for i in day_starts:
        shapes.append(
            dict(
                type="line",
                xref="x",
                yref="paper",
                x0=i,
                x1=i,
                y0=0,
                y1=1,
                line=dict(width=1, dash="dot"),
                opacity=0.25,
            )
        )
    return shapes


def _make_shapes_day_separators_date(starts_ts: List[pd.Timestamp]) -> List[dict]:
    shapes = []
    for x in starts_ts:
        shapes.append(
            dict(
                type="line",
                xref="x",
                yref="paper",
                x0=x,
                x1=x,
                y0=0,
                y1=1,
                line=dict(width=1, dash="dot"),
                opacity=0.25,
            )
        )
    return shapes


@st.cache_resource
def start_streamer(symbols: List[str], feed: str, db_path: str):
    streamer = AlpacaBarStreamer(symbols=symbols, feed=feed, db_path=db_path)

    def _runner():
        asyncio.run(streamer.run())

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    return streamer


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="US Stocks Bars (DuckDB)", layout="wide")
st.title("K线（DuckDB + Alpaca 实时）")

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")
db_path = os.getenv("DB_PATH", "market.duckdb")

default_symbols = os.getenv("SYMBOLS", "AAPL,MSFT")
default_feed = os.getenv("ALPACA_FEED", "iex")

with st.sidebar:
    st.caption(f"DB: {db_path}")

    auto_refresh = st.checkbox("自动刷新", value=True)
    refresh_sec = st.slider("刷新间隔(秒)", 1, 15, 2)

    symbols_text = st.text_input("Symbols (comma separated)", default_symbols)
    options = ["iex", "sip", "test"]
    idx = options.index(default_feed) if default_feed in options else 0
    feed = st.selectbox("Alpaca Feed", options=options, index=idx)

    timeframe = st.selectbox("周期", options=["1m", "5m", "30m", "1h", "1d"], index=0)

    x_mode = st.selectbox(
        "X轴模式",
        options=["NO_GAP（index轴彻底无gap）", "DATE_AXIS（日期轴 + rangebreaks）"],
        index=0,
    )

    show_day_separators = st.checkbox("非日线：按天分隔虚线", value=True)
    st.caption("sip 需要权限；不够就用 iex。")

    st.markdown("---")
    st.markdown("### View / Autoscale")
    autoscale = st.checkbox("autoscale（仅显示最近90根 + y轴按窗口自动）", value=True)
    window_n = st.slider("可见窗口K线数", 30, 3000, 90, step=10)
    if autoscale:
        window_n = 90

    st.markdown("---")
    st.markdown("### Indicators / Signals")
    ind_list = st.multiselect(
        "叠加指标",
        ["SMA", "EMA", "BBANDS", "BBI", "RSI", "MACD", "ATR"],
        default=["EMA", "BBI"],
    )

    sma_n = st.number_input("SMA n", 5, 200, 20) if "SMA" in ind_list else 20
    ema_fast = st.number_input("EMA fast", 2, 200, 12) if "EMA" in ind_list else 12
    ema_slow = st.number_input("EMA slow", 2, 400, 26) if "EMA" in ind_list else 26

    bb_n = st.number_input("BB n", 5, 200, 20) if "BBANDS" in ind_list else 20
    bb_k = st.number_input("BB k", 1.0, 5.0, 2.0, step=0.5) if "BBANDS" in ind_list else 2.0

    rsi_n = st.number_input("RSI n", 2, 200, 14) if "RSI" in ind_list else 14
    rsi_buy = st.number_input("RSI buy <= (cross up)", 5, 50, 30) if "RSI" in ind_list else 30
    rsi_sell = st.number_input("RSI sell >= (cross down)", 50, 95, 70) if "RSI" in ind_list else 70

    macd_fast = st.number_input("MACD fast", 2, 50, 12) if "MACD" in ind_list else 12
    macd_slow = st.number_input("MACD slow", 5, 100, 26) if "MACD" in ind_list else 26
    macd_signal = st.number_input("MACD signal", 2, 50, 9) if "MACD" in ind_list else 9

    atr_n = st.number_input("ATR n", 2, 200, 14) if "ATR" in ind_list else 14

    use_signal_model = st.checkbox("启用指示器模型（给买卖点）", value=False)
    signal_mode = (
        st.selectbox("模型类型", ["Rule-based", "Score-based"], index=0) if use_signal_model else "Rule-based"
    )

    show_rsi_panel = st.checkbox("显示 RSI 面板", value=("RSI" in ind_list))
    show_macd_panel = st.checkbox("显示 MACD 面板", value=("MACD" in ind_list))

    st.markdown("---")
    st.markdown("### 策略系统")
    selected_strategy = st.selectbox("选择策略", options=STRATEGY_OPTIONS, index=0)

    # Strategy-specific parameters
    kdj_n = 9
    cci_n = 20
    donchian_n = 100
    vol_ma_n = 20
    week_n = 5
    trail_pct = 0.20
    dca_period_monthly = 20
    dca_period_weekly = 5

    if selected_strategy == "KDJ极值反转(J线探底)":
        kdj_n = st.number_input("KDJ n", 5, 30, 9)
    elif selected_strategy == "CCI顺势指标(±100突破)":
        cci_n = st.number_input("CCI n", 5, 100, 20)
    elif selected_strategy == "唐奇安通道(20周突破)":
        donchian_n = st.number_input("唐奇安通道 n(bars)", 20, 500, 100)
    elif selected_strategy == "天量突破(2倍量上涨)":
        vol_ma_n = st.number_input("量均线 n", 5, 100, 20)
    elif selected_strategy == "上周高低点(周K突破)":
        week_n = st.number_input("周期bars数", 2, 20, 5)
    elif selected_strategy == "移动止损(回撤20%)":
        trail_pct = st.number_input("回撤阈值(%)", 5, 50, 20) / 100.0
    elif selected_strategy == "半仓定投(月投2万)":
        dca_period_monthly = st.number_input("定投周期(bars)", 5, 100, 20)
    elif selected_strategy == "半仓高频投(周投5k)":
        dca_period_weekly = st.number_input("高频投周期(bars)", 1, 30, 5)

    show_kdj_panel = st.checkbox("显示 KDJ 面板", value=(selected_strategy == "KDJ极值反转(J线探底)"))
    show_cci_panel = st.checkbox("显示 CCI 面板", value=(selected_strategy == "CCI顺势指标(±100突破)"))

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
    st.subheader("最新分钟bar（stream）")
    if rows:
        st.dataframe(pd.DataFrame(rows).sort_values(["symbol"]), width="stretch")
    else:
        st.info("还没收到数据（可能休市 / Key 不对 / feed 权限不足）。")

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
if df_hist.empty:
    st.info("没有可画的数据：确认写入 daily_bars / minute_bars，并且 DB_PATH 一致。")
    st.stop()

df_hist = df_hist.dropna(subset=["timestamp", "open", "high", "low", "close"]).reset_index(drop=True)
df_hist["timestamp"] = pd.to_datetime(df_hist["timestamp"], utc=True)
df_hist["timestamp"] = df_hist["timestamp"].dt.tz_convert("America/New_York")
df_hist = df_hist.dropna(subset=["timestamp"]).reset_index(drop=True)

if only_completed and timeframe != "1m" and len(df_hist) > 1:
    df_hist = df_hist.iloc[:-1].reset_index(drop=True)

st.sidebar.caption(f"rows(total): {len(df_hist):,}")
st.sidebar.caption(f"range: {df_hist['timestamp'].min()} → {df_hist['timestamp'].max()}")

# -----------------------------
# Compute indicators on FULL history first (stable rolling/ewm),
# then slice view window for rendering.
# -----------------------------
df_hist_full = df_hist.copy().reset_index(drop=True)

df_hist_full = _compute_indicators(
    df=df_hist_full,
    ind_list=ind_list,
    sma_n=int(sma_n),
    ema_fast=int(ema_fast),
    ema_slow=int(ema_slow),
    bb_n=int(bb_n),
    bb_k=float(bb_k),
    rsi_n=int(rsi_n),
    macd_fast=int(macd_fast),
    macd_slow=int(macd_slow),
    macd_signal=int(macd_signal),
    atr_n=int(atr_n),
)
df_hist_full = _build_signals(
    df=df_hist_full,
    use_signal_model=use_signal_model,
    signal_mode=signal_mode,
    ind_list=ind_list,
    ema_fast=int(ema_fast),
    ema_slow=int(ema_slow),
    rsi_buy=float(rsi_buy),
    rsi_sell=float(rsi_sell),
)
df_hist_full = _build_strategy_signals(
    df_hist_full,
    selected_strategy,
    macd_fast=int(macd_fast),
    macd_slow=int(macd_slow),
    macd_signal=int(macd_signal),
    rsi_n=int(rsi_n),
    bb_n=int(bb_n),
    bb_k=float(bb_k),
    kdj_n=int(kdj_n),
    cci_n=int(cci_n),
    donchian_n=int(donchian_n),
    vol_ma_n=int(vol_ma_n),
    week_n=int(week_n),
    trail_pct=float(trail_pct),
    dca_period=int(dca_period_monthly) if selected_strategy == "半仓定投(月投2万)" else int(dca_period_weekly),
)

df_hist = df_hist_full.tail(int(window_n)).copy().reset_index(drop=True)
is_crypto = is_crypto_symbol(symbol_for_chart)

tab_names = ["Main"]
if show_rsi_panel and "RSI" in ind_list:
    tab_names.append("RSI")
if show_macd_panel and "MACD" in ind_list:
    tab_names.append("MACD")
if show_kdj_panel:
    tab_names.append("KDJ")
if show_cci_panel:
    tab_names.append("CCI")
tabs = st.tabs(tab_names)

# -----------------------------
# Main chart
# -----------------------------
with tabs[0]:
    fig = go.Figure()

    if x_mode.startswith("NO_GAP"):
        n = len(df_hist)
        x_idx = list(range(n))

        hover_time = _build_time_strings(df_hist, timeframe).to_numpy()

        fig.add_trace(
            go.Candlestick(
                x=x_idx,
                open=df_hist["open"],
                high=df_hist["high"],
                low=df_hist["low"],
                close=df_hist["close"],
                name="OHLC",
                customdata=hover_time,
                hovertemplate=(
                    "Time: %{customdata}<br>"
                    "O: %{open}<br>H: %{high}<br>L: %{low}<br>C: %{close}<extra></extra>"
                ),
            )
        )

        fig.add_trace(
            go.Bar(
                x=x_idx,
                y=df_hist["volume"].astype(float),
                name="Volume",
                opacity=0.30,
                yaxis="y2",
                customdata=hover_time,
                hovertemplate="Time: %{customdata}<br>Vol: %{y}<extra></extra>",
            )
        )

        # overlays
        if "SMA" in ind_list and f"sma_{int(sma_n)}" in df_hist.columns:
            fig.add_trace(go.Scatter(x=x_idx, y=df_hist[f"sma_{int(sma_n)}"], mode="lines", name=f"SMA{sma_n}"))

        if "EMA" in ind_list:
            if f"ema_{int(ema_fast)}" in df_hist.columns:
                fig.add_trace(
                    go.Scatter(x=x_idx, y=df_hist[f"ema_{int(ema_fast)}"], mode="lines", name=f"EMA{ema_fast}")
                )
            if f"ema_{int(ema_slow)}" in df_hist.columns:
                fig.add_trace(
                    go.Scatter(x=x_idx, y=df_hist[f"ema_{int(ema_slow)}"], mode="lines", name=f"EMA{ema_slow}")
                )

        if "BBANDS" in ind_list and {"bb_up", "bb_mid", "bb_dn"}.issubset(df_hist.columns):
            fig.add_trace(go.Scatter(x=x_idx, y=df_hist["bb_up"], mode="lines", name="BB Up"))
            fig.add_trace(go.Scatter(x=x_idx, y=df_hist["bb_mid"], mode="lines", name="BB Mid"))
            fig.add_trace(go.Scatter(x=x_idx, y=df_hist["bb_dn"], mode="lines", name="BB Dn"))

        if "BBI" in ind_list and "bbi" in df_hist.columns:
            fig.add_trace(go.Scatter(x=x_idx, y=df_hist["bbi"], mode="lines", name="BBI"))

        # buy/sell markers (existing signal model)
        if use_signal_model and "signal" in df_hist.columns:
            buys = df_hist.index[df_hist["signal"] == 1].tolist()
            sells = df_hist.index[df_hist["signal"] == -1].tolist()

            if buys:
                fig.add_trace(
                    go.Scatter(
                        x=buys,
                        y=(df_hist.loc[buys, "low"].astype(float) * 0.999),
                        mode="markers",
                        name="BUY",
                        marker=dict(symbol="triangle-up", size=10),
                        customdata=hover_time[buys],
                        hovertemplate="BUY<br>Time: %{customdata}<extra></extra>",
                    )
                )
            if sells:
                fig.add_trace(
                    go.Scatter(
                        x=sells,
                        y=(df_hist.loc[sells, "high"].astype(float) * 1.001),
                        mode="markers",
                        name="SELL",
                        marker=dict(symbol="triangle-down", size=10),
                        customdata=hover_time[sells],
                        hovertemplate="SELL<br>Time: %{customdata}<extra></extra>",
                    )
                )

        # strategy signal markers
        if selected_strategy != "无 (None)" and "strat_signal" in df_hist.columns:
            s_buys = df_hist.index[df_hist["strat_signal"] == 1].tolist()
            s_sells = df_hist.index[df_hist["strat_signal"] == -1].tolist()
            if s_buys:
                fig.add_trace(
                    go.Scatter(
                        x=s_buys,
                        y=(df_hist.loc[s_buys, "low"].astype(float) * _STRAT_BUY_OFFSET),
                        mode="markers",
                        name=f"策略买 ({selected_strategy})",
                        marker=dict(symbol="triangle-up", size=12, color="orange"),
                        customdata=hover_time[s_buys],
                        hovertemplate="策略买<br>Time: %{customdata}<extra></extra>",
                    )
                )
            if s_sells:
                fig.add_trace(
                    go.Scatter(
                        x=s_sells,
                        y=(df_hist.loc[s_sells, "high"].astype(float) * _STRAT_SELL_OFFSET),
                        mode="markers",
                        name=f"策略卖 ({selected_strategy})",
                        marker=dict(symbol="triangle-down", size=12, color="purple"),
                        customdata=hover_time[s_sells],
                        hovertemplate="策略卖<br>Time: %{customdata}<extra></extra>",
                    )
                )

        # day separator shapes (batch)
        shapes = []
        if timeframe != "1d" and show_day_separators:
            day_starts = _day_start_positions_idx(df_hist, timeframe)
            shapes.extend(_make_shapes_day_separators_nogap(day_starts))

        tick_idx, tick_text = _tick_for_nogap(df_hist, timeframe)

        fig.update_layout(
            uirevision=f"{symbol_for_chart}-{timeframe}-{x_mode}",
            shapes=shapes,
            xaxis=dict(
                type="linear",
                tickmode="array",
                tickvals=tick_idx,
                ticktext=tick_text,
                tickangle=0,
                rangeslider=dict(visible=False),
            ),
            height=700,
            margin=dict(l=10, r=10, t=30, b=10),
            yaxis=dict(title="Price", fixedrange=False),
            yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False, fixedrange=False),
            legend=dict(orientation="h"),
        )

    else:
        rb = _rangebreaks_for_us_market(timeframe, is_crypto=is_crypto)

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
                y=df_hist["volume"].astype(float),
                name="Volume",
                opacity=0.30,
                yaxis="y2",
            )
        )

        if "SMA" in ind_list and f"sma_{int(sma_n)}" in df_hist.columns:
            fig.add_trace(
                go.Scatter(x=df_hist["timestamp"], y=df_hist[f"sma_{int(sma_n)}"], mode="lines", name=f"SMA{sma_n}")
            )

        if "EMA" in ind_list:
            if f"ema_{int(ema_fast)}" in df_hist.columns:
                fig.add_trace(
                    go.Scatter(x=df_hist["timestamp"], y=df_hist[f"ema_{int(ema_fast)}"], mode="lines", name=f"EMA{ema_fast}")
                )
            if f"ema_{int(ema_slow)}" in df_hist.columns:
                fig.add_trace(
                    go.Scatter(x=df_hist["timestamp"], y=df_hist[f"ema_{int(ema_slow)}"], mode="lines", name=f"EMA{ema_slow}")
                )

        if "BBANDS" in ind_list and {"bb_up", "bb_mid", "bb_dn"}.issubset(df_hist.columns):
            fig.add_trace(go.Scatter(x=df_hist["timestamp"], y=df_hist["bb_up"], mode="lines", name="BB Up"))
            fig.add_trace(go.Scatter(x=df_hist["timestamp"], y=df_hist["bb_mid"], mode="lines", name="BB Mid"))
            fig.add_trace(go.Scatter(x=df_hist["timestamp"], y=df_hist["bb_dn"], mode="lines", name="BB Dn"))

        if "BBI" in ind_list and "bbi" in df_hist.columns:
            fig.add_trace(go.Scatter(x=df_hist["timestamp"], y=df_hist["bbi"], mode="lines", name="BBI"))

        if use_signal_model and "signal" in df_hist.columns:
            buys = df_hist.index[df_hist["signal"] == 1].tolist()
            sells = df_hist.index[df_hist["signal"] == -1].tolist()
            if buys:
                fig.add_trace(
                    go.Scatter(
                        x=df_hist.loc[buys, "timestamp"],
                        y=(df_hist.loc[buys, "low"].astype(float) * 0.999),
                        mode="markers",
                        name="BUY",
                        marker=dict(symbol="triangle-up", size=10),
                    )
                )
            if sells:
                fig.add_trace(
                    go.Scatter(
                        x=df_hist.loc[sells, "timestamp"],
                        y=(df_hist.loc[sells, "high"].astype(float) * 1.001),
                        mode="markers",
                        name="SELL",
                        marker=dict(symbol="triangle-down", size=10),
                    )
                )

        # strategy signal markers
        if selected_strategy != "无 (None)" and "strat_signal" in df_hist.columns:
            s_buys = df_hist.index[df_hist["strat_signal"] == 1].tolist()
            s_sells = df_hist.index[df_hist["strat_signal"] == -1].tolist()
            if s_buys:
                fig.add_trace(
                    go.Scatter(
                        x=df_hist.loc[s_buys, "timestamp"],
                        y=(df_hist.loc[s_buys, "low"].astype(float) * _STRAT_BUY_OFFSET),
                        mode="markers",
                        name=f"策略买 ({selected_strategy})",
                        marker=dict(symbol="triangle-up", size=12, color="orange"),
                    )
                )
            if s_sells:
                fig.add_trace(
                    go.Scatter(
                        x=df_hist.loc[s_sells, "timestamp"],
                        y=(df_hist.loc[s_sells, "high"].astype(float) * _STRAT_SELL_OFFSET),
                        mode="markers",
                        name=f"策略卖 ({selected_strategy})",
                        marker=dict(symbol="triangle-down", size=12, color="purple"),
                    )
                )

        shapes = []
        if timeframe != "1d" and show_day_separators:
            ts_local = _local_ts(df_hist["timestamp"])
            dates = ts_local.dt.date
            starts = df_hist.loc[dates.ne(dates.shift()), "timestamp"].tolist()
            shapes.extend(_make_shapes_day_separators_date(starts))

        xmin = df_hist["timestamp"].min()
        xmax = df_hist["timestamp"].max()
        if getattr(xmin, "tzinfo", None) is None:
            xmin = xmin.replace(tzinfo=timezone.utc)
        if getattr(xmax, "tzinfo", None) is None:
            xmax = xmax.replace(tzinfo=timezone.utc)

        fig.update_layout(
            uirevision=f"{symbol_for_chart}-{timeframe}-{x_mode}",
            shapes=shapes,
            xaxis=dict(
                type="date",
                rangebreaks=rb,
                rangeslider=dict(visible=not autoscale),
            ),
            height=700,
            margin=dict(l=10, r=10, t=30, b=10),
            yaxis=dict(title="Price", fixedrange=False),
            yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False, fixedrange=False),
            legend=dict(orientation="h"),
        )

    st.plotly_chart(
        fig,
        width="stretch",
        config={"scrollZoom": True, "displaylogo": False},
    )
    st.caption("BBI = (SMA3 + SMA6 + SMA12 + SMA24) / 4；autoscale 时默认仅显示最近 90 根。")


# -----------------------------
# RSI panel
# -----------------------------
tab_offset = 1
if show_rsi_panel and "RSI" in ind_list:
    with tabs[tab_offset]:
        if "rsi" not in df_hist.columns:
            st.info("RSI 未计算（检查是否勾选 RSI）。")
        else:
            fig_rsi = go.Figure()
            if x_mode.startswith("NO_GAP"):
                x = list(range(len(df_hist)))
                hover_time = _build_time_strings(df_hist, timeframe).to_numpy()
                fig_rsi.add_trace(
                    go.Scatter(
                        x=x,
                        y=df_hist["rsi"],
                        mode="lines",
                        name="RSI",
                        customdata=hover_time,
                        hovertemplate="Time: %{customdata}<br>RSI: %{y:.2f}<extra></extra>",
                    )
                )
                fig_rsi.update_layout(
                    xaxis=dict(type="linear", rangeslider=dict(visible=False)),
                    yaxis=dict(range=[0, 100]),
                    height=300,
                    margin=dict(l=10, r=10, t=30, b=10),
                )
            else:
                fig_rsi.add_trace(go.Scatter(x=df_hist["timestamp"], y=df_hist["rsi"], mode="lines", name="RSI"))
                fig_rsi.update_layout(
                    xaxis=dict(type="date", rangeslider=dict(visible=False)),
                    yaxis=dict(range=[0, 100]),
                    height=300,
                    margin=dict(l=10, r=10, t=30, b=10),
                )
            st.plotly_chart(fig_rsi, width="stretch", config={"scrollZoom": True, "displaylogo": False})
    tab_offset += 1


# -----------------------------
# MACD panel
# -----------------------------
if show_macd_panel and "MACD" in ind_list:
    with tabs[tab_offset]:
        need_cols = {"macd", "macd_signal", "macd_hist"}
        if not need_cols.issubset(df_hist.columns):
            st.info("MACD 未计算（检查是否勾选 MACD）。")
        else:
            fig_macd = go.Figure()
            if x_mode.startswith("NO_GAP"):
                x = list(range(len(df_hist)))
                fig_macd.add_trace(go.Bar(x=x, y=df_hist["macd_hist"], name="Hist", opacity=0.5))
                fig_macd.add_trace(go.Scatter(x=x, y=df_hist["macd"], mode="lines", name="MACD"))
                fig_macd.add_trace(go.Scatter(x=x, y=df_hist["macd_signal"], mode="lines", name="Signal"))
                fig_macd.update_layout(
                    xaxis=dict(type="linear", rangeslider=dict(visible=False)),
                    height=320,
                    margin=dict(l=10, r=10, t=30, b=10),
                )
            else:
                fig_macd.add_trace(go.Bar(x=df_hist["timestamp"], y=df_hist["macd_hist"], name="Hist", opacity=0.5))
                fig_macd.add_trace(go.Scatter(x=df_hist["timestamp"], y=df_hist["macd"], mode="lines", name="MACD"))
                fig_macd.add_trace(
                    go.Scatter(x=df_hist["timestamp"], y=df_hist["macd_signal"], mode="lines", name="Signal")
                )
                fig_macd.update_layout(
                    xaxis=dict(type="date", rangeslider=dict(visible=False)),
                    height=320,
                    margin=dict(l=10, r=10, t=30, b=10),
                )
            st.plotly_chart(fig_macd, width="stretch", config={"scrollZoom": True, "displaylogo": False})
    tab_offset += 1

# -----------------------------
# KDJ panel
# -----------------------------
if show_kdj_panel:
    with tabs[tab_offset]:
        kdj_n_val = int(kdj_n)
        close_s = df_hist["close"].astype(float)
        high_s = df_hist["high"].astype(float)
        low_s = df_hist["low"].astype(float)
        low_n_s = low_s.rolling(kdj_n_val).min()
        high_n_s = high_s.rolling(kdj_n_val).max()
        denom_s = (high_n_s - low_n_s).replace(0, pd.NA)
        rsv_s = (close_s - low_n_s) / denom_s * 100
        rsv_s = rsv_s.fillna(50.0)
        K_s = rsv_s.ewm(com=2, adjust=False).mean()
        D_s = K_s.ewm(com=2, adjust=False).mean()
        J_s = 3 * K_s - 2 * D_s

        fig_kdj = go.Figure()
        if x_mode.startswith("NO_GAP"):
            x = list(range(len(df_hist)))
            fig_kdj.add_trace(go.Scatter(x=x, y=K_s, mode="lines", name="K"))
            fig_kdj.add_trace(go.Scatter(x=x, y=D_s, mode="lines", name="D"))
            fig_kdj.add_trace(go.Scatter(x=x, y=J_s, mode="lines", name="J"))
            fig_kdj.update_layout(
                xaxis=dict(type="linear", rangeslider=dict(visible=False)),
                height=320,
                margin=dict(l=10, r=10, t=30, b=10),
            )
        else:
            fig_kdj.add_trace(go.Scatter(x=df_hist["timestamp"], y=K_s, mode="lines", name="K"))
            fig_kdj.add_trace(go.Scatter(x=df_hist["timestamp"], y=D_s, mode="lines", name="D"))
            fig_kdj.add_trace(go.Scatter(x=df_hist["timestamp"], y=J_s, mode="lines", name="J"))
            fig_kdj.update_layout(
                xaxis=dict(type="date", rangeslider=dict(visible=False)),
                height=320,
                margin=dict(l=10, r=10, t=30, b=10),
            )
        fig_kdj.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)
        fig_kdj.add_hline(y=100, line_dash="dash", line_color="red", opacity=0.5)
        st.plotly_chart(fig_kdj, width="stretch", config={"scrollZoom": True, "displaylogo": False})
    tab_offset += 1

# -----------------------------
# CCI panel
# -----------------------------
if show_cci_panel:
    with tabs[tab_offset]:
        cci_n_val = int(cci_n)
        close_c = df_hist["close"].astype(float)
        high_c = df_hist["high"].astype(float)
        low_c = df_hist["low"].astype(float)
        tp_c = (high_c + low_c + close_c) / 3
        tp_ma_c = tp_c.rolling(cci_n_val).mean()
        mean_dev_c = tp_c.rolling(cci_n_val).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True
        )
        cci_vals = (tp_c - tp_ma_c) / (0.015 * mean_dev_c.replace(0, pd.NA))

        fig_cci = go.Figure()
        if x_mode.startswith("NO_GAP"):
            x = list(range(len(df_hist)))
            fig_cci.add_trace(go.Scatter(x=x, y=cci_vals, mode="lines", name="CCI"))
            fig_cci.update_layout(
                xaxis=dict(type="linear", rangeslider=dict(visible=False)),
                height=300,
                margin=dict(l=10, r=10, t=30, b=10),
            )
        else:
            fig_cci.add_trace(go.Scatter(x=df_hist["timestamp"], y=cci_vals, mode="lines", name="CCI"))
            fig_cci.update_layout(
                xaxis=dict(type="date", rangeslider=dict(visible=False)),
                height=300,
                margin=dict(l=10, r=10, t=30, b=10),
            )
        fig_cci.add_hline(y=100, line_dash="dash", line_color="green", opacity=0.5)
        fig_cci.add_hline(y=-100, line_dash="dash", line_color="red", opacity=0.5)
        st.plotly_chart(fig_cci, width="stretch", config={"scrollZoom": True, "displaylogo": False})

# -----------------------------
# Auto refresh
# -----------------------------
if auto_refresh:
    time.sleep(refresh_sec)
    st.rerun()