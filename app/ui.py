import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import asyncio
import os
import threading
import time
from typing import List, Tuple, Optional, Dict

import numpy as np
import duckdb
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from ingest.alpaca_stream import AlpacaBarStreamer, Bar


# -----------------------------
# Constants (hidden)
# -----------------------------
NY_TZ = "America/New_York"
INITIAL_CAPITAL = 10000.0  # hidden: 初始本金
DCA_N = 20  # 固定：永远分20次定投

# 移除：无 / 移动止损 / 天量突破
STRATEGY_OPTIONS = [
    "全仓买入并持有",
    "分期定投(20次均匀)",
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
    "唐奇安通道(20周突破)",
    "上周高低点(周K突破)",
    "三阳买两阴卖(形态跟踪)",
    "半仓小网格(5%间距)",
    "半仓大网格(10%间距)",
    "BBI上穿下穿(收盘确认)",
    "BBI回踩不破做多(顺势二次上车)",
    "BBI回踩不破+斜率过滤",
    "BBI跌破反抽不过做空/卖出",
]


# -----------------------------
# Helpers
# -----------------------------
def parse_symbols(raw: str) -> List[str]:
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


def make_hint(bar: Bar) -> str:
    direction = "上涨" if bar.close >= bar.open else "下跌"
    rng = (bar.high - bar.low) if (bar.high is not None and bar.low is not None) else 0.0
    return f"{bar.symbol} {direction},本分钟振幅 {rng:.4f},成交量 {bar.volume:.0f}"


def is_crypto_symbol(symbol: str) -> bool:
    s = symbol.upper()
    return s.endswith("-USD") or (s.endswith("USD") and "-" in s)


def _to_utc_timestamp(s: pd.Series) -> pd.Series:
    ts = pd.to_datetime(s, errors="coerce")
    if getattr(ts.dt, "tz", None) is not None:
        return ts.dt.tz_convert("UTC")
    return ts.dt.tz_localize(NY_TZ, ambiguous="infer", nonexistent="shift_forward").dt.tz_convert("UTC")


def _utc_to_ny(ts_utc: pd.Series) -> pd.Series:
    ts = pd.to_datetime(ts_utc, errors="coerce", utc=True)
    return ts.dt.tz_convert(NY_TZ)


def _rangebreaks_for_us_market(timeframe: str, is_crypto: bool):
    if is_crypto:
        return []
    if timeframe == "1d":
        return [dict(bounds=["sat", "mon"])]
    return [dict(bounds=["sat", "mon"]), dict(bounds=[16, 9.5], pattern="hour")]


def _build_time_strings(df: pd.DataFrame, timeframe: str) -> pd.Series:
    ts_local = _utc_to_ny(df["timestamp_utc"])
    fmt = "%Y-%m-%d" if timeframe == "1d" else "%Y-%m-%d %H:%M"
    return ts_local.dt.strftime(fmt)


def _day_start_positions_idx(df: pd.DataFrame, timeframe: str) -> List[int]:
    if timeframe == "1d":
        return []
    ts_local = _utc_to_ny(df["timestamp_utc"])
    dates = ts_local.dt.date
    start_mask = dates.ne(dates.shift())
    return df.index[start_mask].tolist()


def _tick_for_nogap(df: pd.DataFrame, timeframe: str) -> Tuple[List[int], List[str]]:
    n = len(df)
    if n == 0:
        return [], []

    ts_local = _utc_to_ny(df["timestamp_utc"])

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


def _col_int_signal(df: pd.DataFrame, col: str) -> np.ndarray:
    s = pd.to_numeric(df[col], errors="coerce")
    return s.fillna(0).astype(int).to_numpy()


def _bool_shift_prev(x: pd.Series) -> pd.Series:
    s = x.shift(1)
    return s.where(s.notna(), False).astype(bool)


def _bbi_series(close: pd.Series) -> pd.Series:
    ma3 = close.rolling(3).mean()
    ma6 = close.rolling(6).mean()
    ma12 = close.rolling(12).mean()
    ma24 = close.rolling(24).mean()
    return (ma3 + ma6 + ma12 + ma24) / 4


def _cross_above_intrabar(prev_high: pd.Series, high: pd.Series, prev_thr: pd.Series, thr: pd.Series) -> pd.Series:
    return (prev_high <= prev_thr) & (high > thr)


def _cross_below_intrabar(prev_low: pd.Series, low: pd.Series, prev_thr: pd.Series, thr: pd.Series) -> pd.Series:
    return (prev_low >= prev_thr) & (low < thr)


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

    if "BBI" in ind_list:
        out["bbi"] = _bbi_series(close)

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
        tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        out["atr"] = tr.rolling(int(atr_n)).mean()

    return out


def _apply_grid_signals_intrabar(out: pd.DataFrame, high: pd.Series, low: pd.Series, grid_pct: float) -> None:
    if len(high) == 0:
        return
    ref_price = float(out["close"].astype(float).iloc[0])
    next_buy = ref_price * (1.0 - grid_pct)
    next_sell = ref_price * (1.0 + grid_pct)

    for i in range(1, len(out)):
        hi = float(high.iloc[i])
        lo = float(low.iloc[i])

        hit_buy = np.isfinite(lo) and lo <= next_buy
        hit_sell = np.isfinite(hi) and hi >= next_sell
        if hit_buy and hit_sell:
            continue

        if hit_buy:
            out.at[i, "strat_signal"] = 1
            out.at[i, "strat_exec_px"] = float(next_buy)
            ref_price = float(next_buy)
        elif hit_sell:
            out.at[i, "strat_signal"] = -1
            out.at[i, "strat_exec_px"] = float(next_sell)
            ref_price = float(next_sell)
        else:
            continue

        next_buy = ref_price * (1.0 - grid_pct)
        next_sell = ref_price * (1.0 + grid_pct)


def _build_strategy_signals(df: pd.DataFrame, strategy: str, **kwargs) -> pd.DataFrame:
    out = df.copy()
    out["strat_signal"] = 0
    out["strat_exec_px"] = np.nan

    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    open_ = out["open"].astype(float)

    prev_high = high.shift(1)
    prev_low = low.shift(1)

    if strategy == "5/20均线金叉死叉":
        ma5 = close.rolling(5).mean()
        ma20 = close.rolling(20).mean()
        buy = _cross_above_intrabar(prev_high, high, ma20.shift(1), ma20) & (ma5 > ma20)
        sell = _cross_below_intrabar(prev_low, low, ma20.shift(1), ma20) & (ma5 < ma20)
        out.loc[buy, "strat_signal"] = 1
        out.loc[buy, "strat_exec_px"] = ma20
        out.loc[sell, "strat_signal"] = -1
        out.loc[sell, "strat_exec_px"] = ma20

    elif strategy == "10/30均线双线波段":
        ma10 = close.rolling(10).mean()
        ma30 = close.rolling(30).mean()
        buy = _cross_above_intrabar(prev_high, high, ma30.shift(1), ma30) & (ma10 > ma30)
        sell = _cross_below_intrabar(prev_low, low, ma30.shift(1), ma30) & (ma10 < ma30)
        out.loc[buy, "strat_signal"] = 1
        out.loc[buy, "strat_exec_px"] = ma30
        out.loc[sell, "strat_signal"] = -1
        out.loc[sell, "strat_exec_px"] = ma30

    elif strategy == "20/60均线长线趋势":
        ma20 = close.rolling(20).mean()
        ma60 = close.rolling(60).mean()
        buy = _cross_above_intrabar(prev_high, high, ma60.shift(1), ma60) & (ma20 > ma60)
        sell = _cross_below_intrabar(prev_low, low, ma60.shift(1), ma60) & (ma20 < ma60)
        out.loc[buy, "strat_signal"] = 1
        out.loc[buy, "strat_exec_px"] = ma60
        out.loc[sell, "strat_signal"] = -1
        out.loc[sell, "strat_exec_px"] = ma60

    elif strategy == "MACD零轴战法":
        mf = kwargs.get("macd_fast", 12)
        ms = kwargs.get("macd_slow", 26)
        ef = close.ewm(span=int(mf), adjust=False).mean()
        es = close.ewm(span=int(ms), adjust=False).mean()
        macd = ef - es
        buy = (macd.shift(1) <= 0) & (macd > 0)
        sell = (macd.shift(1) >= 0) & (macd < 0)
        out.loc[buy, "strat_signal"] = 1
        out.loc[buy, "strat_exec_px"] = close
        out.loc[sell, "strat_signal"] = -1
        out.loc[sell, "strat_exec_px"] = close

    elif strategy == "MACD信号线红绿柱":
        mf = kwargs.get("macd_fast", 12)
        ms = kwargs.get("macd_slow", 26)
        msig = kwargs.get("macd_signal", 9)
        ef = close.ewm(span=int(mf), adjust=False).mean()
        es = close.ewm(span=int(ms), adjust=False).mean()
        macd = ef - es
        sig = macd.ewm(span=int(msig), adjust=False).mean()
        hist = macd - sig
        buy = (hist.shift(1) <= 0) & (hist > 0)
        sell = (hist.shift(1) >= 0) & (hist < 0)
        out.loc[buy, "strat_signal"] = 1
        out.loc[buy, "strat_exec_px"] = close
        out.loc[sell, "strat_signal"] = -1
        out.loc[sell, "strat_exec_px"] = close

    elif strategy == "RSI震荡战法(60买40卖)":
        rsi_n = int(kwargs.get("rsi_n", 14))
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.rolling(rsi_n).mean()
        avg_loss = loss.rolling(rsi_n).mean()
        rs = avg_gain / avg_loss.replace(0, pd.NA)
        rsi = 100 - (100 / (1 + rs))
        buy = (rsi.shift(1) < 60) & (rsi >= 60)
        sell = (rsi.shift(1) > 40) & (rsi <= 40)
        out.loc[buy, "strat_signal"] = 1
        out.loc[buy, "strat_exec_px"] = close
        out.loc[sell, "strat_signal"] = -1
        out.loc[sell, "strat_exec_px"] = close

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
        buy = (J.shift(1) <= 0) & (J > 0)
        sell = (J.shift(1) >= 100) & (J < 100)
        out.loc[buy, "strat_signal"] = 1
        out.loc[buy, "strat_exec_px"] = close
        out.loc[sell, "strat_signal"] = -1
        out.loc[sell, "strat_exec_px"] = close

    elif strategy == "布林带突破(上下轨)":
        bb_n = int(kwargs.get("bb_n", 20))
        bb_k = float(kwargs.get("bb_k", 2.0))
        mid = close.rolling(bb_n).mean()
        sd = close.rolling(bb_n).std(ddof=0)
        bb_up = mid + bb_k * sd
        bb_dn = mid - bb_k * sd
        buy = _cross_above_intrabar(prev_high, high, bb_up.shift(1), bb_up)
        sell = _cross_below_intrabar(prev_low, low, bb_dn.shift(1), bb_dn)
        out.loc[buy, "strat_signal"] = 1
        out.loc[buy, "strat_exec_px"] = bb_up
        out.loc[sell, "strat_signal"] = -1
        out.loc[sell, "strat_exec_px"] = bb_dn

    elif strategy == "布林带均值回归(探底回升)":
        bb_n = int(kwargs.get("bb_n", 20))
        bb_k = float(kwargs.get("bb_k", 2.0))
        mid = close.rolling(bb_n).mean()
        sd = close.rolling(bb_n).std(ddof=0)
        bb_dn = mid - bb_k * sd
        buy = _cross_below_intrabar(prev_low, low, bb_dn.shift(1), bb_dn)
        sell = _cross_above_intrabar(prev_high, high, mid.shift(1), mid)
        out.loc[buy, "strat_signal"] = 1
        out.loc[buy, "strat_exec_px"] = bb_dn
        out.loc[sell, "strat_signal"] = -1
        out.loc[sell, "strat_exec_px"] = mid

    elif strategy == "CCI顺势指标(±100突破)":
        cci_n = int(kwargs.get("cci_n", 20))
        tp = (high + low + close) / 3
        tp_ma = tp.rolling(cci_n).mean()
        mean_dev = tp.rolling(cci_n).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        cci = (tp - tp_ma) / (0.015 * mean_dev.replace(0, pd.NA))
        buy = (cci.shift(1) <= 100) & (cci > 100)
        sell = (cci.shift(1) >= -100) & (cci < -100)
        out.loc[buy, "strat_signal"] = 1
        out.loc[buy, "strat_exec_px"] = close
        out.loc[sell, "strat_signal"] = -1
        out.loc[sell, "strat_exec_px"] = close

    elif strategy == "唐奇安通道(20周突破)":
        dc_n = int(kwargs.get("donchian_n", 100))
        dc_upper = high.rolling(dc_n).max().shift(1)
        dc_lower = low.rolling(dc_n).min().shift(1)
        buy = (high > dc_upper) & (prev_high <= dc_upper.shift(1))
        sell = (low < dc_lower) & (prev_low >= dc_lower.shift(1))
        out.loc[buy, "strat_signal"] = 1
        out.loc[buy, "strat_exec_px"] = dc_upper
        out.loc[sell, "strat_signal"] = -1
        out.loc[sell, "strat_exec_px"] = dc_lower

    elif strategy == "上周高低点(周K突破)":
        week_n = int(kwargs.get("week_n", 5))
        prev_high_lvl = high.rolling(week_n).max().shift(week_n)
        prev_low_lvl = low.rolling(week_n).min().shift(week_n)
        buy = _cross_above_intrabar(prev_high, high, prev_high_lvl.shift(1), prev_high_lvl)
        sell = _cross_below_intrabar(prev_low, low, prev_low_lvl.shift(1), prev_low_lvl)
        out.loc[buy, "strat_signal"] = 1
        out.loc[buy, "strat_exec_px"] = prev_high_lvl
        out.loc[sell, "strat_signal"] = -1
        out.loc[sell, "strat_exec_px"] = prev_low_lvl

    elif strategy == "三阳买两阴卖(形态跟踪)":
        green = (close > open_).astype(int)
        red = (close < open_).astype(int)
        three_yang = (green == 1) & (green.shift(1) == 1) & (green.shift(2) == 1)
        two_yin = (red == 1) & (red.shift(1) == 1)
        out.loc[three_yang, "strat_signal"] = 1
        out.loc[three_yang, "strat_exec_px"] = close
        out.loc[two_yin, "strat_signal"] = -1
        out.loc[two_yin, "strat_exec_px"] = close

    elif strategy == "半仓小网格(5%间距)":
        _apply_grid_signals_intrabar(out, high, low, 0.05)

    elif strategy == "半仓大网格(10%间距)":
        _apply_grid_signals_intrabar(out, high, low, 0.10)

    elif strategy == "BBI上穿下穿(收盘确认)":
        bbi = out["bbi"] if "bbi" in out.columns else _bbi_series(close)
        buy = _cross_above_intrabar(prev_high, high, bbi.shift(1), bbi)
        sell = _cross_below_intrabar(prev_low, low, bbi.shift(1), bbi)
        out.loc[buy, "strat_signal"] = 1
        out.loc[buy, "strat_exec_px"] = bbi
        out.loc[sell, "strat_signal"] = -1
        out.loc[sell, "strat_exec_px"] = bbi

    elif strategy == "BBI回踩不破做多(顺势二次上车)":
        bbi = out["bbi"] if "bbi" in out.columns else _bbi_series(close)
        eps = float(kwargs.get("bbi_eps", 0.002))
        breakout = bool(kwargs.get("bbi_breakout", True))

        trend = close > bbi
        touch = low <= bbi * (1.0 + eps)
        hold_ = close >= bbi
        setup = trend & touch & hold_

        if breakout:
            setup_prev = _bool_shift_prev(setup)
            confirm_now = high > high.shift(1)
            buy = setup_prev & confirm_now
            out.loc[buy, "strat_signal"] = 1
            out.loc[buy, "strat_exec_px"] = high.shift(1)
        else:
            buy = setup
            out.loc[buy, "strat_signal"] = 1
            out.loc[buy, "strat_exec_px"] = bbi

        sell = _cross_below_intrabar(prev_low, low, bbi.shift(1), bbi)
        out.loc[sell, "strat_signal"] = -1
        out.loc[sell, "strat_exec_px"] = bbi

    elif strategy == "BBI回踩不破+斜率过滤":
        bbi = out["bbi"] if "bbi" in out.columns else _bbi_series(close)
        eps = float(kwargs.get("bbi_eps", 0.002))
        breakout = bool(kwargs.get("bbi_breakout", True))

        slope_up = bbi > bbi.shift(1)
        trend = (close > bbi) & slope_up
        touch = low <= bbi * (1.0 + eps)
        hold_ = close >= bbi
        setup = trend & touch & hold_

        if breakout:
            setup_prev = _bool_shift_prev(setup)
            confirm_now = high > high.shift(1)
            buy = setup_prev & confirm_now
            out.loc[buy, "strat_signal"] = 1
            out.loc[buy, "strat_exec_px"] = high.shift(1)
        else:
            buy = setup
            out.loc[buy, "strat_signal"] = 1
            out.loc[buy, "strat_exec_px"] = bbi

        sell = _cross_below_intrabar(prev_low, low, bbi.shift(1), bbi)
        out.loc[sell, "strat_signal"] = -1
        out.loc[sell, "strat_exec_px"] = bbi

    elif strategy == "BBI跌破反抽不过做空/卖出":
        bbi = out["bbi"] if "bbi" in out.columns else _bbi_series(close)
        eps = float(kwargs.get("bbi_fail_eps", 0.002))

        below = close < bbi
        retest = high >= bbi * (1.0 - eps)
        fail = close <= bbi
        setup = below & retest & fail
        out.loc[setup, "strat_signal"] = -1
        out.loc[setup, "strat_exec_px"] = bbi * (1.0 - eps)

        buy = _cross_above_intrabar(prev_high, high, bbi.shift(1), bbi)
        out.loc[buy, "strat_signal"] = 1
        out.loc[buy, "strat_exec_px"] = bbi

    return out


# -----------------------------
# Balance / Win-rate (NO forced sell; mark-to-market)
# -----------------------------
def _simulate_signal_strategy_mark_to_market(
    df: pd.DataFrame,
    signal_col: str = "strat_signal",
    exec_col: str = "strat_exec_px",
    leverage: float = 1.0,
    initial_capital: float = INITIAL_CAPITAL,
) -> Tuple[int, int, float]:
    if df is None or df.empty or signal_col not in df.columns or exec_col not in df.columns:
        return 0, 0, float(initial_capital)

    L = float(leverage)
    if not np.isfinite(L) or L < 1.0:
        L = 1.0

    sig = _col_int_signal(df, signal_col)
    px = pd.to_numeric(df[exec_col], errors="coerce").to_numpy(dtype=float)
    close_last = float(pd.to_numeric(df["close"], errors="coerce").iloc[-1])
    if not np.isfinite(close_last) or close_last <= 0:
        close_last = np.nan

    pos = 0
    entry_px: Optional[float] = None
    wins = 0
    trades = 0
    equity = float(initial_capital)

    for i in range(len(df)):
        if equity <= 0:
            equity = 0.0
            pos = 0
            entry_px = None
            break

        s = int(sig[i])
        if s == 0:
            continue

        p = float(px[i]) if np.isfinite(px[i]) else np.nan
        if not np.isfinite(p) or p <= 0:
            continue

        if pos == 0:
            pos = 1 if s > 0 else -1
            entry_px = p
            continue

        if (pos == 1 and s < 0) or (pos == -1 and s > 0):
            exit_px = p
            if entry_px is None or not np.isfinite(entry_px) or entry_px <= 0:
                pos = 1 if s > 0 else -1
                entry_px = p
                continue

            trade_ret = (exit_px / entry_px - 1.0) * pos
            trades += 1
            if trade_ret > 0:
                wins += 1

            equity = equity * (1.0 + L * trade_ret)
            if not np.isfinite(equity) or equity < 0:
                equity = 0.0
                pos = 0
                entry_px = None
                break

            pos = 1 if s > 0 else -1
            entry_px = p

    if equity > 0 and pos != 0 and entry_px is not None and np.isfinite(entry_px) and entry_px > 0:
        if np.isfinite(close_last) and close_last > 0:
            unreal_ret = (close_last / entry_px - 1.0) * pos
            equity = equity * (1.0 + L * unreal_ret)
            if not np.isfinite(equity) or equity < 0:
                equity = 0.0

    return wins, trades, float(equity)


def _simulate_buy_hold_balance(df: pd.DataFrame, leverage: float = 1.0, initial_capital: float = INITIAL_CAPITAL) -> Tuple[int, int, float]:
    if df is None or df.empty:
        return 0, 0, float(initial_capital)

    L = float(leverage)
    if not np.isfinite(L) or L < 1.0:
        L = 1.0

    entry_px = float(pd.to_numeric(df["open"], errors="coerce").iloc[0])
    if not np.isfinite(entry_px) or entry_px <= 0:
        entry_px = float(pd.to_numeric(df["close"], errors="coerce").iloc[0])
    if not np.isfinite(entry_px) or entry_px <= 0:
        return 0, 0, float(initial_capital)

    last_px = float(pd.to_numeric(df["close"], errors="coerce").iloc[-1])
    if not np.isfinite(last_px) or last_px <= 0:
        last_px = entry_px

    debt = initial_capital * (L - 1.0)
    shares = (initial_capital * L) / entry_px
    equity = shares * last_px - debt
    if not np.isfinite(equity) or equity < 0:
        equity = 0.0
    return 0, 0, float(equity)


def _dca_indices(nbars: int, n_tranches: int = DCA_N) -> List[int]:
    if nbars <= 0:
        return []
    if n_tranches <= 1:
        return [0]
    if nbars == 1:
        return [0]
    idx = [int(round(k * (nbars - 1) / (n_tranches - 1))) for k in range(n_tranches)]
    out = []
    prev = None
    for i in idx:
        i = max(0, min(nbars - 1, i))
        if prev is None or i != prev:
            out.append(i)
        prev = i
    return out


def _simulate_dca_20_balance(df: pd.DataFrame, leverage: float = 1.0, initial_capital: float = INITIAL_CAPITAL) -> Tuple[int, int, float]:
    if df is None or df.empty:
        return 0, 0, float(initial_capital)

    L = float(leverage)
    if not np.isfinite(L) or L < 1.0:
        L = 1.0

    nbars = len(df)
    idxs = _dca_indices(nbars, DCA_N)
    if len(idxs) == 0:
        return 0, 0, float(initial_capital)

    tranche = initial_capital / float(DCA_N)
    cash = float(initial_capital)
    debt = 0.0
    shares = 0.0

    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=float)

    for i in idxs:
        if cash <= 0:
            cash = 0.0
            break

        px = float(close[i]) if np.isfinite(close[i]) else np.nan
        if not np.isfinite(px) or px <= 0:
            continue

        own = min(tranche, cash)
        exposure = own * L
        borrow = exposure - own
        shares += exposure / px
        cash -= own
        debt += borrow

        equity_now = cash + shares * px - debt
        if not np.isfinite(equity_now) or equity_now < 0:
            return 0, 0, 0.0

    last_px = float(close[-1]) if np.isfinite(close[-1]) and close[-1] > 0 else np.nan
    if not np.isfinite(last_px):
        last_px = float(close[idxs[-1]]) if np.isfinite(close[idxs[-1]]) else 0.0

    equity = cash + shares * last_px - debt
    if not np.isfinite(equity) or equity < 0:
        equity = 0.0
    return 0, 0, float(equity)


def _months_label_from_window(df_window: pd.DataFrame) -> int:
    if df_window is None or df_window.empty or "timestamp_utc" not in df_window.columns:
        return 1
    ts = pd.to_datetime(df_window["timestamp_utc"], errors="coerce", utc=True).dropna()
    if ts.empty:
        return 1
    days = int((ts.max() - ts.min()).days)
    m = int(round(days / 30.0))
    return max(1, m)


def _fmt_int_cell(x) -> str:
    if x is None:
        return ""
    try:
        if isinstance(x, str):
            return x
        v = float(x)
        if not np.isfinite(v):
            return ""
        return str(int(round(v)))
    except Exception:
        return ""


def _span_label(df_window: pd.DataFrame) -> str:
    """
    用 calendar time span 给窗口做标签（更适合 1m 数据）：
      >=60d -> Xm
      >=1d  -> Xd
      >=1h  -> Xh
      else  -> Xbars
    """
    if df_window is None or df_window.empty or "timestamp_utc" not in df_window.columns:
        return "0bars"
    ts = pd.to_datetime(df_window["timestamp_utc"], errors="coerce", utc=True).dropna()
    if ts.empty:
        return "0bars"

    delta = ts.max() - ts.min()
    days = int(delta.days)
    hours = int(delta.total_seconds() // 3600)

    if days >= 60:
        m = max(1, int(round(days / 30.0)))
        return f"{m}m"
    if days >= 1:
        return f"{days}d"
    if hours >= 1:
        return f"{hours}h"
    return f"{len(ts)}bars"


def _indicator_and_one_price(
    df: pd.DataFrame,
    signal_col: str = "strat_signal",
    exec_col: str = "strat_exec_px",
) -> Tuple[str, str]:
    """
    indicator 逻辑（你之前定的）：
      current_px < last_buy_px  => buy
      current_px > last_sell_px => sell
      else hold
    price 只显示最终 indicator 对应的那个价位（整数显示）
    """
    if df is None or df.empty or signal_col not in df.columns or exec_col not in df.columns:
        return "hold", ""

    current_px = float(pd.to_numeric(df["close"], errors="coerce").iloc[-1])
    if not np.isfinite(current_px) or current_px <= 0:
        return "hold", ""

    sig = _col_int_signal(df, signal_col)
    exec_px = pd.to_numeric(df[exec_col], errors="coerce").to_numpy(dtype=float)

    buy_idx = np.where(sig == 1)[0]
    sell_idx = np.where(sig == -1)[0]
    last_buy_i = int(buy_idx[-1]) if len(buy_idx) else None
    last_sell_i = int(sell_idx[-1]) if len(sell_idx) else None

    last_buy_px = float(exec_px[last_buy_i]) if last_buy_i is not None and np.isfinite(exec_px[last_buy_i]) else None
    last_sell_px = float(exec_px[last_sell_i]) if last_sell_i is not None and np.isfinite(exec_px[last_sell_i]) else None

    cond_buy = (last_buy_px is not None) and (current_px < last_buy_px)
    cond_sell = (last_sell_px is not None) and (current_px > last_sell_px)

    if cond_buy and cond_sell:
        # 两者都触发时，取“最近那次信号”的方向
        if last_buy_i is not None and last_sell_i is not None and last_sell_i > last_buy_i:
            return "sell", _fmt_int_cell(last_sell_px)
        return "buy", _fmt_int_cell(last_buy_px)

    if cond_buy:
        return "buy", _fmt_int_cell(last_buy_px)
    if cond_sell:
        return "sell", _fmt_int_cell(last_sell_px)

    return "hold", ""


@st.cache_data(ttl=10)
def build_strategy_stats_table_3windows(
    df_full: pd.DataFrame,
    strategies: List[str],
    base_params: dict,
    leverage: float,
) -> pd.DataFrame:
    def _window_stats(df_use: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for strat in strategies:
            if strat == "全仓买入并持有":
                _, _, bal = _simulate_buy_hold_balance(df_use, leverage=leverage, initial_capital=INITIAL_CAPITAL)
                rows.append({"strategy": strat, "win_rate": "0/0 (0%)", "balance": int(round(bal))})
                continue

            if strat == "分期定投(20次均匀)":
                _, _, bal = _simulate_dca_20_balance(df_use, leverage=leverage, initial_capital=INITIAL_CAPITAL)
                rows.append({"strategy": strat, "win_rate": "0/0 (0%)", "balance": int(round(bal))})
                continue

            df_s = _build_strategy_signals(df_use, strat, **base_params)
            w, t, bal = _simulate_signal_strategy_mark_to_market(
                df_s, "strat_signal", "strat_exec_px", leverage=leverage, initial_capital=INITIAL_CAPITAL
            )
            wr_int = int(round((w / t * 100.0))) if t > 0 else 0
            rows.append({"strategy": strat, "win_rate": f"{w}/{t} ({wr_int}%)", "balance": int(round(bal))})

        return pd.DataFrame(rows)

    # 3 windows
    df360_raw = df_full.tail(360).copy().reset_index(drop=True)
    df1000_raw = df_full.tail(1000).copy().reset_index(drop=True)
    dfmax_raw = df_full.copy().reset_index(drop=True)

    span360 = _span_label(df360_raw)
    span1000 = _span_label(df1000_raw)
    spanmax = _span_label(dfmax_raw)

    df360 = _window_stats(df360_raw).rename(columns={"win_rate": f"win_{span360}_w360", "balance": f"bal_{span360}_w360"})
    df1000 = _window_stats(df1000_raw).rename(columns={"win_rate": f"win_{span1000}_w1000", "balance": f"bal_{span1000}_w1000"})
    dfmax = _window_stats(dfmax_raw).rename(columns={"win_rate": f"win_{spanmax}_wmax", "balance": f"bal_{spanmax}_wmax"})

    out = df360.merge(df1000, on="strategy", how="outer").merge(dfmax, on="strategy", how="outer")

    # indicator + one price
    indicator_map: Dict[str, str] = {}
    price_map: Dict[str, str] = {}

    for strat in strategies:
        if strat in ["全仓买入并持有", "分期定投(20次均匀)"]:
            indicator_map[strat] = "hold"
            price_map[strat] = ""
            continue

        df_s_full = _build_strategy_signals(df_full.copy(), strat, **base_params)
        ind, px = _indicator_and_one_price(df_s_full, "strat_signal", "strat_exec_px")
        indicator_map[strat] = ind
        price_map[strat] = px

    out["indicator"] = out["strategy"].map(indicator_map).fillna("hold")
    out["signal_price"] = out["strategy"].map(price_map).fillna("").map(_fmt_int_cell)

    # 所有余额列变成“无小数”的字符串
    for c in out.columns:
        if c.startswith("bal_"):
            out[c] = out[c].map(_fmt_int_cell)

    # 排序：买入持有、定投永远第1/2；其余按 wmax balance desc
    top_names = ["全仓买入并持有", "分期定投(20次均匀)"]
    top = out[out["strategy"].isin(top_names)].copy()
    rest = out[~out["strategy"].isin(top_names)].copy()

    max_bal_col = f"bal_{spanmax}_wmax"
    rest["_bal_max_num"] = pd.to_numeric(rest[max_bal_col], errors="coerce").fillna(-np.inf)
    rest = rest.sort_values("_bal_max_num", ascending=False).drop(columns=["_bal_max_num"])
    top = top.set_index("strategy").reindex(top_names).reset_index()

    out = pd.concat([top, rest], ignore_index=True)

    # 最终列顺序（不会再 KeyError，因为列名唯一）
    out = out[
        [
            "strategy",
            f"win_{span360}_w360",
            f"bal_{span360}_w360",
            f"win_{span1000}_w1000",
            f"bal_{span1000}_w1000",
            f"win_{spanmax}_wmax",
            f"bal_{spanmax}_wmax",
            "indicator",
            "signal_price",
        ]
    ]
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

    sec = tf_seconds.get(timeframe, 60)
    bucket_expr = f"to_timestamp(floor(epoch(ts)/{sec})*{sec})"

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
    return [
        dict(type="line", xref="x", yref="paper", x0=i, x1=i, y0=0, y1=1, line=dict(width=1, dash="dot"), opacity=0.25)
        for i in day_starts
    ]


def _make_shapes_day_separators_date(starts_ts: List[pd.Timestamp]) -> List[dict]:
    return [
        dict(type="line", xref="x", yref="paper", x0=x, x1=x, y0=0, y1=1, line=dict(width=1, dash="dot"), opacity=0.25)
        for x in starts_ts
    ]


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
ROOT = Path(__file__).resolve().parents[1]
db_path = str(ROOT / "market.duckdb")

default_symbols = os.getenv("SYMBOLS", "AAPL,MSFT")
default_feed = os.getenv("ALPACA_FEED", "iex")

with st.sidebar:
    st.caption(f"DB: {db_path}")

    auto_refresh = st.checkbox("自动刷新", value=False)
    refresh_sec = st.slider("刷新间隔(秒)", 1, 15, 2)

    symbols_text = st.text_input("Symbols (comma separated)", default_symbols)
    options = ["iex", "sip", "test"]
    idx = options.index(default_feed) if default_feed in options else 0
    feed = st.selectbox("Alpaca Feed", options=options, index=idx)

    timeframe = st.selectbox("周期", options=["1m", "5m", "30m", "1h", "1d"], index=2)

    x_mode = st.selectbox(
        "X轴模式",
        options=["NO_GAP（index轴彻底无gap）", "DATE_AXIS（日期轴 + rangebreaks）"],
        index=0,
    )

    show_day_separators = st.checkbox("非日线：按天分隔虚线", value=True)
    st.caption("sip 需要权限；不够就用 iex。")

    st.markdown("---")
    st.markdown("### 资金 / 杠杆")
    leverage = st.number_input("杠杆 (>=1)", min_value=1.0, max_value=20.0, value=1.0, step=0.5)

    st.markdown("---")
    st.markdown("### View / Autoscale")
    autoscale = st.checkbox("autoscale（仅显示最近90根 + y轴按窗口自动）", value=False)
    window_n = st.slider("可见窗口K线数", 30, 3000, 120, step=10)
    if autoscale:
        window_n = 90

    st.markdown("---")
    st.markdown("### Indicators")
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
    macd_fast = st.number_input("MACD fast", 2, 50, 12) if "MACD" in ind_list else 12
    macd_slow = st.number_input("MACD slow", 5, 100, 26) if "MACD" in ind_list else 26
    macd_signal = st.number_input("MACD signal", 2, 50, 9) if "MACD" in ind_list else 9
    atr_n = st.number_input("ATR n", 2, 200, 14) if "ATR" in ind_list else 14

    st.markdown("---")
    st.markdown("### 策略参数")
    selected_strategy = st.selectbox(
        "图表显示策略买卖点",
        options=[s for s in STRATEGY_OPTIONS if s not in ["全仓买入并持有", "分期定投(20次均匀)"]],
        index=0,
    )

    kdj_n = st.number_input("KDJ n", 5, 30, 9)
    cci_n = st.number_input("CCI n", 5, 100, 20)
    donchian_n = st.number_input("唐奇安通道 n(bars)", 20, 500, 100)
    week_n = st.number_input("周期bars数(周高低)", 2, 20, 5)

    bbi_eps = st.number_input("BBI回踩容忍(%)", 0.0, 2.0, 0.2, step=0.05) / 100.0
    bbi_breakout = st.checkbox("BBI确认：本根 intrabar 突破上一根高点", value=True)
    bbi_fail_eps = st.number_input("BBI反抽容忍(%)", 0.0, 2.0, 0.2, step=0.05) / 100.0

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
    st.info("没有可画的数据：确认写入 daily_bars / minute_bars,并且 DB_PATH 一致。")
    st.stop()

df_hist = df_hist.dropna(subset=["timestamp", "open", "high", "low", "close"]).reset_index(drop=True)
df_hist["timestamp_utc"] = _to_utc_timestamp(df_hist["timestamp"])

if only_completed and timeframe != "1m" and len(df_hist) > 1:
    df_hist = df_hist.iloc[:-1].reset_index(drop=True)

st.sidebar.caption(f"rows(total): {len(df_hist):,}")
st.sidebar.caption(f"range(UTC): {df_hist['timestamp_utc'].min()} → {df_hist['timestamp_utc'].max()}")

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

df_hist_win = df_hist_full.tail(int(window_n)).copy().reset_index(drop=True)
is_crypto = is_crypto_symbol(symbol_for_chart)

base_params_for_all = dict(
    macd_fast=int(macd_fast),
    macd_slow=int(macd_slow),
    macd_signal=int(macd_signal),
    rsi_n=int(rsi_n),
    bb_n=int(bb_n),
    bb_k=float(bb_k),
    kdj_n=int(kdj_n),
    cci_n=int(cci_n),
    donchian_n=int(donchian_n),
    week_n=int(week_n),
    bbi_eps=float(bbi_eps),
    bbi_breakout=bool(bbi_breakout),
    bbi_fail_eps=float(bbi_fail_eps),
)

df_sel = _build_strategy_signals(df_hist_win, selected_strategy, **base_params_for_all)

tabs = st.tabs(["Main"])
with tabs[0]:
    fig = go.Figure()
    hover_time = _build_time_strings(df_hist_win, timeframe).to_numpy()

    if x_mode.startswith("NO_GAP"):
        x = list(range(len(df_hist_win)))

        fig.add_trace(
            go.Candlestick(
                x=x,
                open=df_hist_win["open"],
                high=df_hist_win["high"],
                low=df_hist_win["low"],
                close=df_hist_win["close"],
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
                x=x,
                y=df_hist_win["volume"].astype(float),
                name="Volume",
                opacity=0.30,
                yaxis="y2",
                customdata=hover_time,
                hovertemplate="Time: %{customdata}<br>Vol: %{y}<extra></extra>",
            )
        )

        if "SMA" in ind_list and f"sma_{int(sma_n)}" in df_hist_win.columns:
            fig.add_trace(go.Scatter(x=x, y=df_hist_win[f"sma_{int(sma_n)}"], mode="lines", name=f"SMA{sma_n}"))

        if "EMA" in ind_list:
            if f"ema_{int(ema_fast)}" in df_hist_win.columns:
                fig.add_trace(go.Scatter(x=x, y=df_hist_win[f"ema_{int(ema_fast)}"], mode="lines", name=f"EMA{ema_fast}"))
            if f"ema_{int(ema_slow)}" in df_hist_win.columns:
                fig.add_trace(go.Scatter(x=x, y=df_hist_win[f"ema_{int(ema_slow)}"], mode="lines", name=f"EMA{ema_slow}"))

        if "BBANDS" in ind_list and {"bb_up", "bb_mid", "bb_dn"}.issubset(df_hist_win.columns):
            fig.add_trace(go.Scatter(x=x, y=df_hist_win["bb_up"], mode="lines", name="BB Up"))
            fig.add_trace(go.Scatter(x=x, y=df_hist_win["bb_mid"], mode="lines", name="BB Mid"))
            fig.add_trace(go.Scatter(x=x, y=df_hist_win["bb_dn"], mode="lines", name="BB Dn"))

        if "BBI" in ind_list and "bbi" in df_hist_win.columns:
            fig.add_trace(go.Scatter(x=x, y=df_hist_win["bbi"], mode="lines", name="BBI"))

        buy_idx = df_sel.index[df_sel["strat_signal"] == 1].tolist()
        sell_idx = df_sel.index[df_sel["strat_signal"] == -1].tolist()
        if buy_idx:
            fig.add_trace(
                go.Scatter(
                    x=buy_idx,
                    y=df_sel.loc[buy_idx, "strat_exec_px"],
                    mode="markers",
                    name=f"{selected_strategy} BUY",
                    marker=dict(symbol="triangle-up", size=12),
                    customdata=[hover_time[i] for i in buy_idx],
                    hovertemplate=f"{selected_strategy} BUY<br>Time: %{{customdata}}<br>Px: %{{y}}<extra></extra>",
                )
            )
        if sell_idx:
            fig.add_trace(
                go.Scatter(
                    x=sell_idx,
                    y=df_sel.loc[sell_idx, "strat_exec_px"],
                    mode="markers",
                    name=f"{selected_strategy} SELL",
                    marker=dict(symbol="triangle-down", size=12),
                    customdata=[hover_time[i] for i in sell_idx],
                    hovertemplate=f"{selected_strategy} SELL<br>Time: %{{customdata}}<br>Px: %{{y}}<extra></extra>",
                )
            )

        shapes = []
        if timeframe != "1d" and show_day_separators:
            shapes.extend(_make_shapes_day_separators_nogap(_day_start_positions_idx(df_hist_win, timeframe)))

        tick_idx, tick_text = _tick_for_nogap(df_hist_win, timeframe)

        fig.update_layout(
            uirevision=f"{symbol_for_chart}-{timeframe}-{x_mode}",
            shapes=shapes,
            xaxis=dict(type="linear", tickmode="array", tickvals=tick_idx, ticktext=tick_text, tickangle=0, rangeslider=dict(visible=False)),
            height=700,
            margin=dict(l=10, r=10, t=30, b=10),
            yaxis=dict(title="Price", fixedrange=False),
            yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False, fixedrange=False),
            legend=dict(orientation="h"),
        )

    else:
        rb = _rangebreaks_for_us_market(timeframe, is_crypto=is_crypto)
        x_ts = _utc_to_ny(df_hist_win["timestamp_utc"])

        fig.add_trace(go.Candlestick(x=x_ts, open=df_hist_win["open"], high=df_hist_win["high"], low=df_hist_win["low"], close=df_hist_win["close"], name="OHLC"))
        fig.add_trace(go.Bar(x=x_ts, y=df_hist_win["volume"].astype(float), name="Volume", opacity=0.30, yaxis="y2"))

        if "SMA" in ind_list and f"sma_{int(sma_n)}" in df_hist_win.columns:
            fig.add_trace(go.Scatter(x=x_ts, y=df_hist_win[f"sma_{int(sma_n)}"], mode="lines", name=f"SMA{sma_n}"))

        if "EMA" in ind_list:
            if f"ema_{int(ema_fast)}" in df_hist_win.columns:
                fig.add_trace(go.Scatter(x=x_ts, y=df_hist_win[f"ema_{int(ema_fast)}"], mode="lines", name=f"EMA{ema_fast}"))
            if f"ema_{int(ema_slow)}" in df_hist_win.columns:
                fig.add_trace(go.Scatter(x=x_ts, y=df_hist_win[f"ema_{int(ema_slow)}"], mode="lines", name=f"EMA{ema_slow}"))

        if "BBANDS" in ind_list and {"bb_up", "bb_mid", "bb_dn"}.issubset(df_hist_win.columns):
            fig.add_trace(go.Scatter(x=x_ts, y=df_hist_win["bb_up"], mode="lines", name="BB Up"))
            fig.add_trace(go.Scatter(x=x_ts, y=df_hist_win["bb_mid"], mode="lines", name="BB Mid"))
            fig.add_trace(go.Scatter(x=x_ts, y=df_hist_win["bb_dn"], mode="lines", name="BB Dn"))

        if "BBI" in ind_list and "bbi" in df_hist_win.columns:
            fig.add_trace(go.Scatter(x=x_ts, y=df_hist_win["bbi"], mode="lines", name="BBI"))

        buy_idx = df_sel.index[df_sel["strat_signal"] == 1].tolist()
        sell_idx = df_sel.index[df_sel["strat_signal"] == -1].tolist()
        if buy_idx:
            fig.add_trace(go.Scatter(x=x_ts.iloc[buy_idx], y=df_sel.loc[buy_idx, "strat_exec_px"], mode="markers", name=f"{selected_strategy} BUY", marker=dict(symbol="triangle-up", size=12)))
        if sell_idx:
            fig.add_trace(go.Scatter(x=x_ts.iloc[sell_idx], y=df_sel.loc[sell_idx, "strat_exec_px"], mode="markers", name=f"{selected_strategy} SELL", marker=dict(symbol="triangle-down", size=12)))

        shapes = []
        if timeframe != "1d" and show_day_separators:
            dates = x_ts.dt.date
            starts = x_ts.loc[dates.ne(dates.shift())].tolist()
            shapes.extend(_make_shapes_day_separators_date(starts))

        fig.update_layout(
            uirevision=f"{symbol_for_chart}-{timeframe}-{x_mode}",
            shapes=shapes,
            xaxis=dict(type="date", rangebreaks=rb, rangeslider=dict(visible=not autoscale)),
            height=700,
            margin=dict(l=10, r=10, t=30, b=10),
            yaxis=dict(title="Price", fixedrange=False),
            yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False, fixedrange=False),
            legend=dict(orientation="h"),
        )

    st.plotly_chart(fig, width="stretch", config={"scrollZoom": True, "displaylogo": False})

    stats_df = build_strategy_stats_table_3windows(
        df_full=df_hist_full,
        strategies=STRATEGY_OPTIONS,
        base_params=base_params_for_all,
        leverage=float(leverage),
    )

    st.markdown("### 策略历史统计（xx_months；胜率=胜/总(%)；本金余额；indicator + signal_price；全表无小数）")
    if stats_df.empty:
        st.info("没有可用统计（数据不足或策略无交易信号）。")
    else:
        row_h = 32
        height = min(2000, row_h * (len(stats_df) + 1) + 12)
        st.dataframe(stats_df, width="stretch", hide_index=True, height=height)

if auto_refresh:
    time.sleep(refresh_sec)
    st.rerun()