"""
strategy_core.py
Pure strategy signal functions — no Streamlit dependencies.
Replicates the logic from app/ui.py so this module is self-contained.

Each strategy receives a DataFrame with columns:
    timestamp_utc, open, high, low, close, volume

Returns the same DataFrame with two new columns:
    strat_signal  : int  (+1 buy, -1 sell, 0 hold)
    strat_exec_px : float  (suggested execution price, NaN on hold)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import ta

# ---------------------------------------------------------------------------
# Available strategies (mirrors STRATEGY_OPTIONS in app/ui.py)
# ---------------------------------------------------------------------------
STRATEGY_OPTIONS = [
    "全仓买入并持有",
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
    "ADX趋势过滤(ta)",
    "Stoch超买超卖(ta)",
    "Williams %R反转(ta)",
    "MFI量价共振(ta)",
]

# Canonical default parameter values shared by replay CLI and any other caller.
# Override individual keys as needed — do not hardcode these elsewhere.
DEFAULT_STRATEGY_PARAMS: dict = dict(
    macd_fast=12, macd_slow=26, macd_signal=9,
    rsi_n=14, bb_n=20, bb_k=2.0, kdj_n=9, cci_n=20,
    donchian_n=100, week_n=5,
    bbi_eps=0.002, bbi_breakout=True, bbi_fail_eps=0.002,
    ta_n=14,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bbi_series(close: pd.Series) -> pd.Series:
    ma3 = close.rolling(3).mean()
    ma6 = close.rolling(6).mean()
    ma12 = close.rolling(12).mean()
    ma24 = close.rolling(24).mean()
    return (ma3 + ma6 + ma12 + ma24) / 4


def _rsi_from_avg(avg_gain: pd.Series, avg_loss: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=avg_gain.index, dtype=float)
    gain = pd.to_numeric(avg_gain, errors="coerce").astype(float)
    loss = pd.to_numeric(avg_loss, errors="coerce").astype(float)

    both_zero = (gain == 0) & (loss == 0)
    loss_zero = (loss == 0) & (gain > 0)
    gain_zero = (gain == 0) & (loss > 0)
    normal = ~(both_zero | loss_zero | gain_zero)

    rs = gain[normal] / loss[normal]
    out.loc[normal] = 100 - (100 / (1 + rs))
    out.loc[loss_zero] = 100.0
    out.loc[gain_zero] = 0.0
    out.loc[both_zero] = 50.0
    return out


def _cross_above_intrabar(
    prev_high: pd.Series, high: pd.Series,
    prev_thr: pd.Series, thr: pd.Series,
) -> pd.Series:
    return (prev_high <= prev_thr) & (high > thr)


def _cross_below_intrabar(
    prev_low: pd.Series, low: pd.Series,
    prev_thr: pd.Series, thr: pd.Series,
) -> pd.Series:
    return (prev_low >= prev_thr) & (low < thr)


def _bool_shift_prev(x: pd.Series) -> pd.Series:
    s = x.shift(1)
    return s.where(s.notna(), False).astype(bool)


def _apply_grid_signals_intrabar(
    out: pd.DataFrame,
    high: pd.Series,
    low: pd.Series,
    grid_pct: float,
) -> None:
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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_signals(df: pd.DataFrame, strategy: str, **kwargs) -> pd.DataFrame:
    """
    Compute buy/sell signals for *strategy* on bar data *df*.

    Parameters
    ----------
    df       : DataFrame with columns timestamp_utc, open, high, low, close, volume
    strategy : One of STRATEGY_OPTIONS
    **kwargs : Strategy-specific parameter overrides

    Returns
    -------
    df copy with additional columns strat_signal (+1/-1/0) and strat_exec_px.
    """
    out = df.copy()
    out["strat_signal"] = 0
    out["strat_exec_px"] = np.nan

    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    open_ = out["open"].astype(float)

    prev_high = high.shift(1)
    prev_low = low.shift(1)

    if strategy == "全仓买入并持有":
        signal_bar = str(kwargs.get("buy_hold_signal_bar", "last")).lower()
        target_idx = out.index[0] if signal_bar == "first" else out.index[-1]
        # Default to the LAST bar so the live engine, which checks result.iloc[-1],
        # picks it up. Research views can request "first" for chart markers.
        if len(out) > 0:
            out.at[target_idx, "strat_signal"] = 1
            out.at[target_idx, "strat_exec_px"] = float(out.at[target_idx, "close"])

    elif strategy == "5/20均线金叉死叉":
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
        mf = int(kwargs.get("macd_fast", 12))
        ms = int(kwargs.get("macd_slow", 26))
        ef = close.ewm(span=mf, adjust=False).mean()
        es = close.ewm(span=ms, adjust=False).mean()
        macd = ef - es
        buy = (macd.shift(1) <= 0) & (macd > 0)
        sell = (macd.shift(1) >= 0) & (macd < 0)
        out.loc[buy, "strat_signal"] = 1
        out.loc[buy, "strat_exec_px"] = close
        out.loc[sell, "strat_signal"] = -1
        out.loc[sell, "strat_exec_px"] = close

    elif strategy == "MACD信号线红绿柱":
        mf = int(kwargs.get("macd_fast", 12))
        ms = int(kwargs.get("macd_slow", 26))
        msig = int(kwargs.get("macd_signal", 9))
        ef = close.ewm(span=mf, adjust=False).mean()
        es = close.ewm(span=ms, adjust=False).mean()
        macd = ef - es
        sig = macd.ewm(span=msig, adjust=False).mean()
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
        rsi = _rsi_from_avg(avg_gain, avg_loss)
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
        mean_dev = tp.rolling(cci_n).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True)
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
        bbi = _bbi_series(close)
        buy = _cross_above_intrabar(prev_high, high, bbi.shift(1), bbi)
        sell = _cross_below_intrabar(prev_low, low, bbi.shift(1), bbi)
        out.loc[buy, "strat_signal"] = 1
        out.loc[buy, "strat_exec_px"] = bbi
        out.loc[sell, "strat_signal"] = -1
        out.loc[sell, "strat_exec_px"] = bbi

    elif strategy == "BBI回踩不破做多(顺势二次上车)":
        bbi = _bbi_series(close)
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
            out.loc[setup, "strat_signal"] = 1
            out.loc[setup, "strat_exec_px"] = bbi
        sell = _cross_below_intrabar(prev_low, low, bbi.shift(1), bbi)
        out.loc[sell, "strat_signal"] = -1
        out.loc[sell, "strat_exec_px"] = bbi

    elif strategy == "BBI回踩不破+斜率过滤":
        bbi = _bbi_series(close)
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
            out.loc[setup, "strat_signal"] = 1
            out.loc[setup, "strat_exec_px"] = bbi
        sell = _cross_below_intrabar(prev_low, low, bbi.shift(1), bbi)
        out.loc[sell, "strat_signal"] = -1
        out.loc[sell, "strat_exec_px"] = bbi

    elif strategy == "BBI跌破反抽不过做空/卖出":
        bbi = _bbi_series(close)
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

    elif strategy == "ADX趋势过滤(ta)":
        ta_window = int(kwargs.get("ta_n", kwargs.get("atr_n", 14)))
        adx_ind = ta.trend.ADXIndicator(
            high=high, low=low, close=close, window=ta_window, fillna=False)
        adx = adx_ind.adx()
        plus_di = adx_ind.adx_pos()
        minus_di = adx_ind.adx_neg()
        buy = (plus_di > minus_di) & (adx >= 20) & (plus_di.shift(1) <= minus_di.shift(1))
        sell = (minus_di > plus_di) & (adx >= 20) & (minus_di.shift(1) <= plus_di.shift(1))
        out.loc[buy, "strat_signal"] = 1
        out.loc[buy, "strat_exec_px"] = close
        out.loc[sell, "strat_signal"] = -1
        out.loc[sell, "strat_exec_px"] = close

    elif strategy == "Stoch超买超卖(ta)":
        ta_window = int(kwargs.get("ta_n", kwargs.get("rsi_n", 14)))
        stoch = ta.momentum.StochasticOscillator(
            high=high, low=low, close=close,
            window=ta_window, smooth_window=3, fillna=False)
        k = stoch.stoch()
        d = stoch.stoch_signal()
        buy = (k.shift(1) <= d.shift(1)) & (k > d) & (k < 20)
        sell = (k.shift(1) >= d.shift(1)) & (k < d) & (k > 80)
        out.loc[buy, "strat_signal"] = 1
        out.loc[buy, "strat_exec_px"] = close
        out.loc[sell, "strat_signal"] = -1
        out.loc[sell, "strat_exec_px"] = close

    elif strategy == "Williams %R反转(ta)":
        ta_window = int(kwargs.get("ta_n", kwargs.get("rsi_n", 14)))
        willr = ta.momentum.WilliamsRIndicator(
            high=high, low=low, close=close, lbp=ta_window, fillna=False
        ).williams_r()
        buy = (willr.shift(1) <= -80) & (willr > -80)
        sell = (willr.shift(1) >= -20) & (willr < -20)
        out.loc[buy, "strat_signal"] = 1
        out.loc[buy, "strat_exec_px"] = close
        out.loc[sell, "strat_signal"] = -1
        out.loc[sell, "strat_exec_px"] = close

    elif strategy == "MFI量价共振(ta)":
        ta_window = int(kwargs.get("ta_n", kwargs.get("atr_n", 14)))
        mfi = ta.volume.MFIIndicator(
            high=high, low=low, close=close,
            volume=out["volume"].astype(float),
            window=ta_window, fillna=False,
        ).money_flow_index()
        buy = (mfi.shift(1) <= 50) & (mfi > 50)
        sell = (mfi.shift(1) >= 50) & (mfi < 50)
        out.loc[buy, "strat_signal"] = 1
        out.loc[buy, "strat_exec_px"] = close
        out.loc[sell, "strat_signal"] = -1
        out.loc[sell, "strat_exec_px"] = close

    return out
