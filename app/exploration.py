"""
app/exploration.py
Exploration / paper simulation panel — called from trader/monitor.py as a tab.
All controls are rendered inline inside the tab (no sidebar dependency).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy import stats
import quantstats as qs
import ta

_ROOT = Path(__file__).resolve().parents[1]

import logging as _logging
logger_exp = _logging.getLogger(__name__)

from trader.strategy_core import STRATEGY_OPTIONS as _CORE_STRATEGY_OPTIONS  # noqa: E402
from trader.strategies.registry import build_default_registry as _build_default_registry  # noqa: E402
from trader.engine import simulate as _simulate  # noqa: E402

try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(_ROOT / ".env", override=True)
except Exception:
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NY_TZ = "America/New_York"
INITIAL_CAPITAL = 10_000.0
DCA_N = 20

_EXPLORATION_ONLY_STRATEGIES = ["分期定投(20次均匀)"]
STRATEGY_OPTIONS = [
    _name
    for _name in (
        [_CORE_STRATEGY_OPTIONS[0]]
        + _EXPLORATION_ONLY_STRATEGIES
        + list(_CORE_STRATEGY_OPTIONS[1:])
    )
]
_SHARED_STRATEGIES = _build_default_registry()

# Strategy groupings for the category filter radio button
_STRATEGY_CATEGORIES: Dict[str, Optional[List[str]]] = {
    "全部": None,
    "📈 趋势 (均线/BBI)": [
        "5/20均线金叉死叉", "10/30均线双线波段", "20/60均线长线趋势",
        "BBI上穿下穿(收盘确认)", "BBI回踩不破做多(顺势二次上车)",
        "BBI回踩不破+斜率过滤", "BBI跌破反抽不过做空/卖出",
        "ADX趋势过滤(ta)",
    ],
    "⚡ 动量 & 震荡": [
        "MACD零轴战法", "MACD信号线红绿柱", "RSI震荡战法(60买40卖)",
        "KDJ极值反转(J线探底)", "Stoch超买超卖(ta)", "Williams %R反转(ta)",
        "MFI量价共振(ta)", "CCI顺势指标(±100突破)",
    ],
    "💥 突破 & 形态": [
        "布林带突破(上下轨)", "布林带均值回归(探底回升)",
        "唐奇安通道(20周突破)", "上周高低点(周K突破)", "三阳买两阴卖(形态跟踪)",
    ],
    "🔲 网格": [
        "半仓小网格(5%间距)", "半仓大网格(10%间距)",
    ],
}

# ---------------------------------------------------------------------------
# Pure helpers (stateless)
# ---------------------------------------------------------------------------

def _parse_symbols(raw: str) -> List[str]:
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


def _is_crypto(symbol: str) -> bool:
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


def _rangebreaks(timeframe: str, is_crypto: bool):
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


def _cross_above(prev_high, high, prev_thr, thr):
    return (prev_high <= prev_thr) & (high > thr)


def _cross_below(prev_low, low, prev_thr, thr):
    return (prev_low >= prev_thr) & (low < thr)


def _compute_indicators(df, ind_list, sma_n, ema_fast, ema_slow, bb_n, bb_k,
                        rsi_n, macd_fast, macd_slow, macd_signal, atr_n, ta_n):
    out = df.copy()
    close = out["close"].astype(float)
    high  = out["high"].astype(float)
    low   = out["low"].astype(float)

    if "SMA"   in ind_list:
        out[f"sma_{sma_n}"] = close.rolling(int(sma_n)).mean()
    if "EMA"   in ind_list:
        out[f"ema_{ema_fast}"] = close.ewm(span=int(ema_fast), adjust=False).mean()
        out[f"ema_{ema_slow}"] = close.ewm(span=int(ema_slow), adjust=False).mean()
    if "BBANDS" in ind_list:
        mid = close.rolling(int(bb_n)).mean()
        sd  = close.rolling(int(bb_n)).std(ddof=0)
        out["bb_mid"] = mid
        out["bb_up"]  = mid + float(bb_k) * sd
        out["bb_dn"]  = mid - float(bb_k) * sd
    if "BBI"   in ind_list:
        out["bbi"] = _bbi_series(close)
    if "RSI"   in ind_list:
        n = int(rsi_n)
        delta    = close.diff()
        gain     = delta.clip(lower=0.0)
        loss     = (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
        avg_loss = loss.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
        out["rsi"] = _rsi_from_avg(avg_gain, avg_loss)
    if "MACD"  in ind_list:
        ef  = close.ewm(span=int(macd_fast), adjust=False).mean()
        es  = close.ewm(span=int(macd_slow), adjust=False).mean()
        macd = ef - es
        sig  = macd.ewm(span=int(macd_signal), adjust=False).mean()
        out["macd"]        = macd
        out["macd_signal"] = sig
        out["macd_hist"]   = macd - sig
    if "ATR"   in ind_list:
        prev_close = close.shift(1)
        tr = pd.concat(
            [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
        ).max(axis=1)
        out["atr"] = tr.ewm(span=int(atr_n), adjust=False).mean()
    if "ADX"   in ind_list:
        adx_ind  = ta.trend.ADXIndicator(high=high, low=low, close=close, window=int(ta_n), fillna=False)
        out["adx"]     = adx_ind.adx()
        out["adx_pos"] = adx_ind.adx_pos()
        out["adx_neg"] = adx_ind.adx_neg()
    if "STOCH" in ind_list:
        stoch = ta.momentum.StochasticOscillator(
            high=high, low=low, close=close, window=int(ta_n), smooth_window=3, fillna=False)
        out["stoch_k"] = stoch.stoch()
        out["stoch_d"] = stoch.stoch_signal()
    if "WILLR" in ind_list:
        out["willr"] = ta.momentum.WilliamsRIndicator(
            high=high, low=low, close=close, lbp=int(ta_n), fillna=False).williams_r()
    if "MFI"   in ind_list:
        out["mfi"] = ta.volume.MFIIndicator(
            high=high, low=low, close=close,
            volume=out["volume"].astype(float), window=int(ta_n), fillna=False,
        ).money_flow_index()
    if "OBV"   in ind_list:
        out["obv"] = ta.volume.OnBalanceVolumeIndicator(
            close=close, volume=out["volume"].astype(float), fillna=False,
        ).on_balance_volume()
    return out


# ---------------------------------------------------------------------------
# Strategy signal computation — single source of truth: strategy_core
# ---------------------------------------------------------------------------

def _build_strategy_signals(df: pd.DataFrame, strategy: str, **kwargs) -> pd.DataFrame:
    """Thin wrapper: delegates entirely to strategy_core via the shared registry.
    DCA is the only exploration-only strategy; it carries no signals."""
    if strategy == "分期定投(20次均匀)":
        out = df.copy()
        out["strat_signal"]  = 0
        out["strat_exec_px"] = np.nan
        return out
    shared_kwargs = dict(kwargs)
    if strategy == "全仓买入并持有":
        shared_kwargs.setdefault("buy_hold_signal_bar", "first")
    return _SHARED_STRATEGIES.compute(strategy, df, **shared_kwargs)


# ---------------------------------------------------------------------------
# Historical simulation helpers
# ---------------------------------------------------------------------------

def _sim_resample_4h(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df2 = df.copy().sort_values("timestamp_utc").reset_index(drop=True)
    df2["timestamp_utc"] = pd.to_datetime(df2["timestamp_utc"], utc=True)
    df2 = df2.set_index("timestamp_utc")
    agg = df2[["open", "high", "low", "close", "volume"]].resample(
        "4h", closed="left", label="left"
    ).agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["open", "close"])
    agg = agg.reset_index()
    agg["timestamp"] = agg["timestamp_utc"]
    if "symbol" in df.columns:
        agg["symbol"] = df["symbol"].iloc[0]
    return agg


@st.cache_data(ttl=300)
def _sim_load_data(symbol: str, tf: str) -> pd.DataFrame:
    fetch_tf = "1h" if tf == "4h" else tf
    df = _fetch_bars_merged(symbol, fetch_tf)
    if df is None or df.empty:
        return pd.DataFrame()
    if tf == "4h":
        df = _sim_resample_4h(df)
    return df.sort_values("timestamp_utc").drop_duplicates(subset=["timestamp_utc"]).reset_index(drop=True)


def _sim_check_signal(df_slice: pd.DataFrame, strategy: str, kwargs: dict) -> tuple:
    """Compute signals on df_slice and return (signal_int, exec_px) for the last bar."""
    if len(df_slice) < 3:
        return 0, float("nan")
    lookback = max(120, kwargs.get("week_n", 5) * 3 + 60, kwargs.get("donchian_n", 100) + 5)
    df_work = df_slice.tail(lookback).copy().reset_index(drop=True)
    try:
        df_sig = _build_strategy_signals(df_work, strategy, **kwargs)
    except Exception:
        return 0, float("nan")
    last = df_sig.iloc[-1]
    sig  = int(last.get("strat_signal", 0))
    px   = float(last.get("strat_exec_px", float("nan")))
    if not np.isfinite(px) or px <= 0:
        px = float(last["close"])
    return sig, px


def _sim_render_chart(df_slice: pd.DataFrame, trades: list, visible_n: int, tf: str) -> go.Figure:
    n_total  = len(df_slice)
    start_idx = max(0, n_total - visible_n)
    df_vis   = df_slice.iloc[start_idx:].copy().reset_index(drop=True)
    n_vis    = len(df_vis)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.02, row_heights=[0.78, 0.22])
    x          = list(range(n_vis))
    hover_time = _build_time_strings(df_vis, tf).to_numpy()

    fig.add_trace(go.Candlestick(
        x=x, open=df_vis["open"], high=df_vis["high"],
        low=df_vis["low"], close=df_vis["close"], name="OHLC",
        customdata=hover_time,
        hovertemplate="Time: %{customdata}<br>O: %{open}<br>H: %{high}<br>L: %{low}<br>C: %{close}<extra></extra>",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=x, y=df_vis["volume"].astype(float), name="Volume", opacity=0.35,
        marker_color=[
            "#26a69a" if float(df_vis["close"].iloc[i]) >= float(df_vis["open"].iloc[i]) else "#ef5350"
            for i in range(n_vis)
        ],
    ), row=2, col=1)

    global_to_vis = {start_idx + i: i for i in range(n_vis)}
    _dir_cfg = {
        "买入": dict(symbol="triangle-up",   color="#00c853", size=14),
        "卖出": dict(symbol="triangle-down",  color="#ff1744", size=14),
        "做空": dict(symbol="triangle-down",  color="#aa00ff", size=14),
        "回补": dict(symbol="triangle-up",    color="#2979ff", size=14),
    }
    for direction, cfg in _dir_cfg.items():
        pts = [(t["bar_idx"], t["exec_px"]) for t in trades
               if t["direction"] == direction and t["bar_idx"] in global_to_vis]
        if not pts:
            continue
        vis_x = [global_to_vis[p[0]] for p in pts]
        vis_y = [p[1] for p in pts]
        offset_dir = 1 if cfg["symbol"] == "triangle-up" else -1
        bar_range  = float(df_vis["high"].max() - df_vis["low"].min()) if n_vis > 0 else 1.0
        offset     = bar_range * 0.012 * offset_dir
        fig.add_trace(go.Scatter(
            x=vis_x, y=[y - offset for y in vis_y], mode="markers+text",
            name=direction,
            marker=dict(symbol=cfg["symbol"], size=cfg["size"], color=cfg["color"]),
            text=[direction] * len(vis_x),
            textposition="top center" if offset_dir > 0 else "bottom center",
            textfont=dict(size=9, color=cfg["color"]),
            hovertemplate=f"{direction}<br>Px: %{{y:.2f}}<extra></extra>",
        ), row=1, col=1)

    tick_idx, tick_text = _tick_for_nogap(df_vis, tf)
    shapes = []
    if tf != "1d":
        shapes = _make_shapes_nogap(_day_start_positions_idx(df_vis, tf))
    fig.update_xaxes(row=2, col=1, type="linear", tickmode="array",
                     tickvals=tick_idx, ticktext=tick_text, tickangle=-30)
    fig.update_xaxes(row=1, col=1, showticklabels=False)
    fig.update_layout(
        height=520, margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.01),
        xaxis_rangeslider_visible=False, shapes=shapes, uirevision="sim-chart",
    )
    fig.update_yaxes(title_text="Price", row=1, col=1, fixedrange=False)
    fig.update_yaxes(title_text="Vol",   row=2, col=1, fixedrange=True, showgrid=False)
    return fig


# ---------------------------------------------------------------------------
# Performance / stats helpers
# ---------------------------------------------------------------------------

def _fmt_pval(p: Optional[float]) -> str:
    if p is None:
        return ""
    try:
        p = float(p)
        if not np.isfinite(p):
            return ""
        return "<0.001" if p < 0.001 else f"{p:.3f}"
    except Exception:
        return ""


def _pval_h0_mean_zero(trade_rets: List[float]) -> str:
    if not trade_rets:
        return ""
    x = np.asarray(trade_rets, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return ""
    res = stats.ttest_1samp(x, popmean=0.0, alternative="two-sided")
    return _fmt_pval(float(res.pvalue))


def _simulate_signal_strategy_mark_to_market(
    df, signal_col="strat_signal", exec_col="strat_exec_px",
    leverage=1.0, initial_capital=INITIAL_CAPITAL,
):
    """Thin adapter over the unified engine. Returns (wins, trades, equity, trade_rets)
    to keep the existing UI call sites unchanged."""
    if df is None or df.empty or signal_col not in df.columns or exec_col not in df.columns:
        return 0, 0, float(initial_capital), []
    res = _simulate(
        df, capital=float(initial_capital), leverage=float(leverage),
        signal_col=signal_col, exec_col=exec_col,
    )
    trade_rets = [float(t.ret) for t in res.trades if t.realized_pnl != 0.0]
    return res.wins, res.closed_trades, float(res.final_equity), trade_rets


def _simulate_buy_hold_balance(df, leverage=1.0, initial_capital=INITIAL_CAPITAL):
    if df is None or df.empty:
        return 0, 1, float(initial_capital)
    L        = max(float(leverage), 1.0)
    entry_px = float(pd.to_numeric(df["open"], errors="coerce").iloc[0])
    if not np.isfinite(entry_px) or entry_px <= 0:
        entry_px = float(pd.to_numeric(df["close"], errors="coerce").iloc[0])
    if not np.isfinite(entry_px) or entry_px <= 0:
        return 0, 1, float(initial_capital)
    last_px = float(pd.to_numeric(df["close"], errors="coerce").iloc[-1])
    if not np.isfinite(last_px) or last_px <= 0:
        last_px = entry_px
    debt   = initial_capital * (L - 1.0)
    shares = (initial_capital * L) / entry_px
    equity = max(shares * last_px - debt, 0.0)
    return 0, 1, float(equity)


def _dca_indices(nbars, n_tranches=DCA_N):
    if nbars <= 0:
        return []
    if n_tranches <= 1:
        return [0]
    if nbars == 1:
        return [0]
    idx = [int(round(k * (nbars - 1) / (n_tranches - 1))) for k in range(n_tranches)]
    out, prev = [], None
    for i in idx:
        i = max(0, min(nbars - 1, i))
        if prev is None or i != prev:
            out.append(i)
        prev = i
    return out


def _simulate_dca_20_balance(df, leverage=1.0, initial_capital=INITIAL_CAPITAL):
    if df is None or df.empty:
        return 0, DCA_N, float(initial_capital)
    L      = max(float(leverage), 1.0)
    nbars  = len(df); idxs = _dca_indices(nbars, DCA_N)
    if not idxs:
        return 0, DCA_N, float(initial_capital)
    tranche = initial_capital / float(DCA_N)
    cash = float(initial_capital); debt = 0.0; shares = 0.0
    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=float)
    for i in idxs:
        if cash <= 0:
            cash = 0.0; break
        px = float(close[i]) if np.isfinite(close[i]) else np.nan
        if not np.isfinite(px) or px <= 0:
            continue
        own = min(tranche, cash); exposure = own * L; borrow = exposure - own
        shares += exposure / px; cash -= own; debt += borrow
        if not np.isfinite(cash + shares * px - debt) or cash + shares * px - debt < 0:
            return 0, DCA_N, 0.0
    last_px = float(close[-1]) if np.isfinite(close[-1]) and close[-1] > 0 else float(close[idxs[-1]])
    equity  = max(cash + shares * last_px - debt, 0.0)
    return 0, DCA_N, float(equity)


def _dca_max_drawdown(df, leverage=1.0, initial_capital=INITIAL_CAPITAL) -> float:
    if df is None or df.empty:
        return 0.0
    L     = max(float(leverage), 1.0)
    nbars = len(df); idxs = _dca_indices(nbars, DCA_N)
    if not idxs:
        return 0.0
    tranche = initial_capital / float(DCA_N)
    cash = float(initial_capital); debt = 0.0; shares = 0.0
    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=float)
    equity_curve = []
    next_dca_idx = 0
    for i in range(nbars):
        if next_dca_idx < len(idxs) and i == idxs[next_dca_idx]:
            px = float(close[i]) if np.isfinite(close[i]) else np.nan
            if np.isfinite(px) and px > 0 and cash > 0:
                own = min(tranche, cash); exposure = own * L; borrow = exposure - own
                shares += exposure / px; cash -= own; debt += borrow
            next_dca_idx += 1
        px = float(close[i]) if np.isfinite(close[i]) else np.nan
        if np.isfinite(px) and px > 0:
            equity_curve.append(max(cash + shares * px - debt, 0.0))
        else:
            equity_curve.append(equity_curve[-1] if equity_curve else initial_capital)
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]; max_dd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return float(max_dd)


def _bars_per_year(timeframe: str) -> int:
    return {"1m": 252 * 390, "5m": 252 * 78, "30m": 252 * 13, "1h": 252 * 7, "1d": 252}.get(timeframe, 252)


def _strategy_return_curve(df, strategy, base_params, leverage, initial_capital=INITIAL_CAPITAL):
    """Equity / return curve from the unified engine (timestamp-indexed)."""
    empty = pd.Series(dtype=float)
    if df is None or df.empty or strategy == "分期定投(20次均匀)":
        return empty, empty
    df_s = _build_strategy_signals(df.copy(), strategy, **base_params)
    res = _simulate(df_s, capital=float(initial_capital), leverage=float(leverage))
    equity = res.equity_curve
    if equity is None or equity.empty:
        return empty, empty
    returns = equity.pct_change(fill_method=None).fillna(0.0)
    returns.name = "returns"; equity.name = "equity"
    return returns, equity


def _performance_summary_table(returns, equity, timeframe, initial_capital=INITIAL_CAPITAL):
    if returns is None or returns.empty:
        return pd.DataFrame()
    ppy = _bars_per_year(timeframe); summary = []

    def _safe(name, fn, fb=np.nan):
        try:    v = fn()
        except Exception: v = fb
        summary.append({"metric": name, "value": v})

    _safe("累计收益",       lambda: float(equity.iloc[-1] / initial_capital - 1.0) if not equity.empty else np.nan)
    _safe("年化收益(CAGR)", lambda: float(qs.stats.cagr(returns, periods=ppy)))
    _safe("夏普比率",       lambda: float(qs.stats.sharpe(returns, periods=ppy)))
    _safe("索提诺比率",     lambda: float(qs.stats.sortino(returns, periods=ppy)))
    _safe("最大回撤",       lambda: float(qs.stats.max_drawdown(returns)))
    _safe("波动率",         lambda: float(qs.stats.volatility(returns, periods=ppy)))
    _safe("胜率",           lambda: float((returns > 0).mean()))
    return pd.DataFrame(summary)


def _fmt_int_cell(x) -> str:
    if x is None: return ""
    try:
        v = float(x)
        return "" if not np.isfinite(v) else str(int(round(v)))
    except Exception: return ""


def _span_label(df_window) -> str:
    if df_window is None or df_window.empty or "timestamp_utc" not in df_window.columns:
        return "0bars"
    ts    = pd.to_datetime(df_window["timestamp_utc"], errors="coerce", utc=True).dropna()
    if ts.empty: return "0bars"
    delta = ts.max() - ts.min(); days = int(delta.days); hours = int(delta.total_seconds() // 3600)
    if days >= 60: return f"{max(1, int(round(days / 30)))}m"
    if days >= 1:  return f"{days}d"
    if hours >= 1: return f"{hours}h"
    return f"{len(ts)}bars"


def _indicator_and_one_price(df, signal_col="strat_signal", exec_col="strat_exec_px"):
    if df is None or df.empty or signal_col not in df.columns or exec_col not in df.columns:
        return "hold", ""
    current_px = float(pd.to_numeric(df["close"], errors="coerce").iloc[-1])
    if not np.isfinite(current_px) or current_px <= 0: return "hold", ""
    sig      = _col_int_signal(df, signal_col)
    exec_px  = pd.to_numeric(df[exec_col], errors="coerce").to_numpy(dtype=float)
    buy_idx  = np.where(sig == 1)[0];  sell_idx = np.where(sig == -1)[0]
    last_buy_i  = int(buy_idx[-1])  if len(buy_idx)  else None
    last_sell_i = int(sell_idx[-1]) if len(sell_idx) else None
    last_buy_px  = float(exec_px[last_buy_i])  if last_buy_i  is not None and np.isfinite(exec_px[last_buy_i])  else None
    last_sell_px = float(exec_px[last_sell_i]) if last_sell_i is not None and np.isfinite(exec_px[last_sell_i]) else None
    cond_buy  = (last_buy_px  is not None) and (current_px < last_buy_px)
    cond_sell = (last_sell_px is not None) and (current_px > last_sell_px)
    if cond_buy and cond_sell:
        if last_buy_i is not None and last_sell_i is not None and last_sell_i > last_buy_i:
            return "sell", _fmt_int_cell(last_sell_px)
        return "buy", _fmt_int_cell(last_buy_px)
    if cond_buy:  return "buy",  _fmt_int_cell(last_buy_px)
    if cond_sell: return "sell", _fmt_int_cell(last_sell_px)
    return "hold", ""


# ---------------------------------------------------------------------------
# Chart shape helpers
# ---------------------------------------------------------------------------

def _make_shapes_nogap(day_starts):
    return [dict(type="line", xref="x", yref="paper", x0=i, x1=i, y0=0, y1=1,
                 line=dict(width=1, dash="dot"), opacity=0.25) for i in day_starts]


def _make_shapes_date(starts_ts):
    return [dict(type="line", xref="x", yref="paper", x0=x, x1=x, y0=0, y1=1,
                 line=dict(width=1, dash="dot"), opacity=0.25) for x in starts_ts]


# ---------------------------------------------------------------------------
# Cached I/O helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def _fetch_bars_merged(symbol: str, timeframe: str) -> pd.DataFrame:
    """Local-first: memory cache / Parquet → yfinance fallback. No automatic Alpaca call."""
    try:
        from trader.data_cache import get_bars as _dc_get
        df = _dc_get(symbol, timeframe)
        if df is not None and not df.empty:
            if "symbol" not in df.columns:
                df = df.copy(); df["symbol"] = symbol
            if "timestamp" not in df.columns:
                df = df.copy(); df["timestamp"] = df["timestamp_utc"]
            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
            df["timestamp"]     = df["timestamp_utc"]
            return df.sort_values("timestamp_utc").drop_duplicates(subset=["timestamp_utc"]).reset_index(drop=True)
    except Exception as _e:
        logger_exp.warning("_fetch_bars_merged data_cache error: %s", _e)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Strategy stats table (cached)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60)
def _build_stats_table_exp(
    df_full: pd.DataFrame, strategies: tuple, base_params_frozen: tuple,
    leverage: float, timeframe: str,
) -> pd.DataFrame:
    base_params      = dict(base_params_frozen)
    strategies_list  = list(strategies)
    return _build_stats_impl(df_full, strategies_list, base_params, leverage, timeframe)


def _build_stats_impl(df_full, strategies, base_params, leverage, timeframe):
    if df_full is None or df_full.empty:
        return pd.DataFrame()
    ts_now  = pd.Timestamp.now(tz="UTC")
    ts_ytd  = pd.Timestamp(year=ts_now.year, month=1, day=1, tz="UTC")
    ts_30d  = ts_now - pd.Timedelta(days=30)
    windows = {
        f"全部({_span_label(df_full)})":
            df_full,
        f"YTD({_span_label(df_full[df_full['timestamp_utc'] >= ts_ytd])})":
            df_full[df_full["timestamp_utc"] >= ts_ytd],
        f"30天({_span_label(df_full[df_full['timestamp_utc'] >= ts_30d])})":
            df_full[df_full["timestamp_utc"] >= ts_30d],
    }
    rows = []
    for name, df_w in windows.items():
        if df_w is None or df_w.empty:
            continue
        for s in strategies:
            wins, trades, final_bal, trade_rets = 0, 0, INITIAL_CAPITAL, []
            max_dd = 0.0
            if s == "全仓买入并持有":
                wins, trades, final_bal = _simulate_buy_hold_balance(df_w, leverage)
                returns, _ = _strategy_return_curve(df_w, s, base_params, leverage)
                if not returns.empty:
                    max_dd = qs.stats.max_drawdown(returns)
                win_rate = np.nan
            elif s == "分期定投(20次均匀)":
                wins, trades, final_bal = _simulate_dca_20_balance(df_w, leverage)
                max_dd   = _dca_max_drawdown(df_w, leverage)
                win_rate = np.nan
            else:
                df_s = _build_strategy_signals(df_w.copy(), s, **base_params)
                wins, trades, final_bal, trade_rets = _simulate_signal_strategy_mark_to_market(df_s, leverage=leverage)
                returns, _ = _strategy_return_curve(df_w, s, base_params, leverage)
                if not returns.empty:
                    max_dd = qs.stats.max_drawdown(returns)
                win_rate = (wins / trades) if trades > 0 else 0.0
            p_val = _pval_h0_mean_zero(trade_rets)
            rows.append({
                "期间": name, "策略": s, "胜率": win_rate, "交易次数": trades,
                "最终收益": final_bal / INITIAL_CAPITAL - 1.0,
                "最大回撤": max_dd, "p-value": p_val,
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Inline control panel — call this at the TOP of render_exploration_tab()
# ---------------------------------------------------------------------------

def _render_exp_controls() -> None:
    """Render all exploration settings as an inline expandable panel.
    Widgets write directly into session_state via their keys."""
    default_symbols = os.getenv("SYMBOLS", "AAPL,MSFT")

    with st.expander("⚙️ 参数设置", expanded=True):

        # Row 0: initial capital + symbols + timeframe + x-mode
        c0, c1, c3, c4 = st.columns([1.5, 3, 1, 2])
        c0.number_input(
            "初始本金 ($)", min_value=1_000.0, max_value=10_000_000.0,
            value=INITIAL_CAPITAL, step=1_000.0, key="exp_initial_capital",
        )
        c1.text_input("回测标的 (逗号分隔)", default_symbols, key="exp_symbols")
        c3.selectbox("周期", ["1m", "5m", "15m", "30m", "1h", "1d"], index=3, key="exp_tf")
        c4.selectbox(
            "X轴模式",
            ["NO_GAP（index轴彻底无gap）", "DATE_AXIS（日期轴 + rangebreaks）"],
            index=0, key="exp_xmode",
        )

        # Row 1: leverage, autoscale, window, day_sep
        c5, c6, c7, c8 = st.columns([1, 1, 2, 1])
        c5.number_input("杠杆 (>=1)", min_value=1.0, max_value=20.0, value=1.0, step=0.5, key="exp_leverage")
        autoscale = c6.checkbox("自动缩放(最近90根)", value=False, key="exp_autoscale")
        if not autoscale:
            c7.slider("可见窗口K线数", 30, 5000, 240, step=10, key="exp_window")
        c8.checkbox("按天分隔线", value=True, key="exp_day_sep")

        # Row 2: indicator multiselect
        ind_list = st.multiselect(
            "叠加指标",
            ["SMA", "EMA", "BBANDS", "BBI", "RSI", "MACD", "ATR", "ADX", "STOCH", "WILLR", "MFI", "OBV"],
            default=["EMA", "BBI"], key="exp_ind_list",
        )

        # Row 3: indicator-specific params (conditional)
        has_params = any(k in ind_list for k in
                         ["SMA", "EMA", "BBANDS", "RSI", "MACD", "ATR", "ADX", "STOCH", "WILLR", "MFI", "OBV"])
        if has_params:
            ip = st.columns(6); i = 0
            if "SMA"   in ind_list: ip[i%6].number_input("SMA n",       5,   200,  20,            key="exp_sma_n");   i+=1
            if "EMA"   in ind_list:
                ip[i%6].number_input("EMA fast",  2,   200,  12,            key="exp_ema_fast"); i+=1
                ip[i%6].number_input("EMA slow",  2,   400,  26,            key="exp_ema_slow"); i+=1
            if "BBANDS" in ind_list:
                ip[i%6].number_input("BB n",      5,   200,  20,            key="exp_bb_n");    i+=1
                ip[i%6].number_input("BB k",      1.0, 5.0,  2.0, step=0.5, key="exp_bb_k");   i+=1
            if "RSI"   in ind_list: ip[i%6].number_input("RSI n",       2,   200,  14,            key="exp_rsi_n");   i+=1
            if "MACD"  in ind_list:
                ip[i%6].number_input("MACD fast", 2,    50,  12,            key="exp_macd_f");  i+=1
                ip[i%6].number_input("MACD slow", 5,   100,  26,            key="exp_macd_s");  i+=1
                ip[i%6].number_input("MACD sig",  2,    50,   9,            key="exp_macd_sg"); i+=1
            if "ATR"   in ind_list: ip[i%6].number_input("ATR n",       2,   200,  14,            key="exp_atr_n");   i+=1
            if any(k in ind_list for k in ["ADX", "STOCH", "WILLR", "MFI", "OBV"]):
                                    ip[i%6].number_input("TA window",   2,   100,  14,            key="exp_ta_n");    i+=1

        # Row 4: strategy category filter + strategy selectbox
        all_chartable   = [s for s in STRATEGY_OPTIONS if s not in ("全仓买入并持有", "分期定投(20次均匀)")]
        cat_names       = list(_STRATEGY_CATEGORIES.keys())
        sel_cat         = st.radio("策略分类", cat_names, index=0, horizontal=True, key="exp_strat_cat")
        cat_filter      = _STRATEGY_CATEGORIES.get(sel_cat)
        filtered_strats = all_chartable if cat_filter is None else [s for s in cat_filter if s in all_chartable]
        if not filtered_strats:
            filtered_strats = all_chartable

        # Reset selection when switching category and current selection is no longer in list
        current_strat = st.session_state.get("exp_strategy", filtered_strats[0])
        if current_strat not in filtered_strats:
            st.session_state["exp_strategy"] = filtered_strats[0]
        default_idx = filtered_strats.index(st.session_state.get("exp_strategy", filtered_strats[0]))

        sc1, sc2, sc3, sc4, sc5s = st.columns([3, 1, 1, 1, 1])
        sc1.selectbox("图表策略买卖点", filtered_strats, index=default_idx, key="exp_strategy")
        sc2.number_input("KDJ n",    5,  30,   9,           key="exp_kdj_n")
        sc3.number_input("CCI n",    5, 100,  20,           key="exp_cci_n")
        sc4.number_input("唐奇安 n", 20, 500, 100,           key="exp_dc_n")
        sc5s.number_input("周期bars(周高低)", 2, 20, 5,      key="exp_week_n")

        sc6, sc7, sc8, _ = st.columns([1, 1, 1, 2])
        sc6.number_input("BBI回踩容忍(%)", 0.0, 2.0, 0.2, step=0.05, key="exp_bbi_eps")
        sc7.number_input("BBI反抽容忍(%)", 0.0, 2.0, 0.2, step=0.05, key="exp_bbi_feps")
        sc8.checkbox("BBI intrabar确认", value=True, key="exp_bbi_bo")


# ---------------------------------------------------------------------------
# Main entry point — called from trader/monitor.py inside tab_explore
# ---------------------------------------------------------------------------

def render_exploration_tab() -> None:
    """Render the full exploration panel. Controls are rendered inline at the top."""

    # Render controls first so all session_state keys are populated before we read them
    _render_exp_controls()

    # ── Read all settings from session_state ─────────────────────────────────
    initial_capital  = float(st.session_state.get("exp_initial_capital", INITIAL_CAPITAL))
    symbols_text     = st.session_state.get("exp_symbols", os.getenv("SYMBOLS", "AAPL,MSFT"))
    timeframe        = st.session_state.get("exp_tf", "30m")
    x_mode           = st.session_state.get("exp_xmode", "NO_GAP（index轴彻底无gap）")
    show_day_sep     = st.session_state.get("exp_day_sep", True)
    leverage         = float(st.session_state.get("exp_leverage", 1.0))
    autoscale        = st.session_state.get("exp_autoscale", False)
    window_n         = 90 if autoscale else int(st.session_state.get("exp_window", 240))
    ind_list         = list(st.session_state.get("exp_ind_list", ["EMA", "BBI"]))
    sma_n            = int(st.session_state.get("exp_sma_n",    20))
    ema_fast         = int(st.session_state.get("exp_ema_fast", 12))
    ema_slow         = int(st.session_state.get("exp_ema_slow", 26))
    bb_n             = int(st.session_state.get("exp_bb_n",     20))
    bb_k             = float(st.session_state.get("exp_bb_k",   2.0))
    rsi_n            = int(st.session_state.get("exp_rsi_n",    14))
    macd_fast        = int(st.session_state.get("exp_macd_f",   12))
    macd_slow        = int(st.session_state.get("exp_macd_s",   26))
    macd_sig         = int(st.session_state.get("exp_macd_sg",   9))
    atr_n            = int(st.session_state.get("exp_atr_n",    14))
    ta_n             = int(st.session_state.get("exp_ta_n",     14))
    selected_strategy = st.session_state.get("exp_strategy", "5/20均线金叉死叉")
    kdj_n            = int(st.session_state.get("exp_kdj_n",    9))
    cci_n            = int(st.session_state.get("exp_cci_n",   20))
    donchian_n       = int(st.session_state.get("exp_dc_n",   100))
    week_n           = int(st.session_state.get("exp_week_n",   5))
    bbi_eps          = float(st.session_state.get("exp_bbi_eps",  0.2)) / 100.0
    bbi_breakout     = bool(st.session_state.get("exp_bbi_bo",  True))
    bbi_fail_eps     = float(st.session_state.get("exp_bbi_feps", 0.2)) / 100.0

    # ── Parse symbols ─────────────────────────────────────────────────────────
    symbols = _parse_symbols(symbols_text)
    if not symbols:
        st.warning('请在"参数设置"中输入至少 1 个股票代码（例如 AAPL）。')
        return

    # ── Load historical data ──────────────────────────────────────────────────
    st.subheader(f"K线图（{timeframe}）")
    symbol_for_chart = st.selectbox("选择标的", options=symbols, index=0, key="exp_chart_sym")
    only_completed   = st.checkbox("只显示已完成K线（更稳定）", value=False, key="exp_only_done")

    with st.spinner(f"正在加载 {symbol_for_chart} 本地数据…"):
        df_hist = _fetch_bars_merged(symbol_for_chart, timeframe)

    if df_hist.empty:
        st.warning(f"本地暂无 **{symbol_for_chart} {timeframe}** 数据。"
                   "请先在侧边栏点击 **📥 下载/更新数据** 按钮。")
        return

    df_hist = df_hist.dropna(subset=["timestamp", "open", "high", "low", "close"]).reset_index(drop=True)
    df_hist = df_hist.sort_values("timestamp").reset_index(drop=True)
    if "timestamp_utc" not in df_hist.columns:
        df_hist["timestamp_utc"] = _to_utc_timestamp(df_hist["timestamp"])
    df_hist["timestamp_utc"] = pd.to_datetime(df_hist["timestamp_utc"], utc=True)

    if only_completed and timeframe != "1m" and len(df_hist) > 1:
        df_hist = df_hist.iloc[:-1].reset_index(drop=True)

    st.caption(f"数据来自: 本地缓存  ·  rows: {len(df_hist):,}  ·  "
               f"range(UTC): {df_hist['timestamp_utc'].min()} → {df_hist['timestamp_utc'].max()}")

    # ── Compute indicators ────────────────────────────────────────────────────
    df_hist_full = _compute_indicators(
        df_hist.copy().reset_index(drop=True),
        ind_list=ind_list, sma_n=int(sma_n), ema_fast=int(ema_fast), ema_slow=int(ema_slow),
        bb_n=int(bb_n), bb_k=float(bb_k), rsi_n=int(rsi_n),
        macd_fast=int(macd_fast), macd_slow=int(macd_slow), macd_signal=int(macd_sig),
        atr_n=int(atr_n), ta_n=int(ta_n),
    )
    df_hist_win = df_hist_full.tail(int(window_n)).copy().reset_index(drop=True)
    is_crypto   = _is_crypto(symbol_for_chart)

    # ── Shared strategy params (used by chart, backtest, AND sim) ─────────────
    base_params = dict(
        macd_fast=int(macd_fast), macd_slow=int(macd_slow), macd_signal=int(macd_sig),
        rsi_n=int(rsi_n), bb_n=int(bb_n), bb_k=float(bb_k),
        kdj_n=int(kdj_n), cci_n=int(cci_n), donchian_n=int(donchian_n), week_n=int(week_n),
        bbi_eps=float(bbi_eps), bbi_breakout=bool(bbi_breakout), bbi_fail_eps=float(bbi_fail_eps),
        ta_n=int(ta_n),
    )
    base_params_frozen = tuple(sorted(base_params.items()))

    df_sel = _build_strategy_signals(df_hist_win, selected_strategy, **base_params)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    chart_tab, backtest_tab, sim_tab = st.tabs(["📈 Main", "📊 历史回测", "🎬 历史模拟"])

    # =========================================================================
    # Main chart tab
    # =========================================================================
    with chart_tab:
        fig        = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                   vertical_spacing=0.02, row_heights=[0.78, 0.22])
        hover_time = _build_time_strings(df_hist_win, timeframe).to_numpy()

        # ── Compute x values once (NO_GAP = integers, DATE_AXIS = timestamps) ─
        if x_mode.startswith("NO_GAP"):
            x_vals      = list(range(len(df_hist_win)))
            x_ts_local  = None
        else:
            x_ts_local  = _utc_to_ny(df_hist_win["timestamp_utc"])
            x_vals      = x_ts_local

        # Compute signal marker positions
        buy_idx_list  = df_sel.index[df_sel["strat_signal"] == 1].tolist()
        sell_idx_list = df_sel.index[df_sel["strat_signal"] == -1].tolist()
        if x_mode.startswith("NO_GAP"):
            x_buy  = buy_idx_list
            x_sell = sell_idx_list
        else:
            x_buy  = x_ts_local.iloc[buy_idx_list].tolist()  if buy_idx_list  else []
            x_sell = x_ts_local.iloc[sell_idx_list].tolist() if sell_idx_list else []
        buy_px  = df_sel.loc[buy_idx_list,  "strat_exec_px"].tolist() if buy_idx_list  else []
        sell_px = df_sel.loc[sell_idx_list, "strat_exec_px"].tolist() if sell_idx_list else []

        # ── Candlestick ───────────────────────────────────────────────────────
        _candle_kw = {}
        if x_mode.startswith("NO_GAP"):
            _candle_kw = dict(
                customdata=hover_time,
                hovertemplate="Time: %{customdata}<br>O: %{open}<br>H: %{high}<br>L: %{low}<br>C: %{close}<extra></extra>",
            )
        fig.add_trace(go.Candlestick(
            x=x_vals, open=df_hist_win["open"], high=df_hist_win["high"],
            low=df_hist_win["low"], close=df_hist_win["close"], name="OHLC", **_candle_kw,
        ), row=1, col=1)

        # ── Volume ────────────────────────────────────────────────────────────
        _vol_colors = [
            "#26a69a" if float(df_hist_win["close"].iloc[i]) >= float(df_hist_win["open"].iloc[i])
            else "#ef5350" for i in range(len(df_hist_win))
        ]
        _vol_kw = {}
        if x_mode.startswith("NO_GAP"):
            _vol_kw = dict(customdata=hover_time,
                           hovertemplate="Time: %{customdata}<br>Vol: %{y}<extra></extra>")
        fig.add_trace(go.Bar(
            x=x_vals, y=df_hist_win["volume"].astype(float), name="Volume",
            opacity=0.35, marker_color=_vol_colors, **_vol_kw,
        ), row=2, col=1)

        # ── Indicator overlays (single code path for both axis modes) ─────────
        if "SMA"   in ind_list and f"sma_{int(sma_n)}" in df_hist_win.columns:
            fig.add_trace(go.Scatter(x=x_vals, y=df_hist_win[f"sma_{int(sma_n)}"],
                                     mode="lines", name=f"SMA{sma_n}"), row=1, col=1)
        if "EMA"   in ind_list:
            for _span, _c in [(ema_fast, f"ema_{int(ema_fast)}"), (ema_slow, f"ema_{int(ema_slow)}")]:
                if _c in df_hist_win.columns:
                    fig.add_trace(go.Scatter(x=x_vals, y=df_hist_win[_c],
                                             mode="lines", name=f"EMA{_span}"), row=1, col=1)
        if "BBANDS" in ind_list and {"bb_up", "bb_mid", "bb_dn"}.issubset(df_hist_win.columns):
            for _n, _c in [("BB Up", "bb_up"), ("BB Mid", "bb_mid"), ("BB Dn", "bb_dn")]:
                fig.add_trace(go.Scatter(x=x_vals, y=df_hist_win[_c],
                                         mode="lines", name=_n), row=1, col=1)
        if "BBI"   in ind_list and "bbi" in df_hist_win.columns:
            fig.add_trace(go.Scatter(x=x_vals, y=df_hist_win["bbi"],
                                     mode="lines", name="BBI"), row=1, col=1)

        # ── Buy / sell markers ────────────────────────────────────────────────
        if x_buy:
            _bkw = (dict(customdata=[hover_time[i] for i in buy_idx_list],
                         hovertemplate=f"{selected_strategy} BUY<br>Time: %{{customdata}}<br>Px: %{{y}}<extra></extra>")
                    if x_mode.startswith("NO_GAP")
                    else dict(hovertemplate=f"{selected_strategy} BUY<br>Px: %{{y}}<extra></extra>"))
            fig.add_trace(go.Scatter(x=x_buy, y=buy_px, mode="markers",
                                     name=f"{selected_strategy} BUY",
                                     marker=dict(symbol="triangle-up", size=12), **_bkw), row=1, col=1)
        if x_sell:
            _skw = (dict(customdata=[hover_time[i] for i in sell_idx_list],
                         hovertemplate=f"{selected_strategy} SELL<br>Time: %{{customdata}}<br>Px: %{{y}}<extra></extra>")
                    if x_mode.startswith("NO_GAP")
                    else dict(hovertemplate=f"{selected_strategy} SELL<br>Px: %{{y}}<extra></extra>"))
            fig.add_trace(go.Scatter(x=x_sell, y=sell_px, mode="markers",
                                     name=f"{selected_strategy} SELL",
                                     marker=dict(symbol="triangle-down", size=12), **_skw), row=1, col=1)

        # ── Axis configuration (only this part differs between modes) ─────────
        shapes = []
        if x_mode.startswith("NO_GAP"):
            tick_idx, tick_text = _tick_for_nogap(df_hist_win, timeframe)
            fig.update_xaxes(row=2, col=1, type="linear", tickmode="array",
                             tickvals=tick_idx, ticktext=tick_text, tickangle=0,
                             rangeslider=dict(visible=False))
            fig.update_xaxes(row=1, col=1, rangeslider=dict(visible=False))
            if timeframe != "1d" and show_day_sep:
                shapes = _make_shapes_nogap(_day_start_positions_idx(df_hist_win, timeframe))
        else:
            rb = _rangebreaks(timeframe, is_crypto)
            fig.update_xaxes(row=2, col=1, type="date", rangebreaks=rb,
                             rangeslider=dict(visible=not autoscale))
            fig.update_xaxes(row=1, col=1, rangeslider=dict(visible=False), rangebreaks=rb)
            if timeframe != "1d" and show_day_sep:
                dates  = x_ts_local.dt.date
                shapes = _make_shapes_date(x_ts_local.loc[dates.ne(dates.shift())].tolist())

        fig.update_layout(
            uirevision=f"{symbol_for_chart}-{timeframe}-{x_mode}",
            shapes=shapes, height=700,
            margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"),
        )
        fig.update_yaxes(title_text="Price",  row=1, col=1, fixedrange=False)
        fig.update_yaxes(title_text="Volume", row=2, col=1, fixedrange=False, showgrid=False)
        st.plotly_chart(fig, width="stretch", config={"scrollZoom": True, "displaylogo": False})

        # Strategy stats tables (全部 / YTD / 近30天)
        stats_df = _build_stats_table_exp(
            df_hist_full, tuple(STRATEGY_OPTIONS), base_params_frozen, float(leverage), timeframe,
        )
        st.markdown("### 策略历史统计")
        if stats_df.empty:
            st.info("没有可用统计（数据不足或策略无交易信号）。")
        elif "期间" not in stats_df.columns:
            st.dataframe(stats_df, width="stretch", hide_index=True)
        else:
            _col_cfg = {
                "最终收益": st.column_config.NumberColumn(label="最终收益", format="%.2f%%"),
                "胜率":     st.column_config.NumberColumn(label="胜率",     format="%.1f%%"),
                "最大回撤": st.column_config.NumberColumn(label="最大回撤", format="%.2f%%"),
            }
            for _prefix, _title in [("全部", "📊 全量历史"), ("YTD", "📅 今年（YTD）"), ("30天", "📆 近30天")]:
                _period_key = next((p for p in stats_df["期间"].unique() if p.startswith(_prefix)), None)
                if _period_key is None:
                    continue
                _pdf = stats_df[stats_df["期间"] == _period_key].drop(columns=["期间"]).copy()
                if _pdf.empty:
                    continue
                for _c in ["最终收益", "胜率", "最大回撤"]:
                    _pdf[_c] = pd.to_numeric(_pdf[_c], errors="coerce") * 100
                st.markdown(f"#### {_title}  <small style='color:gray'>({_period_key})</small>",
                            unsafe_allow_html=True)
                st.dataframe(_pdf, width="stretch", hide_index=True,
                             height=min(900, 32 * (len(_pdf) + 1) + 12),
                             column_config=_col_cfg)

    # =========================================================================
    # Backtest tab
    # =========================================================================
    with backtest_tab:
        st.subheader("历史回测 / 指标收益对比")
        st.caption("直接用本地缓存数据，对比多个策略的收益、胜率和回撤。"
                   f"  初始本金: ${initial_capital:,.0f}")

        backtest_window = st.selectbox("回测窗口", ["全量", "最近1000根", "最近360根"], index=0, key="exp_bt_window")
        backtest_source = {
            "全量":      df_hist_full,
            "最近1000根": df_hist_full.tail(1000).copy().reset_index(drop=True),
            "最近360根":  df_hist_full.tail(360).copy().reset_index(drop=True),
        }[backtest_window]

        default_compare = [selected_strategy, "全仓买入并持有", "分期定投(20次均匀)", "ADX趋势过滤(ta)", "Stoch超买超卖(ta)"]
        default_compare = [s for s in dict.fromkeys(default_compare) if s in STRATEGY_OPTIONS]
        compare_strategies = st.multiselect("对比策略", STRATEGY_OPTIONS, default=default_compare, key="exp_bt_strats")
        compare_strategies = [s for s in dict.fromkeys(compare_strategies) if s in STRATEGY_OPTIONS]

        if not compare_strategies:
            st.info("先选择至少一个策略，再查看历史回测结果。")
        else:
            history_stats = _build_stats_table_exp(
                backtest_source, tuple(compare_strategies), base_params_frozen, float(leverage), timeframe,
            )
            st.markdown("### 策略对比")
            display_history = history_stats.copy()
            for col in ["最终收益", "胜率", "最大回撤"]:
                display_history[col] = pd.to_numeric(display_history[col], errors="coerce") * 100
            st.dataframe(
                display_history, width="stretch", hide_index=True,
                column_config={
                    "最终收益": st.column_config.NumberColumn(label="最终收益", format="%.2f%%"),
                    "胜率":     st.column_config.NumberColumn(label="胜率",     format="%.1f%%"),
                    "最大回撤": st.column_config.NumberColumn(label="最大回撤", format="%.2f%%"),
                },
            )

            compare_strategy = st.selectbox("查看单个策略曲线", compare_strategies, index=0, key="exp_bt_sel")

            if compare_strategy == "分期定投(20次均匀)":
                _, _, exact_balance = _simulate_dca_20_balance(backtest_source, leverage=float(leverage),
                                                                initial_capital=initial_capital)
                exact_wins, exact_trades = 0, 0
                compare_returns, compare_equity = pd.Series(dtype=float), pd.Series(dtype=float)
            else:
                compare_df = backtest_source.copy().reset_index(drop=True)
                if compare_strategy != "全仓买入并持有":
                    compare_df = _build_strategy_signals(compare_df, compare_strategy, **base_params)
                exact_wins, exact_trades, exact_balance, _trets = _simulate_signal_strategy_mark_to_market(
                    compare_df, leverage=float(leverage), initial_capital=initial_capital,
                )
                compare_returns, compare_equity = _strategy_return_curve(
                    backtest_source, compare_strategy, base_params, float(leverage),
                    initial_capital=initial_capital,
                )

            sc = st.columns(4)
            sc[0].metric("最终余额",   f"${exact_balance:,.0f}")
            sc[1].metric("累计收益",   f"{(exact_balance / initial_capital - 1.0) * 100:.2f}%")
            sc[2].metric("胜率",       f"{exact_wins / exact_trades * 100:.1f}%" if exact_trades > 0 else "0.0%")
            if compare_strategy == "分期定投(20次均匀)":
                sc[3].metric("最大回撤", "N/A")
            else:
                dd = (compare_equity / compare_equity.cummax() - 1.0) if not compare_equity.empty else pd.Series(dtype=float)
                sc[3].metric("最大回撤", f"{dd.min() * 100:.2f}%" if not dd.empty else "N/A")

            if not compare_returns.empty and not compare_equity.empty:
                perf_df = _performance_summary_table(compare_returns, compare_equity, timeframe,
                                                     initial_capital=initial_capital)
                if not perf_df.empty:
                    pc = st.columns(3)
                    perf_map = {row["metric"]: row["value"] for _, row in perf_df.iterrows()}
                    pc[0].metric("CAGR",    f"{float(perf_map.get('年化收益(CAGR)', np.nan)) * 100:.2f}%"
                                            if pd.notna(perf_map.get('年化收益(CAGR)', np.nan)) else "N/A")
                    pc[1].metric("Sharpe",  f"{float(perf_map.get('夏普比率', np.nan)):.2f}"
                                            if pd.notna(perf_map.get('夏普比率', np.nan)) else "N/A")
                    pc[2].metric("Sortino", f"{float(perf_map.get('索提诺比率', np.nan)):.2f}"
                                            if pd.notna(perf_map.get('索提诺比率', np.nan)) else "N/A")

                perf_fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                         vertical_spacing=0.04, row_heights=[0.72, 0.28])
                perf_fig.add_trace(go.Scatter(x=compare_equity.index, y=compare_equity.values,
                                              name="Strategy Equity", line=dict(width=2)), row=1, col=1)
                bench_close = pd.to_numeric(backtest_source["close"], errors="coerce").astype(float)
                if len(bench_close) > 0 and np.isfinite(bench_close.iloc[0]) and bench_close.iloc[0] > 0:
                    bench    = initial_capital * (bench_close / float(bench_close.iloc[0]))
                    bench_ts = pd.to_datetime(backtest_source["timestamp_utc"], errors="coerce", utc=True)
                    perf_fig.add_trace(go.Scatter(x=bench_ts, y=bench, name="Buy & Hold Benchmark",
                                                  line=dict(width=1, dash="dot")), row=1, col=1)
                drawdown = compare_equity / compare_equity.cummax() - 1.0
                perf_fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values,
                                              name="Drawdown", fill="tozeroy", line=dict(width=1)), row=2, col=1)
                perf_fig.update_layout(height=620, margin=dict(l=10, r=10, t=30, b=10),
                                       legend=dict(orientation="h"))
                perf_fig.update_yaxes(title_text="Equity",   row=1, col=1)
                perf_fig.update_yaxes(title_text="Drawdown", row=2, col=1)
                st.plotly_chart(perf_fig, width="stretch", config={"scrollZoom": True, "displaylogo": False})
                st.dataframe(perf_df, width="stretch", hide_index=True)
            else:
                st.info("当前策略没有可画的连续收益曲线。")

            ta_cols = [c for c in ["adx", "adx_pos", "adx_neg", "stoch_k", "stoch_d", "willr", "mfi", "obv"]
                       if c in backtest_source.columns]
            if ta_cols:
                st.markdown("### 扩展指标快照")
                st.dataframe(backtest_source[["timestamp_utc", "close"] + ta_cols].tail(12),
                             width="stretch", hide_index=True)
            else:
                st.caption("如果你在上方勾选了 ADX / STOCH / WILLR / MFI / OBV，这里会显示最近几根的指标快照。")

    # =========================================================================
    # 历史模拟 tab
    # =========================================================================
    with sim_tab:
        st.subheader("🎬 历史模拟")
        st.caption("逐根放出K线，实时模拟策略买卖信号与账户盈亏。数据来自本地缓存。")

        with st.expander("⚙️ 模拟参数", expanded=True):
            sc1, sc2, sc3 = st.columns(3)
            sim_symbol   = sc1.text_input(
                "标的", value=st.session_state.get("exp_symbols", "QQQ").split(",")[0].strip().upper(),
                key="sim_symbol_input",
            )
            sim_strategy = sc2.selectbox(
                "策略", STRATEGY_OPTIONS,
                index=STRATEGY_OPTIONS.index("上周高低点(周K突破)"),
                key="sim_strategy_sel",
            )
            sim_tf = sc3.selectbox(
                "K线周期", ["1m", "15m", "30m", "4h", "1d"],
                index=2, key="sim_tf_sel",
            )
            sc4, sc5, sc6 = st.columns(3)
            sim_allow_short = sc4.checkbox("允许做空", value=False, key="sim_allow_short")
            sim_capital = sc5.number_input(
                "起始资金 ($)", min_value=1_000.0, max_value=10_000_000.0,
                value=float(st.session_state.get("exp_initial_capital", INITIAL_CAPITAL)),
                step=1_000.0, key="sim_capital_input",
            )
            sim_week_n = (
                sc6.number_input(
                    "周期bars数(周高低)", min_value=2, max_value=50, value=5, step=1,
                    key="sim_week_n_input", help="仅对「上周高低点」策略有效",
                )
                if sim_strategy == "上周高低点(周K突破)" else 5
            )

            sv1, sv2 = st.columns(2)
            sim_visible_n = sv1.slider("图表显示K线数", min_value=50, max_value=500, value=120,
                                       step=10, key="sim_visible_n")
            sim_speed = sv2.select_slider(
                "播放速度 (每次推进根数)",
                options=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
                value=st.session_state.get("sim_speed_val", 1),
                key="sim_speed_slider",
            )
            st.session_state["sim_speed_val"] = sim_speed

        btn_col1, btn_col2, btn_col3, _ = st.columns([1, 1, 1, 3])
        start_btn = btn_col1.button("▶ 开始", key="sim_start_btn", type="primary")
        pause_btn = btn_col2.button("⏸ 暂停", key="sim_pause_btn")
        reset_btn = btn_col3.button("⏹ 重置", key="sim_reset_btn")

        def _sim_reset_state():
            for k in ["sim_running", "sim_cursor", "sim_df", "sim_trades",
                      "sim_cash", "sim_position", "sim_entry_px", "sim_position_side",
                      "sim_long_wins", "sim_long_losses", "sim_short_wins", "sim_short_losses",
                      "sim_long_pnl", "sim_short_pnl", "sim_total_trades",
                      "sim_pending", "sim_bh_shares", "sim_bh_entry_px", "sim_loaded_key"]:
                st.session_state.pop(k, None)

        if reset_btn:
            _sim_reset_state(); st.rerun()
        if pause_btn:
            st.session_state["sim_running"] = False
        if start_btn:
            load_key = f"{sim_symbol}|{sim_tf}"
            if st.session_state.get("sim_loaded_key") != load_key:
                with st.spinner(f"正在加载 {sim_symbol} {sim_tf} 数据…"):
                    df_loaded = _sim_load_data(sim_symbol, sim_tf)
                if df_loaded is None or df_loaded.empty:
                    st.error(f"无法加载 {sim_symbol} {sim_tf} 数据，请检查网络或标的名称。")
                    st.stop()
                st.session_state["sim_df"]            = df_loaded
                st.session_state["sim_cursor"]        = 1
                st.session_state["sim_trades"]        = []
                st.session_state["sim_cash"]          = float(sim_capital)
                st.session_state["sim_position"]      = 0
                st.session_state["sim_position_side"] = 0
                st.session_state["sim_entry_px"]      = 0.0
                st.session_state["sim_long_wins"]     = 0
                st.session_state["sim_long_losses"]   = 0
                st.session_state["sim_short_wins"]    = 0
                st.session_state["sim_short_losses"]  = 0
                st.session_state["sim_long_pnl"]      = 0.0
                st.session_state["sim_short_pnl"]     = 0.0
                st.session_state["sim_total_trades"]  = 0
                st.session_state["sim_pending"]       = None
                _bh_entry  = float(df_loaded["close"].iloc[0])
                _bh_shares = float(sim_capital) / _bh_entry if _bh_entry > 0 else 0.0
                st.session_state["sim_bh_entry_px"]  = _bh_entry
                st.session_state["sim_bh_shares"]    = _bh_shares
                st.session_state["sim_loaded_key"]    = load_key
            st.session_state["sim_running"] = True

        sim_df           = st.session_state.get("sim_df")
        sim_cursor       = st.session_state.get("sim_cursor", 1)
        sim_running      = st.session_state.get("sim_running", False)
        sim_trades       = st.session_state.get("sim_trades", [])
        sim_cash         = st.session_state.get("sim_cash", float(sim_capital))
        sim_position     = st.session_state.get("sim_position", 0)
        sim_pos_side     = st.session_state.get("sim_position_side", 0)
        sim_entry_px     = st.session_state.get("sim_entry_px", 0.0)
        sim_long_wins    = st.session_state.get("sim_long_wins", 0)
        sim_long_losses  = st.session_state.get("sim_long_losses", 0)
        sim_short_wins   = st.session_state.get("sim_short_wins", 0)
        sim_short_losses = st.session_state.get("sim_short_losses", 0)
        sim_long_pnl     = st.session_state.get("sim_long_pnl", 0.0)
        sim_short_pnl    = st.session_state.get("sim_short_pnl", 0.0)
        sim_n_trades     = st.session_state.get("sim_total_trades", 0)
        sim_pending      = st.session_state.get("sim_pending", None)
        sim_bh_shares    = st.session_state.get("sim_bh_shares", 0.0)

        summary_ph = st.empty()
        chart_ph   = st.empty()
        st.markdown("---")
        st.markdown("##### 📋 交易记录")
        log_ph = st.empty()

        # ── sim_kwargs: now uses the same base_params as the main chart ────────
        # This ensures indicator parameters (RSI n, BB n, etc.) are consistent
        # between what the user sees on the chart and what the sim uses.
        sim_kwargs = {**base_params, "week_n": int(sim_week_n)}

        if sim_running and sim_df is not None and len(sim_df) > 0:
            n_total = len(sim_df)
            advance = min(int(sim_speed), n_total - sim_cursor)

            for _step in range(advance):
                sim_cursor += 1
                if sim_cursor > n_total:
                    break

                df_slice  = sim_df.iloc[:sim_cursor]
                bar_row   = df_slice.iloc[-1]
                bar_high  = float(bar_row["high"])
                bar_low   = float(bar_row["low"])
                bar_time  = str(bar_row["timestamp_utc"])[:16]
                bar_idx   = sim_cursor - 1
                bar_open  = float(bar_row["open"])
                just_closed_position = False

                if sim_pending is not None:
                    p_type  = sim_pending["type"]
                    p_sig   = sim_pending["sig"]
                    p_px    = sim_pending["exec_px"]
                    p_pside = sim_pending.get("pos_side", 0)

                    if p_type == "open" and sim_pos_side == 0:
                        cur_sig, _ = _sim_check_signal(df_slice, sim_strategy, sim_kwargs)
                        if cur_sig != 0 and cur_sig != p_sig:
                            sim_pending = None
                        elif bar_low <= p_px <= bar_high:
                            if p_sig > 0:
                                qty = int(sim_cash / p_px)
                                if qty >= 1:
                                    sim_cash    -= qty * p_px
                                    sim_position = qty; sim_pos_side = 1; sim_entry_px = p_px
                                    sim_trades.append(dict(
                                        时间=bar_time, direction="买入",
                                        exec_px=round(p_px, 4), qty=qty, pnl=0.0,
                                        账户总值=round(sim_cash + qty * p_px, 2), bar_idx=bar_idx,
                                    ))
                            elif p_sig < 0 and sim_allow_short:
                                qty = int(sim_cash / p_px)
                                if qty >= 1:
                                    sim_cash    -= qty * p_px
                                    sim_position = qty; sim_pos_side = -1; sim_entry_px = p_px
                                    sim_trades.append(dict(
                                        时间=bar_time, direction="做空",
                                        exec_px=round(p_px, 4), qty=qty, pnl=0.0,
                                        账户总值=round(sim_cash + qty * p_px, 2), bar_idx=bar_idx,
                                    ))
                            sim_pending = None

                    elif p_type == "close" and sim_pos_side == p_pside and sim_position > 0:
                        fill_px = bar_open
                        if fill_px is not None:
                            if p_pside == 1:
                                pnl = (fill_px - sim_entry_px) * sim_position
                                sim_cash += fill_px * sim_position
                                sim_long_pnl += pnl
                                if pnl > 0: sim_long_wins  += 1
                                else:       sim_long_losses += 1
                                sim_n_trades += 1
                                sim_trades.append(dict(
                                    时间=bar_time, direction="卖出",
                                    exec_px=round(fill_px, 4), qty=sim_position,
                                    pnl=round(pnl, 2), 账户总值=round(sim_cash, 2), bar_idx=bar_idx,
                                ))
                            else:
                                pnl = (sim_entry_px - fill_px) * sim_position
                                sim_cash += sim_entry_px * sim_position + pnl
                                sim_short_pnl += pnl
                                if pnl > 0: sim_short_wins  += 1
                                else:       sim_short_losses += 1
                                sim_n_trades += 1
                                sim_trades.append(dict(
                                    时间=bar_time, direction="回补",
                                    exec_px=round(fill_px, 4), qty=sim_position,
                                    pnl=round(pnl, 2), 账户总值=round(sim_cash, 2), bar_idx=bar_idx,
                                ))
                            sim_position = 0; sim_pos_side = 0; sim_entry_px = 0.0
                            sim_pending  = None; just_closed_position = True

                sig, exec_px = _sim_check_signal(df_slice, sim_strategy, sim_kwargs)

                if sig != 0:
                    if sig > 0 and sim_pos_side == -1 and sim_position > 0:
                        if sim_pending is not None and sim_pending["type"] == "open":
                            sim_pending = None
                        sim_pending = {"type": "close", "sig": sig, "exec_px": exec_px,
                                       "pos_side": -1, "bar_time": bar_time, "bar_idx": bar_idx}

                    elif sig < 0 and sim_pos_side == 1 and sim_position > 0:
                        if sim_pending is not None and sim_pending["type"] == "open":
                            sim_pending = None
                        sim_pending = {"type": "close", "sig": sig, "exec_px": exec_px,
                                       "pos_side": 1, "bar_time": bar_time, "bar_idx": bar_idx}

                    elif sim_pos_side == 0 and not just_closed_position:
                        if sig > 0 or (sig < 0 and sim_allow_short):
                            if sim_pending is not None and sim_pending["sig"] != sig:
                                sim_pending = None
                            if sim_pending is None:
                                sim_pending = {"type": "open", "sig": sig, "exec_px": exec_px,
                                               "bar_time": bar_time, "bar_idx": bar_idx}

            st.session_state.update({
                "sim_cursor":        sim_cursor,
                "sim_trades":        sim_trades,
                "sim_cash":          sim_cash,
                "sim_position":      sim_position,
                "sim_position_side": sim_pos_side,
                "sim_entry_px":      sim_entry_px,
                "sim_pending":       sim_pending,
                "sim_long_wins":     sim_long_wins,
                "sim_long_losses":   sim_long_losses,
                "sim_short_wins":    sim_short_wins,
                "sim_short_losses":  sim_short_losses,
                "sim_long_pnl":      sim_long_pnl,
                "sim_short_pnl":     sim_short_pnl,
                "sim_total_trades":  sim_n_trades,
            })

            if sim_cursor >= n_total:
                st.session_state["sim_running"] = False; sim_running = False

        if sim_df is not None and len(sim_df) > 0 and sim_cursor > 0:
            df_now     = sim_df.iloc[:sim_cursor]
            last_close = float(df_now["close"].iloc[-1]) if len(df_now) > 0 else 0.0

            if sim_pos_side == 1 and sim_position > 0:
                cur_equity = sim_cash + sim_position * last_close
            elif sim_pos_side == -1 and sim_position > 0:
                unrealised = (sim_entry_px - last_close) * sim_position
                cur_equity = sim_cash + sim_position * sim_entry_px + unrealised
            else:
                cur_equity = sim_cash

            initial_cap   = float(st.session_state.get("sim_capital_input", sim_capital))
            pnl_dollar    = cur_equity - initial_cap
            pnl_pct       = (cur_equity / initial_cap - 1.0) * 100.0 if initial_cap > 0 else 0.0
            pos_label     = "多头" if sim_pos_side == 1 else ("空头" if sim_pos_side == -1 else "空仓")
            progress_pct  = int(sim_cursor / max(len(sim_df), 1) * 100)

            bh_equity     = sim_bh_shares * last_close if sim_bh_shares > 0 else initial_cap
            bh_pnl_dollar = bh_equity - initial_cap
            bh_pnl_pct    = (bh_equity / initial_cap - 1.0) * 100.0 if initial_cap > 0 else 0.0
            vs_bh_dollar  = cur_equity - bh_equity
            vs_bh_pct     = pnl_pct - bh_pnl_pct

            long_total   = sim_long_wins  + sim_long_losses
            short_total  = sim_short_wins + sim_short_losses
            total_wins   = sim_long_wins  + sim_short_wins
            total_closed = long_total + short_total

            realised_pnl   = sim_long_pnl + sim_short_pnl
            unrealised_pnl = pnl_dollar - realised_pnl

            def _wr(w, t):         return f"{w/t*100:.1f}%" if t > 0 else "—"
            def _pnl_fmt(v):       return f"${v:+,.2f}" if v != 0 else "$0.00"
            def _pnl_pct_fmt(v, c): return f"{v/c*100:+.2f}%" if c > 0 else "—"

            with summary_ph.container():
                st.progress(progress_pct,
                            text=f"进度 {sim_cursor}/{len(sim_df)} 根K线  "
                                 f"{'▶ 运行中' if sim_running else '⏸ 已暂停' if sim_cursor > 1 else '⏹ 未开始'}")
                r1 = st.columns(6)
                r1[0].metric("起始资金",   f"${initial_cap:,.0f}")
                r1[1].metric("当前账户值", f"${cur_equity:,.2f}")
                r1[2].metric("总收益 $",   f"${pnl_dollar:+,.2f}", help="总收益 = 已实现盈亏 + 浮动盈亏")
                r1[3].metric("总收益 %",   f"{pnl_pct:+.2f}%")
                r1[4].metric("持有胜率 💰", f"{bh_pnl_pct:+.2f}%", f"${bh_pnl_dollar:+,.2f}")
                r1[5].metric("超越持有",   f"{vs_bh_pct:+.2f}%",  f"${vs_bh_dollar:+,.2f}")

                r2 = st.columns(8)
                r2[0].metric("持仓方向", pos_label)
                r2[1].metric("总胜率",   _wr(total_wins, total_closed), f"{total_wins}/{total_closed} 笔")
                r2[2].metric("做多胜率", _wr(sim_long_wins, long_total),   f"{sim_long_wins}/{long_total} 笔")
                r2[3].metric("做空胜率", _wr(sim_short_wins, short_total), f"{sim_short_wins}/{short_total} 笔")
                r2[4].metric("做多收益", _pnl_fmt(sim_long_pnl),  _pnl_pct_fmt(sim_long_pnl,  initial_cap), help="已平仓的做多盈亏")
                r2[5].metric("做空收益", _pnl_fmt(sim_short_pnl), _pnl_pct_fmt(sim_short_pnl, initial_cap), help="已平仓的做空盈亏")
                r2[6].metric("浮动盈亏", _pnl_fmt(unrealised_pnl), _pnl_pct_fmt(unrealised_pnl, initial_cap), help="当前持仓的未实现盈亏")
                r2[7].metric("已平仓次数", str(total_closed))
        else:
            with summary_ph.container():
                st.info("点击「▶ 开始」加载数据并开始模拟。")

        if sim_df is not None and sim_cursor > 1:
            df_now = sim_df.iloc[:sim_cursor].copy().reset_index(drop=True)
            if "timestamp_utc" not in df_now.columns:
                df_now["timestamp_utc"] = df_now.get("timestamp", pd.Series(range(len(df_now))))
            chart_ph.plotly_chart(
                _sim_render_chart(df_now, sim_trades, int(sim_visible_n), sim_tf),
                width="stretch", config={"scrollZoom": True, "displaylogo": False},
            )

        if sim_trades:
            log_df = pd.DataFrame(sim_trades)[["时间", "direction", "exec_px", "qty", "pnl", "账户总值"]].copy()
            log_df = log_df.rename(columns={"direction": "方向", "exec_px": "成交价", "qty": "数量", "pnl": "平仓盈亏"})
            log_ph.dataframe(
                log_df.iloc[::-1].reset_index(drop=True), width="stretch", hide_index=True,
                column_config={
                    "成交价":   st.column_config.NumberColumn(format="%.4f"),
                    "平仓盈亏": st.column_config.NumberColumn(format="$%.2f"),
                    "账户总值": st.column_config.NumberColumn(format="$%.2f"),
                },
            )
        else:
            log_ph.caption("暂无交易记录。")

        if sim_running and sim_df is not None and sim_cursor < len(sim_df):
            import time as _time
            if sim_speed <= 5:
                _time.sleep(max(0.0, 0.9 - sim_speed * 0.1))
            st.rerun()
