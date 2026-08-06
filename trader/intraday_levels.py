from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd


@dataclass(frozen=True)
class IntradayLevels:
    symbol: str
    session_date: str
    as_of: str
    open_range_minutes: int
    open_range_high: float
    open_range_low: float
    session_high: float
    session_low: float
    last_price: float
    vwap: Optional[float]
    close_vs_vwap_pct: Optional[float]
    state: str
    is_stale: bool


def compute_intraday_levels(
    symbol: str,
    bars: pd.DataFrame,
    open_range_minutes: int = 15,
    tz_name: str = "America/New_York",
) -> Optional[IntradayLevels]:
    df = _prepare_bars(bars, tz_name)
    if df.empty:
        return None

    regular = _regular_session_rows(df)
    if regular.empty:
        return None

    session_date = regular["_local_ts"].dt.date.max()
    session = regular[regular["_local_ts"].dt.date == session_date].copy()
    if session.empty:
        return None

    open_range = session[session["_minutes_from_open"] < open_range_minutes]
    if open_range.empty:
        return None

    open_range_high = float(open_range["high"].max())
    open_range_low = float(open_range["low"].min())
    last = float(session["close"].iloc[-1])
    vwap = _session_vwap(session)
    close_vs_vwap_pct = None
    if vwap and vwap > 0:
        close_vs_vwap_pct = (last - vwap) / vwap * 100

    # A cache that hasn't been refreshed since a prior trading day still has
    # "regular session" rows, so the checks above alone can't catch it — the
    # caller must be told this isn't today's session before it presents these
    # numbers as a live "现价".
    today_local = datetime.now(timezone.utc).astimezone(ZoneInfo(tz_name)).date()
    is_stale = session_date != today_local

    return IntradayLevels(
        symbol=symbol.upper(),
        session_date=str(session_date),
        as_of=session["_local_ts"].iloc[-1].strftime("%m/%d %H:%M %Z"),
        open_range_minutes=open_range_minutes,
        open_range_high=open_range_high,
        open_range_low=open_range_low,
        session_high=float(session["high"].max()),
        session_low=float(session["low"].min()),
        last_price=last,
        vwap=vwap,
        close_vs_vwap_pct=close_vs_vwap_pct,
        state=_classify_state(last, open_range_high, open_range_low, vwap),
        is_stale=is_stale,
    )


def build_intraday_followup(levels: IntradayLevels | None, symbol: str = "") -> str:
    if levels is None:
        label = f"{symbol.upper()} " if symbol else ""
        return f"{label}盘中 OR/VWAP 暂不可用：缺少当日常规时段分钟线。"

    vwap_text = "VWAP 不可用"
    if levels.vwap is not None and levels.close_vs_vwap_pct is not None:
        vwap_text = f"VWAP {levels.vwap:.2f}（现价 {levels.close_vs_vwap_pct:+.2f}%）"

    stale_prefix = ""
    if levels.is_stale:
        stale_prefix = f"⚠️ 数据非当日（本地缓存最新仅到 {levels.session_date}，非实时）— "

    return (
        f"{stale_prefix}{levels.symbol} {levels.as_of}："
        f"OR{levels.open_range_minutes} {levels.open_range_low:.2f}-{levels.open_range_high:.2f}；"
        f"{vwap_text}；"
        f"现价 {levels.last_price:.2f}；"
        f"{_state_text(levels.state)}"
    )


def _prepare_bars(bars: pd.DataFrame, tz_name: str) -> pd.DataFrame:
    if bars is None or len(bars) == 0:
        return pd.DataFrame()

    df = bars.copy()
    df.columns = [str(col).lower() for col in df.columns]
    if "timestamp_utc" in df.columns:
        ts = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    elif "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    else:
        ts = pd.to_datetime(df.index, utc=True, errors="coerce")

    required = {"high", "low", "close"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    out = df.assign(_ts=ts).dropna(subset=["_ts", "high", "low", "close"])
    if out.empty:
        return out

    if "volume" not in out.columns:
        out["volume"] = 0.0
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0.0)
    for col in ("high", "low", "close"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["high", "low", "close"]).sort_values("_ts")
    out["_local_ts"] = out["_ts"].dt.tz_convert(ZoneInfo(tz_name))
    return out


def _regular_session_rows(df: pd.DataFrame) -> pd.DataFrame:
    start_minute = 9 * 60 + 30
    end_minute = 16 * 60
    minutes = df["_local_ts"].dt.hour * 60 + df["_local_ts"].dt.minute
    regular = df[(minutes >= start_minute) & (minutes < end_minute)].copy()
    regular["_minutes_from_open"] = minutes.loc[regular.index] - start_minute
    return regular


def _session_vwap(session: pd.DataFrame) -> Optional[float]:
    volume = session["volume"].sum()
    if not volume or volume <= 0:
        return None
    typical = (session["high"] + session["low"] + session["close"]) / 3
    return float((typical * session["volume"]).sum() / volume)


def _classify_state(
    last: float,
    open_range_high: float,
    open_range_low: float,
    vwap: Optional[float],
) -> str:
    above_vwap = vwap is None or last >= vwap
    below_vwap = vwap is None or last <= vwap
    if last > open_range_high and above_vwap:
        return "bull_breakout"
    if last < open_range_low and below_vwap:
        return "bear_breakdown"
    if last >= open_range_low and last <= open_range_high:
        return "inside_open_range"
    if vwap is not None and last >= vwap:
        return "above_vwap"
    if vwap is not None and last < vwap:
        return "below_vwap"
    return "mixed"


def _state_text(state: str) -> str:
    """直接给出 OR/VWAP 状态对应的操作结论，不做模糊化处理——用户自行
    判断是否采纳，但结论本身应该是明确的，不是"偏多但……"这种两头都占。"""
    labels = {
        "bull_breakout": "站上开盘区间且高于 VWAP：多头结构，可考虑做多，回踩 OR 上沿/VWAP 未跌破前有效。",
        "bear_breakdown": "跌破开盘区间且低于 VWAP：空头结构，可考虑做空，反抽 OR 下沿/VWAP 未收复前有效。",
        "inside_open_range": "仍在开盘区间内，方向未定，不建议现在进场，等突破确认。",
        "above_vwap": "高于 VWAP 但未突破 OR 上沿：偏多但结构还没走完，突破 OR 上沿再做多，现在进场胜率不够。",
        "below_vwap": "低于 VWAP 但未跌破 OR 下沿：偏空但结构还没走完，跌破 OR 下沿再做空，现在进场胜率不够。",
        "mixed": "信号矛盾（价格/VWAP/OR 三者不一致），不建议现在进场。",
    }
    return labels.get(state, labels["mixed"])
