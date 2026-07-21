from __future__ import annotations

from datetime import datetime
from typing import Iterable, List

import pandas as pd

from .brief_review import evaluate_direction_call, format_brief_call_review
from .intraday_levels import build_intraday_followup, compute_intraday_levels
from .models import Notification
from .notify import make_notifier


def send_intraday_levels_push(symbols: Iterable[str]) -> bool:
    return _send_all(build_intraday_levels_messages(symbols))


def send_direction_review_push(symbols: Iterable[str], bias: str = "中性") -> bool:
    return _send_all([build_direction_review_message(symbols, bias=bias)])


def build_intraday_levels_messages(symbols: Iterable[str]) -> List[Notification]:
    parsed = _parse_symbols(symbols)
    lines = []
    for symbol in parsed[:8]:
        levels = _load_intraday_levels(symbol)
        lines.append(build_intraday_followup(levels))

    if not lines:
        lines = ["没有可用标的。请先在系统页标的输入框填写股票代码。"]

    return [
        Notification(
            title=f"盘中 OR/VWAP 跟踪 · {datetime.now():%m/%d %H:%M}",
            body="\n".join(f"• {line}" for line in lines),
            kind="alert",
            fields={"用途": "盘中手动跟踪", "数据": "本地分钟线"},
        )
    ]


def build_direction_review_message(
    symbols: Iterable[str],
    bias: str = "中性",
) -> Notification:
    parsed = _parse_symbols(symbols)
    reviews = []
    for symbol in parsed[:5]:
        bars = _load_bars(symbol)
        ohlc = _session_ohlc(bars)
        if ohlc is None:
            reviews.append(f"{symbol}: 缺少当日常规时段 bars，暂无法评分。")
            continue
        review = evaluate_direction_call(
            bias=bias,
            session_open=ohlc["open"],
            session_close=ohlc["close"],
            session_high=ohlc["high"],
            session_low=ohlc["low"],
        )
        reviews.append(f"{symbol}: {format_brief_call_review(review)}")

    if not reviews:
        reviews = ["没有可用标的。请先在系统页标的输入框填写股票代码。"]

    return Notification(
        title=f"晨报方向复盘 · {datetime.now():%m/%d %H:%M}",
        body="\n".join(f"• {line}" for line in reviews),
        kind="review",
        fields={"复盘方向": bias, "数据": "本地分钟线"},
    )


def _send_all(messages: List[Notification]) -> bool:
    notifier = make_notifier()
    ok = True
    for message in messages:
        ok = notifier.send(message) and ok
    return ok


def _parse_symbols(symbols: Iterable[str]) -> list[str]:
    out = []
    for item in symbols:
        for raw in str(item or "").split(","):
            symbol = raw.strip().upper()
            if symbol and symbol not in out:
                out.append(symbol)
    return out


def _load_intraday_levels(symbol: str):
    bars = _load_bars(symbol)
    if bars.empty:
        return None
    timeframe = 1 if len(bars) >= 15 else 5
    return compute_intraday_levels(
        symbol,
        bars,
        open_range_minutes=15,
        tz_name="America/New_York",
    ) or compute_intraday_levels(
        symbol,
        bars,
        open_range_minutes=timeframe * 3,
        tz_name="America/New_York",
    )


def _load_bars(symbol: str) -> pd.DataFrame:
    from .data_cache import get_bars

    for timeframe in ("1m", "5m", "15m"):
        try:
            df = get_bars(symbol, timeframe)
            if df is not None and not df.empty:
                return df
        except Exception:
            pass
    return pd.DataFrame()


def _session_ohlc(bars: pd.DataFrame) -> dict | None:
    if bars is None or bars.empty:
        return None

    df = bars.copy()
    df.columns = [str(col).lower() for col in df.columns]
    if "timestamp_utc" in df.columns:
        ts = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    elif "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    else:
        ts = pd.to_datetime(df.index, utc=True, errors="coerce")

    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):
        return None

    regular = df.assign(_ts=ts).dropna(subset=["_ts", "open", "high", "low", "close"])
    if regular.empty:
        return None

    local = regular["_ts"].dt.tz_convert("America/New_York")
    minutes = local.dt.hour * 60 + local.dt.minute
    regular = regular[(minutes >= 9 * 60 + 30) & (minutes < 16 * 60)].copy()
    if regular.empty:
        return None

    local = local.loc[regular.index]
    session_date = local.dt.date.max()
    session = regular[local.dt.date == session_date].sort_values("_ts")
    if session.empty:
        return None

    return {
        "open": float(session["open"].iloc[0]),
        "high": float(session["high"].max()),
        "low": float(session["low"].min()),
        "close": float(session["close"].iloc[-1]),
    }
