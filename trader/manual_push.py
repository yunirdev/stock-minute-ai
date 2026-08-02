from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List
from zoneinfo import ZoneInfo

import pandas as pd

from .brief_review import evaluate_direction_call, format_brief_call_review
from .intraday_levels import build_intraday_followup, compute_intraday_levels
from .models import Notification
from .notify import make_notifier
from .research_monitor import daily_research_monitor

_ROOT = Path(__file__).resolve().parents[1]
_AI_DB_PATH = str(_ROOT / "ai_states.duckdb")
_ET_TZ = ZoneInfo("America/New_York")
logger = logging.getLogger(__name__)


def send_intraday_levels_push(symbols: Iterable[str]) -> bool:
    return _send_all(build_intraday_levels_messages(symbols))


def send_direction_review_push(symbols: Iterable[str], bias: str = "中性") -> bool:
    return _send_all([build_direction_review_message(symbols, bias=bias)])


def send_stock_analysis_push(symbol: str, ai_db_path: str | None = None) -> bool:
    return _send_all([build_stock_analysis_message(symbol, ai_db_path=ai_db_path)])


_MAX_INTRADAY_SYMBOLS = 8
_MAX_REVIEW_SYMBOLS = 5


def build_intraday_levels_messages(symbols: Iterable[str]) -> List[Notification]:
    parsed = _parse_symbols(symbols)
    lines = []
    for symbol in parsed[:_MAX_INTRADAY_SYMBOLS]:
        levels = _load_intraday_levels(symbol)
        lines.append(build_intraday_followup(levels, symbol=symbol))
    if len(parsed) > _MAX_INTRADAY_SYMBOLS:
        omitted = parsed[_MAX_INTRADAY_SYMBOLS:]
        lines.append(f"（还有 {len(omitted)} 个标的未处理：{', '.join(omitted)}，单次最多跟踪 {_MAX_INTRADAY_SYMBOLS} 个）")

    if not lines:
        lines = ["没有可用标的。请先在系统页标的输入框填写股票代码。"]

    return [
        Notification(
            title=f"盘中 OR/VWAP 跟踪 · {datetime.now():%m/%d %H:%M}",
            body="\n".join(f"• {line}" for line in lines),
            kind="alert",
            fields={"用途": "盘中手动跟踪", "数据": "实时拉取（失败时回退本地缓存，回退会在正文标注）"},
        )
    ]


def build_direction_review_message(
    symbols: Iterable[str],
    bias: str = "中性",
) -> Notification:
    parsed = _parse_symbols(symbols)
    reviews = []
    today_et = datetime.now(timezone.utc).astimezone(_ET_TZ).date()
    for symbol in parsed[:_MAX_REVIEW_SYMBOLS]:
        bars = _load_bars(symbol)
        ohlc = _session_ohlc(bars)
        if ohlc is None:
            reviews.append(f"{symbol}: 缺少当日常规时段 bars，暂无法评分。")
            continue
        stale_prefix = ""
        if ohlc["session_date"] != today_et:
            stale_prefix = f"⚠️ 数据非当日（最新只有 {ohlc['session_date']} 的数据）— "
        review = evaluate_direction_call(
            bias=bias,
            session_open=ohlc["open"],
            session_close=ohlc["close"],
            session_high=ohlc["high"],
            session_low=ohlc["low"],
        )
        reviews.append(
            f"{stale_prefix}{symbol} ({ohlc['session_date']}): {format_brief_call_review(review)}"
        )
    if len(parsed) > _MAX_REVIEW_SYMBOLS:
        omitted = parsed[_MAX_REVIEW_SYMBOLS:]
        reviews.append(f"（还有 {len(omitted)} 个标的未处理：{', '.join(omitted)}，单次最多复盘 {_MAX_REVIEW_SYMBOLS} 个）")

    if not reviews:
        reviews = ["没有可用标的。请先在系统页标的输入框填写股票代码。"]

    return Notification(
        title=f"晨报方向复盘 · {datetime.now():%m/%d %H:%M}",
        body="\n".join(f"• {line}" for line in reviews),
        kind="review",
        fields={"复盘方向": bias, "数据": "实时拉取（失败时回退本地缓存）"},
    )


def build_stock_analysis_message(
    symbol: str, ai_db_path: str | None = None
) -> Notification:
    """Build a manual push of today's frozen AI research conclusion for one symbol."""
    normalized = symbol.strip().upper()
    if not normalized:
        return Notification(
            title="个股分析",
            body="未提供标的代码。",
            kind="alert",
        )
    monitor = daily_research_monitor(ai_db_path or _AI_DB_PATH)
    item = next(
        (it for it in monitor["items"] if it["symbol"].upper() == normalized),
        None,
    )
    if item is None:
        return Notification(
            title=f"📈 {normalized} 个股分析",
            body=f"今日研究结果中没有 {normalized} 的结论（未入选筛选/深度研究，或今日研究尚未运行）。",
            kind="alert",
            fields={"标的": normalized},
        )
    status = item.get("status", "—")
    if status in ("RUNNING", "PENDING"):
        # 研究还没跑完，item 里的 recommendation/ai_score/confidence 全是
        # 未填充的占位默认值（比如 recommendation 默认 "WATCH"）——把这当
        # 成最终结论推出去等于给了一个假结论。
        run = monitor.get("run") or {}
        return Notification(
            title=f"📈 {normalized} 个股分析 · 研究进行中",
            body=(
                f"今日研究仍在进行中（状态：{status}），还没有最终结论。\n"
                f"筛选分：{_fmt_val(item.get('screening_score'))}"
                "（这是初筛分，不是最终 AI 结论）"
            ),
            kind="alert",
            fields={"标的": normalized, "研究批次": run.get("run_id", "—")},
        )
    risks = item.get("risks") or []
    lines = [
        f"研究状态：{status}",
        f"筛选结论：{item.get('screening_status', '—')}",
        f"深度结论：{item.get('recommendation', '—')}",
        f"综合分：{_fmt_val(item.get('ai_score'))}　筛选分：{_fmt_val(item.get('screening_score'))}",
        f"置信度：{_fmt_val(item.get('confidence'))}",
    ]
    if item.get("thesis"):
        lines.append(f"依据：{item['thesis']}")
    if risks:
        lines.append("风险：" + "；".join(str(r) for r in risks))
    if item.get("error_code"):
        lines.append(f"错误码：{item['error_code']}")
    run = monitor.get("run") or {}
    return Notification(
        title=f"📈 {normalized} 个股分析 · {run.get('trading_date', '—')}",
        body="\n".join(lines),
        kind="review",
        fields={"标的": normalized, "研究批次": run.get("run_id", "—")},
    )


def _fmt_val(value) -> str:
    """dict.get(key, '—') only falls back when the key is missing — a key
    present with value None (e.g. an in-progress research row) still prints
    literal "None" into the Discord message. This is the honest placeholder."""
    return "—" if value is None else str(value)


def _send_all(messages: List[Notification]):
    """返回汇总后的 DeliveryOutcome —— 真值语义是"确实送到 Discord 了"。"""
    from .notify import summarize

    notifier = make_notifier(external_send_enabled=True)
    return summarize([notifier.send(message) for message in messages])


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
    """Bars for a manual "live" push must reflect the actual market, not
    whatever happens to sit in the local research cache — that cache is
    filled only for symbols the running engine currently tracks and can be
    weeks stale for anything else (get_bars() never fetches over the
    network by design). Try a live window fetch first; fall back to the
    local cache — with staleness now surfaced by compute_intraday_levels —
    only if the live fetch fails (e.g. no network / API key)."""
    from .data_cache import fetch_alpaca_bars_window, fetch_and_save, get_bars

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=5)  # covers weekends/holidays back to last session
    try:
        # 1m has no Alpaca window endpoint (free-plan limitation) — fetch_and_save
        # already scopes 1m to a light 7-day yfinance pull, so it's cheap here too.
        df = fetch_and_save(symbol, "1m")
        if df is not None and not df.empty:
            return df
    except Exception as exc:
        logger.debug("manual_push live fetch %s 1m failed: %s", symbol, exc)
    for timeframe in ("5m", "15m"):
        try:
            df = fetch_alpaca_bars_window(symbol, timeframe, start, end)
            if df is not None and not df.empty:
                return df
        except Exception as exc:
            logger.debug("manual_push live fetch %s %s failed: %s", symbol, timeframe, exc)

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
        "session_date": session_date,
    }
