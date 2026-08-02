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


def send_direction_review_push(symbols: Iterable[str]):
    return _send_all([build_direction_review_message(symbols)])


def send_stock_analysis_push(symbol: str, ai_db_path: str | None = None):
    return _send_all([build_stock_analysis_message(symbol, ai_db_path=ai_db_path)])


def send_open_confirmation_push(db_path: str | None = None):
    """手动补发开盘确认。

    周期语义见 report_period：交易日已开始就对账当天，否则对账上一个交易日。
    """
    return _send_all([build_open_confirmation_push_message(db_path=db_path)])


def send_close_report_push(db_path: str | None = None):
    """手动补发收盘报告（复盘 + 研究）。"""
    return _send_all([build_close_report_push_message(db_path=db_path)])


def send_weekly_report_push(db_path: str | None = None):
    """手动补发周报。本周未结束时只统计到此刻，并在正文标注。"""
    from .report_period import period_hours, resolve_weekly_period
    from .teams.maintenance import run_maintenance

    period = resolve_weekly_period(datetime.now(timezone.utc))
    run_maintenance(
        db_path=db_path or str(_ROOT / "trade.duckdb"),
        period_hours=period_hours(period),
        send_discord=True,
        period_label=period.label + ("-partial" if period.is_partial else ""),
    )
    from .notify import DeliveryOutcome

    # run_maintenance 内部自己发送，这里只把结果语义传回给 UI
    return DeliveryOutcome("SENT")


def build_open_confirmation_push_message(db_path: str | None = None) -> Notification:
    """按当前周期组装一份开盘确认。"""
    from .brief_review import BriefCallStore
    from .data_cache import get_bars
    from .intraday_levels import compute_intraday_levels
    from .morning_brief import _fetch_index_technicals
    from .open_confirmation import (
        OPEN_RANGE_MINUTES,
        build_open_confirmation_message,
    )
    from .report_period import resolve_daily_period

    period = resolve_daily_period(datetime.now(timezone.utc))
    resolved_db = db_path or str(_ROOT / "trade.duckdb")

    levels = {}
    for name in ("SPY", "QQQ"):
        try:
            bars = get_bars(name, "5m")
            if bars is None or getattr(bars, "empty", True):
                continue
            computed = compute_intraday_levels(
                name, bars, open_range_minutes=OPEN_RANGE_MINUTES
            )
            if computed is not None:
                levels[name] = computed
        except Exception as exc:  # noqa: BLE001
            logger.debug("开盘确认取 %s 分钟数据失败: %s", name, exc)

    if not levels:
        return Notification(
            title=f"🎯 开盘确认 · {period.label}",
            body="缺少 SPY/QQQ 的分钟级数据，无法计算开盘区间与触发位对账。",
            kind="review",
            fields={"交易日": period.label},
            dedupe_key=f"open_confirmation:{period.label}:nodata",
        )

    call = {}
    try:
        call = BriefCallStore(resolved_db).get(period.label) or {}
    except Exception as exc:  # noqa: BLE001
        logger.debug("读取晨报方向失败: %s", exc)

    note = build_open_confirmation_message(
        trading_date=period.label,
        index_levels=levels,
        technicals=_fetch_index_technicals(),
        morning_bias=call.get("bias", ""),
    )
    if period.note:
        note.body = period.note + "\n\n" + note.body
    return note


def build_close_report_push_message(db_path: str | None = None) -> Notification:
    """按当前周期组装一份收盘报告。

    自动路径会等研究批次；手动补发不等——按下按钮的人要的是现在就看到，所以
    研究没跑完就如实写明缺席原因。
    """
    from .brief_review import BriefCallStore
    from .close_report import (
        build_close_report,
        build_direction_review_line,
        build_unfilled_plans_line,
    )
    from .discord_report import build_review_body
    from .report_period import resolve_daily_period
    from .review import SimpleReviewer

    period = resolve_daily_period(datetime.now(timezone.utc))
    resolved_db = db_path or str(_ROOT / "trade.duckdb")

    try:
        report = SimpleReviewer(db_path=resolved_db).review(
            period="daily", as_of=period.end
        )
        trade_count = report.attribution.get("trade_count", len(report.trades))
        body, fields = build_review_body(
            today=period.label,
            pnl=report.portfolio_pnl,
            trade_count=trade_count,
            trades=report.trades,
            market_summary=getattr(report, "market_summary", ""),
        )
    except Exception as exc:  # noqa: BLE001
        return Notification(
            title=f"📋 收盘报告 · {period.label}",
            body=f"复盘数据读取失败（{type(exc).__name__}），本次没有可用内容。",
            kind="review",
            fields={"交易日": period.label},
            dedupe_key=f"close_report:{period.label}:failed",
        )

    extras = []
    try:
        call = BriefCallStore(resolved_db).get(period.label)
        ohlc = _session_ohlc(_load_bars("SPY"))
        extras.append(build_direction_review_line(call, ohlc))
    except Exception as exc:  # noqa: BLE001
        logger.debug("方向复盘失败: %s", exc)

    try:
        from .signal_reports import SignalStore

        invalid = [
            r
            for r in SignalStore(resolved_db).recent(limit=50)
            if getattr(getattr(r, "state", ""), "value", "") == "INVALIDATED"
        ]
        line = build_unfilled_plans_line(invalid)
        if line:
            extras.append(line)
    except Exception as exc:  # noqa: BLE001
        logger.debug("未成交计划读取失败: %s", exc)

    research_body, missing = "", ""
    try:
        from .daily_discord import build_daily_research_message
        from .daily_research import DailyResearchStore

        store = DailyResearchStore(str(_ROOT / "ai_states.duckdb"))
        run = store.latest_run(period.label)
        if run is not None and str(getattr(run, "status", "")).upper() not in (
            "RUNNING",
            "PENDING",
            "",
        ):
            research_body = build_daily_research_message(run, store.items(run.run_id)).body
        elif run is not None:
            missing = "研究批次仍在进行中"
        else:
            missing = f"{period.label} 没有研究批次记录"
    except Exception as exc:  # noqa: BLE001
        missing = f"研究结果读取失败（{type(exc).__name__}）"

    full_body = "\n\n".join([body] + [e for e in extras if e])
    if period.note:
        full_body = period.note + "\n\n" + full_body

    note = build_close_report(
        trading_date=period.label,
        review_body=full_body,
        review_fields=fields,
        research_body=research_body,
        research_missing_reason=missing,
    )
    if period.is_partial:
        # 未收盘的补发和收盘后的正式报告是两份不同的东西，不能共用身份，
        # 否则盘中补发一次就会把当天正式的收盘报告顶掉。
        note.dedupe_key = f"close_report:{period.label}:partial"
    return note


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
    db_path: str | None = None,
) -> Notification:
    """对账晨报当天真实给出的方向判断。

    方向不再由人从下拉框里选。让使用者在收盘后凭印象挑一个方向再打分，本质上
    是在给一个事后编造的判断评分——最坏的情况是挑一个事后看对的方向，让复盘显
    得很漂亮。真正要回答的问题是"今天早上那句话说对了没有"，所以方向只能来自
    BriefCallStore 里当天记录下来的原话；没有记录就如实说没有，而不是默认一个
    "中性"糊弄过去。
    """
    from .brief_review import BriefCallStore
    from .report_period import resolve_daily_period

    parsed = _parse_symbols(symbols)
    now = datetime.now(timezone.utc)
    period = resolve_daily_period(now)

    call = None
    try:
        call = BriefCallStore(db_path or str(_ROOT / "trade.duckdb")).get(period.label)
    except Exception as exc:  # noqa: BLE001
        logger.debug("读取晨报方向失败: %s", exc)

    if not call or not call.get("bias"):
        return Notification(
            title=f"晨报方向复盘 · {period.label}",
            body=(
                f"{period.label} 没有记录到晨报方向判断，无法复盘。\n"
                "方向来自当天晨报生成时的真实判断；那天如果没有发过晨报，就没有"
                "可对账的对象。"
            ),
            kind="review",
            fields={"交易日": period.label},
            dedupe_key=f"direction_review:{period.label}:none",
        )

    bias = call["bias"]
    reviews = []
    for symbol in parsed[:_MAX_REVIEW_SYMBOLS]:
        bars = _load_bars(symbol)
        ohlc = _session_ohlc(bars)
        if ohlc is None:
            reviews.append(f"{symbol}: 缺少当日常规时段 bars，暂无法评分。")
            continue
        stale_prefix = ""
        if ohlc["session_date"].isoformat() != period.label:
            stale_prefix = f"⚠️ 数据非本期（最新只有 {ohlc['session_date']} 的数据）— "
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

    lines = [f"早上的判断：{bias}"]
    if period.note:
        lines.append(period.note)
    lines.append("")
    lines.extend(f"• {line}" for line in reviews)

    return Notification(
        title=f"晨报方向复盘 · {period.label}",
        body="\n".join(lines),
        kind="review",
        fields={"交易日": period.label, "数据": "实时拉取（失败时回退本地缓存）"},
        dedupe_key=f"direction_review:{period.label}",
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

    notifier = make_notifier()
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
