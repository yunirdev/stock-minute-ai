"""收盘报告 —— 把"今天怎么样"和"明天做什么"合成一份。

以前收盘后是两条独立的推送：16:00 的每日复盘，和一个从来没接线过的每日研究
播报（daily_research 的所有生产调用点都传 notifier=None）。合并成一份是刻意
的选择，代价是要等研究批次跑完。

等待有个显而易见的风险：研究批次耗时不确定，可能拖很久，也可能整批失败。如
果直接串行等待，一次研究故障就会把复盘也一起拖没——而复盘只依赖本地账本，本
来是必定能产出的。所以这里做成两阶段：

    16:05  复盘数据算好，落在内存里，先不发
           ↓ 每个 tick 检查研究批次
    研究完成  → 合并成完整报告发出
    17:00 仍未完成 → 只发复盘部分，并写明 AI 研究为什么缺席

复盘永远发得出去，研究只是可选增强。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from .models import Notification

logger = logging.getLogger(__name__)

_SEPARATOR = "━" * 22


@dataclass
class PendingCloseReport:
    """已经算好、等着和研究批次合并的复盘。"""

    trading_date: str
    review_body: str
    fields: dict[str, str] = field(default_factory=dict)
    deadline: Optional[datetime] = None

    def expired(self, now: datetime) -> bool:
        return self.deadline is not None and now >= self.deadline


def build_direction_review_line(
    call: Optional[dict[str, Any]],
    session: Optional[dict[str, Any]],
) -> str:
    """晨报方向对账 —— 闭环的最后一环。

    晨报每天都给可证伪的判断，但在此之前没有任何东西回来结算它。打分逻辑
    （evaluate_direction_call）早就写好了，只是一直挂在需要人手动选方向的按
    钮上，因为程序不知道早上说过什么；BriefCallStore 补上了那半块。
    """
    if call is None:
        return "晨报方向复盘：今天没有记录到晨报方向判断（晨报可能未生成）。"
    bias = call.get("bias") or "—"
    if not session:
        return f"晨报方向复盘：早上判断「{bias}」，但缺少当日常规时段行情，无法评分。"

    from .brief_review import evaluate_direction_call, format_brief_call_review

    try:
        review = evaluate_direction_call(
            bias=bias,
            session_open=float(session["open"]),
            session_close=float(session["close"]),
            session_high=session.get("high"),
            session_low=session.get("low"),
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("方向复盘评分失败: %s", exc)
        return f"晨报方向复盘：早上判断「{bias}」，评分失败（{type(exc).__name__}）。"
    return f"早上判断「{bias}」 → {format_brief_call_review(review)}"


def build_unfilled_plans_line(reports: Any) -> str:
    """未成交计划的结局。

    INVALIDATED 不走盘中推送（那是事后信息，不是行动信号），但读者仍然需要
    知道"早上挂出去的计划最后怎么了"，否则一个信号就这么无声无息地消失了。
    """
    items = list(reports or [])
    if not items:
        return ""
    lines = ["**未成交计划的结局**"]
    for report in items[:8]:
        symbol = getattr(report, "symbol", "—")
        state = getattr(getattr(report, "state", ""), "value", "")
        if state == "INVALIDATED":
            lines.append(f"• {symbol}：计划到期未触发，已失效")
        else:
            lines.append(f"• {symbol}：{state}")
    if len(items) > 8:
        lines.append(f"（另有 {len(items) - 8} 个未列出）")
    return "\n".join(lines)


def build_overnight_risk_line(
    positions: Any,
    earnings_by_symbol: Optional[dict[str, str]] = None,
) -> str:
    """过夜持仓的风险提示。

    收盘时最该被提醒的一件事：手里这些仓位要过夜，其中哪些今晚或明天有财报。
    """
    held = [getattr(p, "symbol", None) or str(p) for p in (positions or [])]
    held = [s for s in held if s]
    if not held:
        return "过夜持仓：无（空仓过夜）"
    earnings = earnings_by_symbol or {}
    lines = [f"过夜持仓：{len(held)} 个 —— {', '.join(held)}"]
    flagged = [(s, earnings[s]) for s in held if s in earnings]
    for symbol, when in flagged:
        lines.append(f"  ⚠️ {symbol} {when} 有财报，过夜风险显著放大")
    return "\n".join(lines)


def build_health_line(
    *,
    tick_count: int,
    halted: bool,
    halt_reason: str = "",
    data_failures: int = 0,
) -> str:
    """系统健康摘要。

    "没消息就是没事"只有在系统确实活着的时候才成立。把这一行放进每天必发的
    报告里，就等于每天有一次确认——引擎哑掉的那天，这行不会出现。
    """
    parts = [f"今日 tick {tick_count} 次"]
    if data_failures:
        parts.append(f"数据获取失败 {data_failures} 次")
    parts.append(f"风控熔断：{halt_reason or '是'}" if halted else "无熔断")
    return "系统健康：" + " · ".join(parts)


def build_close_report(
    *,
    trading_date: str,
    review_body: str,
    review_fields: Optional[dict[str, str]] = None,
    research_body: str = "",
    research_missing_reason: str = "",
) -> Notification:
    """把复盘和研究拼成一份收盘报告。

    research_missing_reason 非空时如实写明 AI 研究为什么缺席——读者据此知道
    "今天没有明日计划"是系统没跑出来，而不是模型认为无事可做。这两种情况对
    第二天的操作意味着完全不同的事。
    """
    lines = [
        f"{_SEPARATOR} 今天怎么样 {_SEPARATOR}",
        review_body,
    ]
    if research_body:
        lines += ["", f"{_SEPARATOR} 明天做什么 {_SEPARATOR}", research_body]
    elif research_missing_reason:
        lines += [
            "",
            f"{_SEPARATOR} 明天做什么 {_SEPARATOR}",
            f"⚠️ 本次没有 AI 研究结论：{research_missing_reason}",
            "（这说明研究批次没跑出结果，不代表模型认为明天无机会。）",
        ]
    return Notification(
        title=f"📋 收盘报告 · {trading_date}",
        body="\n".join(lines),
        kind="review",
        fields=dict(review_fields or {}),
        dedupe_key=f"close_report:{trading_date}",
    )


def research_wait_deadline(
    now: datetime,
    *,
    minutes: int = 55,
) -> datetime:
    """等研究批次的截止时刻。

    默认到 17:00 ET 左右（16:05 起算 55 分钟）。过了就先把复盘发出去——晚一
    点的完整报告不如按时到达的半份报告有用，何况缺席的原因会写清楚。
    """
    return now + timedelta(minutes=minutes)


def _utc(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
