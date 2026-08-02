"""Discord messages for daily research and runtime signal state changes."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .discord_limits import (
    EMBED_TOTAL_MAX,
    FIELD_VALUE_MAX,
    allocate_budget,
    truncate_text,
)
from .models import Notification


def _trunc(value: Any, length: int) -> str:
    text = str(value or "")
    return text[:length] + "…" if len(text) > length else text


def _fmt_time(value: Any) -> str:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return str(value or "—")


def build_daily_research_message(run: Any, items: list[Any]) -> Notification:
    completed = [item for item in items if getattr(item, "status", "") == "COMPLETED"]
    completed.sort(
        key=lambda item: (
            -float(getattr(item, "ai_score", 0.0) or 0.0),
            int(getattr(item, "rank", 999) or 999),
        )
    )
    failed = [item for item in items if getattr(item, "status", "") == "FAILED"]
    lines = [
        f"研究交易日：**{getattr(run, 'trading_date', '—')}**",
        f"批次：`{getattr(run, 'run_id', '—')}`",
        "",
        "以下标的先经过本地量价筛选，再由 TradingAgents 做深度研究。",
    ]
    fields = {
        "研究状态": str(getattr(run, "status", "—")),
        "完成 / 失败": (
            f"{getattr(run, 'completed_symbols', 0)} / "
            f"{getattr(run, 'failed_symbols', 0)}"
        ),
        "数据截止": _fmt_time(getattr(run, "data_cutoff", None)),
    }
    # thesis 的篇幅按实际内容动态分配，而不是每条都硬切 180 字：几个标的都写得
    # 短时一个字都不裁，有人写得特别长时才逐步压缩，且压缩是均摊的。分配总额
    # 取自还没被上面几个固定字段占掉的 embed 额度。
    shown = completed[:5]
    if shown:
        theses = [str(getattr(item, "thesis", "") or "") for item in shown]
        used = sum(len(k) + len(v) for k, v in fields.items()) + len("\n".join(lines))
        pool = max(
            len(shown) * 80,
            EMBED_TOTAL_MAX - used - 400,  # 400 给标题和每条的分数行留白
        )
        quotas = allocate_budget([len(t) for t in theses], pool, min_each=80)
    else:
        theses, quotas = [], []

    for item, thesis, quota in zip(shown, theses, quotas):
        recommendation = str(getattr(item, "recommendation", "HOLD")).upper()
        icon = {"BUY": "🟢", "SELL": "🔴"}.get(recommendation, "🟡")
        score = float(getattr(item, "ai_score", 0.0) or 0.0)
        screening = float(getattr(item, "screening_score", 0.0) or 0.0)
        headline = f"深度分 {score:.1f} · 初筛 {screening:.1f}"
        # 单个 field.value 上限 1024，扣掉分数行才是 thesis 的真实可用额度
        room = min(quota, FIELD_VALUE_MAX - len(headline) - 2)
        body = truncate_text(thesis, room) if thesis else "无摘要"
        fields[f"{icon} {getattr(item, 'symbol', '—')} · {recommendation}"] = (
            f"{headline}\n{body}"
        )
    if not completed:
        lines.extend(["", "⚠️ 本批次没有产生可验证的深度研究结果。"])
    if failed:
        lines.extend(
            [
                "",
                "失败标的："
                + "、".join(str(getattr(item, "symbol", "")) for item in failed[:8]),
            ]
        )
    lines.extend(
        [
            "",
            "盘中系统只监控实时价格、触发条件、持仓和订单；不会反复重跑完整模型。",
        ]
    )
    return Notification(
        title=f"🧠 每日股票研究 · {getattr(run, 'trading_date', '')}",
        body="\n".join(lines),
        kind="ai",
        fields=fields,
    )


def build_signal_report_message(report: Any) -> Notification:
    state = str(
        getattr(getattr(report, "state", ""), "value", getattr(report, "state", ""))
    )
    side = str(
        getattr(getattr(report, "side", ""), "value", getattr(report, "side", ""))
    )
    symbol = str(getattr(report, "symbol", "—"))
    state_style = {
        "READY": ("🟢", "等待入场"),
        "ENTERED": ("✅", "已经成交"),
        "HOLD": ("📌", "持仓跟踪"),
        "EXIT": ("🟠", "等待退出"),
        "CLOSED": ("🏁", "交易结束"),
        "INVALIDATED": ("⚪", "计划失效"),
        "RISK_BLOCKED": ("🔴", "风控阻断"),
        "DATA_STALE": ("⚠️", "数据过期"),
        "ORDER_ERROR": ("🔴", "订单异常"),
    }
    icon, label = state_style.get(state, ("🔵", state or "状态更新"))
    lines = [
        f"{icon} **{label}** · {side}",
        f"研究批次：`{getattr(report, 'ai_run_id', '') or '—'}`",
        "",
        (
            f"建议入场区间：**${float(getattr(report, 'entry_low', 0.0)):,.2f}"
            f" – ${float(getattr(report, 'entry_high', 0.0)):,.2f}**"
        ),
        f"追价上限：**${float(getattr(report, 'chase_limit', 0.0)):,.2f}**",
        f"止损：**${float(getattr(report, 'stop_loss', 0.0)):,.2f}**",
        f"止盈：**${float(getattr(report, 'take_profit', 0.0)):,.2f}**",
    ]
    if getattr(report, "realized_pnl", None) is not None:
        lines.extend(
            [
                "",
                (
                    f"已实现盈亏：**${float(report.realized_pnl):+,.2f}** "
                    f"({float(report.realized_return_pct or 0.0):+.2f}%)"
                ),
            ]
        )
    reasons = list(getattr(report, "reasons", []) or [])
    if reasons:
        lines.extend(
            ["", "依据：" + "、".join(_trunc(item, 50) for item in reasons[:5])]
        )
    return Notification(
        title=f"{icon} {symbol} · {label}",
        body="\n".join(lines),
        kind=(
            "alert"
            if state in {"RISK_BLOCKED", "DATA_STALE", "ORDER_ERROR"}
            else "plan"
        ),
        fields={
            "模型仓位": f"{float(getattr(report, 'model_weight_pct', 0.0)):.1f}%",
            "单笔风险": f"{float(getattr(report, 'model_risk_pct', 0.0)):.2f}%",
            "风险收益比": f"{float(getattr(report, 'risk_reward', 0.0)):.2f}",
            "AI 分数": (
                f"{float(report.ai_score):.1f}"
                if getattr(report, "ai_score", None) is not None
                else "—"
            ),
        },
        plan_id=str(getattr(report, "plan_id", "") or ""),
    )


def build_exception_message(
    *,
    error_code: str,
    summary: str,
    evidence_refs: list[str],
    occurred_at: datetime,
) -> Notification:
    """Build a bounded operational exception brief without credentials."""
    return Notification(
        title=f"⚠️ 系统异常 · {_trunc(error_code, 80)}",
        body="\n".join(
            (
                f"发生时间：{_fmt_time(occurred_at)}",
                f"摘要：{_trunc(summary, 500)}",
                "证据：" + ("、".join(evidence_refs[:10]) or "—"),
            )
        ),
        kind="alert",
        fields={"错误代码": _trunc(error_code, 100)},
    )
