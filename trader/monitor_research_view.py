"""HTML renderer for the live daily-research/Runtime monitor card."""
from __future__ import annotations

from datetime import datetime, timezone
from html import escape
from typing import Any


_RUN_ERRORS = {
    "NO_ELIGIBLE_DEEP_CANDIDATES": "没有可进入深度研究的候选（仅有市场锚点或当前明确避开的标的）",
}


def live_research_html(
    snapshot: dict[str, Any] | None,
    *,
    now: datetime | None = None,
    runtime_stale_after_seconds: int = 180,
) -> str:
    snapshot = snapshot or {}
    now = now or datetime.now(timezone.utc)
    research = snapshot.get("research") or {}
    run = research.get("run")
    runtime = snapshot.get("runtime")
    if run is None and runtime is None:
        return (
            '<div class="qa-note">还没有每日研究或 Runtime 实时状态。'
            "引擎启动后会自动刷新。</div>"
        )
    parts = ['<div style="display:grid;gap:10px">']
    if run is not None:
        status = escape(str(run.get("status") or "—"))
        error_code = str(run.get("error_code") or "")
        error = escape(_RUN_ERRORS.get(error_code, error_code))
        parts.append(
            '<div class="qa-note">'
            f"研究交易日 <b>{escape(str(run.get('trading_date') or '—'))}</b> · "
            f"状态 <b>{status}</b> · "
            f"深度候选 {int(run.get('total_symbols') or 0)} · "
            f"完成 {int(run.get('completed_symbols') or 0)} · "
            f"失败 {int(run.get('failed_symbols') or 0)} · "
            f"批次 <code>{escape(str(run.get('run_id') or '—'))}</code>"
            + (f"<br>错误：{error}" if error else "")
            + "</div>"
        )
        parts.append(_research_rows(research.get("items") or []))
    if runtime is not None:
        blocked = bool(runtime.get("reconciliation_blocked"))
        kill = bool(runtime.get("kill_switch"))
        stale = _runtime_stale(
            runtime.get("updated_at"),
            now=now,
            stale_after_seconds=runtime_stale_after_seconds,
        )
        if blocked or kill:
            health = "阻断"
            color = "var(--neg)"
        elif stale:
            health = "已停止或心跳过期"
            color = "var(--fg3)"
        else:
            health = "运行正常"
            color = "var(--pos)"
        parts.append(
            '<div class="qa-note">'
            f"Runtime <b style=\"color:{color}\">{health}</b> · "
            f"市场 {escape(str(runtime.get('session') or '—'))} · "
            f"Tick {int(runtime.get('tick_count') or 0)} · "
            f"权益 ${float(runtime.get('equity') or 0):,.2f} · "
            f"更新 {escape(str(runtime.get('updated_at') or '—'))}"
            "</div>"
        )
        parts.append(_runtime_rows(runtime.get("candidates") or []))
    parts.append("</div>")
    return "".join(parts)


def _runtime_stale(
    updated_at: Any,
    *,
    now: datetime,
    stale_after_seconds: int,
) -> bool:
    if stale_after_seconds <= 0 or not isinstance(updated_at, str):
        return True
    try:
        parsed = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
    except ValueError:
        return True
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return True
    if now.tzinfo is None or now.utcoffset() is None:
        now = now.replace(tzinfo=timezone.utc)
    age = (now.astimezone(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds()
    return age < 0 or age > stale_after_seconds


def _research_rows(rows: list[dict[str, Any]]) -> str:
    body = []
    for row in rows[:10]:
        score = row.get("ai_score")
        score_text = f"{float(score):.1f}" if score is not None else "—"
        risks = row.get("risks") or []
        explanation = str(row.get("error_code") or "")
        if not explanation and risks:
            explanation = str(risks[0])
        body.append(
            "<tr>"
            f"<td>{int(row.get('rank') or 0)}</td>"
            f"<td><b>{escape(str(row.get('symbol') or ''))}</b></td>"
            f"<td>{escape(str(row.get('status') or ''))}</td>"
            f"<td>{escape(str(row.get('screening_status') or '—'))}</td>"
            f"<td>{escape(str(row.get('recommendation') or '—'))}</td>"
            f"<td>{score_text}</td>"
            f"<td>{float(row.get('screening_score') or 0):.1f}</td>"
            f"<td>{escape(explanation or '—')}</td>"
            "</tr>"
        )
    if not body:
        return '<div class="qa-note">研究批次尚未产生候选结果。</div>'
    return (
        '<div style="overflow-x:auto"><table class="qa-table" style="width:100%">'
        "<thead><tr><th>#</th><th>标的</th><th>研究状态</th>"
        "<th>初筛状态</th><th>结论</th><th>深度分</th><th>初筛分</th>"
        "<th>风险 / 错误</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def _runtime_rows(rows: list[dict[str, Any]]) -> str:
    body = []
    for row in rows[:12]:
        price = row.get("price")
        distance = row.get("distance_to_entry_pct")
        score = row.get("research_score")
        body.append(
            "<tr>"
            f"<td><b>{escape(str(row.get('symbol') or ''))}</b></td>"
            f"<td>{escape(str(row.get('state') or ''))}</td>"
            f"<td>{f'${float(price):,.2f}' if price is not None else '—'}</td>"
            f"<td>{f'{float(distance):+.2f}%' if distance is not None else '—'}</td>"
            f"<td>{f'{float(score):.1f}' if score is not None else '—'}</td>"
            f"<td>{float(row.get('position_qty') or 0):g}</td>"
            "</tr>"
        )
    if not body:
        return '<div class="qa-note">Runtime 尚未上报候选股票状态。</div>'
    return (
        '<div style="overflow-x:auto"><table class="qa-table" style="width:100%">'
        "<thead><tr><th>标的</th><th>状态</th><th>价格</th>"
        "<th>距入场</th><th>研究分</th><th>持仓</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )
