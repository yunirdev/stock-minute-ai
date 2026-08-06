"""HTML renderer for the live daily-research monitor card."""
from __future__ import annotations

from html import escape
from typing import Any


_RUN_ERRORS = {
    "NO_ELIGIBLE_DEEP_CANDIDATES": "没有可进入深度研究的候选（仅有市场锚点或当前明确避开的标的）",
}

# 说明栏该讲"结果"，不是罗列每个标的都有的通用状态——这两条对每个还没跑
# 完深度分析的标的都成立，混进摘要里看不出任何区别。
_BOILERPLATE_RISKS = {
    "缺少最新 AI 综合分",
    "缺少当前环境的可靠 holdout 统计",
}

_STATUS_LABELS = {
    "SCREENED": "候补，未进深度分析",
    "PENDING": "排队中",
    "RUNNING": "分析中",
}

# error_code 原始值是给日志/排障用的技术代码（比如内部快照写入失败留下的
# "死链接"类错误），说明栏是给人看的简报，要翻译成一句话，不能直接甩代码。
_ERROR_LABELS = {
    "TRADINGAGENTS_SNAPSHOT_LINK_UNAVAILABLE": "内部数据写入失败，本次未分析",
    "TRADINGAGENTS_SNAPSHOT_RUN_MISMATCH": "内部数据写入失败，本次未分析",
    "TRADINGAGENTS_SNAPSHOT_SYMBOL_MISMATCH": "内部数据写入失败，本次未分析",
    "TRADINGAGENTS_PYTHON_UNAVAILABLE": "本地分析环境未就绪",
    "TRADINGAGENTS_MODULE_UNAVAILABLE": "本地分析环境未就绪",
    "TRADINGAGENTS_TIMEOUT": "分析超时",
    "TRADINGAGENTS_WORKER_UNAVAILABLE": "分析进程无法启动",
    "TRADINGAGENTS_WORKER_OUTPUT_INVALID": "分析结果格式异常",
}

# TRADINGAGENTS_WORKER_FAILED:{ExceptionType}:{diagnostic_code} —— 最后一段是
# tradingagents_worker.py 自己分类过的诊断码，直接给对应的人话。
_WORKER_DIAGNOSIS_LABELS = {
    "LLM_API_CONNECTION_UNAVAILABLE": "连不上本地大模型服务",
    "LLM_API_TIMEOUT": "大模型响应超时",
    "LLM_API_HTTP_5XX": "大模型服务报错",
    "MODEL_NOT_FOUND": "配置的模型不存在",
    "CACHE_DATABASE_UNAVAILABLE": "本地缓存数据库打不开",
    "DEPENDENCY_LOAD_BLOCKED": "依赖库加载被拦截",
    "WORKER_FAILURE": "分析进程内部错误",
}


def live_research_html(snapshot: dict[str, Any] | None) -> str:
    snapshot = snapshot or {}
    research = snapshot.get("research") or {}
    run = research.get("run")
    if run is None:
        return (
            '<div class="qa-note">还没有每日研究状态。'
            "引擎启动后会自动刷新。</div>"
        )
    status = escape(str(run.get("status") or "—"))
    error_code = str(run.get("error_code") or "")
    error = escape(_RUN_ERRORS.get(error_code, error_code))
    parts = ['<div style="display:grid;gap:10px">']
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
    parts.append("</div>")
    return "".join(parts)


def _error_note(error_code: str) -> str:
    if error_code in _ERROR_LABELS:
        return _ERROR_LABELS[error_code]
    if error_code.startswith("TRADINGAGENTS_WORKER_FAILED:"):
        diagnosis = error_code.rsplit(":", 1)[-1]
        return _WORKER_DIAGNOSIS_LABELS.get(diagnosis, "分析进程失败")
    return _RUN_ERRORS.get(error_code, error_code)


def _briefing(row: dict[str, Any]) -> str:
    """说明栏内容——已完成的给 AI 自己的结论摘要，失败的给错误原因，
    还没跑完的给一句话状态，不掺风险标签这类"每行都一样"的噪音。"""
    status = str(row.get("status") or "")
    if status == "COMPLETED":
        thesis = str(row.get("thesis") or "").strip()
        if not thesis:
            risks = [
                str(r) for r in (row.get("risks") or []) if str(r) not in _BOILERPLATE_RISKS
            ]
            return "；".join(risks) if risks else "—"
        # trader_investment_plan/final_trade_decision 是 AI 自己写好的简报
        # （结论+理由+入场/止损/仓位），几百字，值得完整显示——不是当年那种
        # 塞满整段辩论记录的长文，80 字截断反而把关键信息切没了。留个上限
        # 只是防止极端情况撑爆页面，不是常规截断。
        limit = 600
        return thesis[:limit] + ("…" if len(thesis) > limit else "")
    if status == "FAILED":
        error_code = str(row.get("error_code") or "")
        return _error_note(error_code) if error_code else "分析失败，原因未知"
    return _STATUS_LABELS.get(status, status or "—")


def _briefing_html(row: dict[str, Any]) -> str:
    """说明栏内容可以是 AI 写的多行简报——先转义防注入，再把换行变成
    <br>，两步顺序不能反：先转义再插 <br> 才不会把我们自己加的标签也转义掉。"""
    return escape(_briefing(row)).replace("\n", "<br>")


def _research_rows(rows: list[dict[str, Any]]) -> str:
    body = []
    for row in rows[:10]:
        status = str(row.get("status") or "")
        completed = status == "COMPLETED"
        recommendation = row.get("recommendation")
        score = row.get("ai_score")
        conclusion = (
            f"{escape(str(recommendation))} · {float(score):.1f} 分"
            if completed and recommendation and score is not None
            else "—"
        )
        symbol = str(row.get("symbol") or "")
        # 点标的名去"研究档案"看完整多空辩论——NiceGUI 的 ui.html() 会用
        # DOMPurify 清洗内容，onclick 这种内联事件属性会被直接剥掉（实测验证
        # 过），所以这里只留 data-symbol，真正的点击监听是 monitor_nice.py
        # 里用 ui.run_javascript() 挂的一次性事件委托，不受清洗影响。
        symbol_cell = (
            f'<a class="qa-symbol-link" data-symbol="{escape(symbol, quote=True)}">'
            f"<b>{escape(symbol)}</b></a>"
        )
        body.append(
            "<tr>"
            f"<td>{symbol_cell}</td>"
            f"<td>{conclusion}</td>"
            f"<td>{_briefing_html(row)}</td>"
            "</tr>"
        )
    if not body:
        return '<div class="qa-note">研究批次尚未产生候选结果。</div>'
    return (
        '<div style="overflow-x:auto"><table class="qa-table qa-table-research" style="width:100%">'
        "<thead><tr><th>标的</th><th>AI 结论</th><th>说明</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )
