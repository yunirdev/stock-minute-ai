"""Read-only order explanations and durable NiceGUI action contracts."""
from __future__ import annotations

import hashlib
import html
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb

from .audit_query import order_traces


@dataclass(frozen=True)
class ButtonActionContract:
    action_id: str
    label: str
    category: str
    important: bool = False
    external_send: bool = False


def _contracts() -> tuple[ButtonActionContract, ...]:
    raw = [
        ("overview.refresh_regime", "刷新市场环境", "overview", False, False),
        ("overview.run_research", "运行今日研究", "overview", False, False),
        *[
            (f"overview.span_{key}", label, "overview", False, False)
            for key, label in (
                ("1d", "1D"), ("1w", "1W"), ("1m", "1M"),
                ("3m", "3M"), ("ytd", "YTD"), ("all", "All"),
            )
        ],
        ("system.start", "启动引擎", "system", True, False),
        ("system.stop", "停止", "system", True, False),
        ("discord.morning", "立即发送晨报", "discord", True, True),
        ("discord.intraday", "发送盘中 OR/VWAP", "discord", True, True),
        ("discord.review", "立即发送复盘", "discord", True, True),
        ("discord.direction", "发送方向复盘", "discord", True, True),
        ("factors.run", "分析", "research", False, False),
        ("backtest.run", "运行回测", "research", False, False),
        ("pool.rebuild_all", "全部重建", "pool", False, False),
        ("pool.full_scan", "全市场扫描", "pool", False, False),
        ("pool.long_term", "更新长期池", "pool", False, False),
        ("pool.decision", "更新决策池", "pool", False, False),
        ("pool.send_decision", "送到决策台", "pool", False, False),
        ("pool.clear_filter", "清除筛选", "pool", False, False),
        ("agent.generate", "生成每日候选", "agent", False, False),
        ("agent.load", "载入每日候选", "agent", False, False),
        ("agent.run", "运行一轮", "agent", False, False),
        ("agent.reset", "重置", "agent", False, False),
        ("agent.export", "导出 JSON", "agent", False, False),
        ("selection.run", "运行选股", "research", False, False),
        ("risk.trigger", "触发急停", "risk", True, False),
        ("risk.clear", "解除急停", "risk", True, False),
        ("maintenance.run", "运行维护分析", "maintenance", False, False),
    ]
    return tuple(ButtonActionContract(*values) for values in raw)


BUTTON_ACTIONS = _contracts()
BUTTON_ACTIONS_BY_ID = {item.action_id: item for item in BUTTON_ACTIONS}
ACTION_TERMINAL_STATES = {"SUCCESS", "EMPTY", "ERROR"}


class ButtonActionAuditStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        connection = duckdb.connect(self.db_path)
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ui_action_audit (
                    action_run_id TEXT PRIMARY KEY,
                    action_id TEXT,
                    request_id TEXT,
                    state TEXT,
                    result_ref TEXT,
                    user_message TEXT,
                    started_at TIMESTAMPTZ,
                    completed_at TIMESTAMPTZ,
                    UNIQUE(action_id, request_id)
                )
                """
            )
            connection.commit()
        finally:
            connection.close()

    def begin(
        self,
        action_id: str,
        request_id: str,
        *,
        now: datetime,
    ) -> dict[str, Any]:
        if action_id not in BUTTON_ACTIONS_BY_ID:
            raise ValueError("UI_ACTION_UNKNOWN")
        if not request_id.strip():
            raise ValueError("UI_ACTION_REQUEST_REQUIRED")
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("UI_ACTION_TIME_TZ_REQUIRED")
        run_id = "ui-action-" + hashlib.sha256(
            f"{action_id}|{request_id}".encode()
        ).hexdigest()[:24]
        connection = duckdb.connect(self.db_path)
        try:
            connection.execute(
                """
                INSERT INTO ui_action_audit VALUES
                (?,?,?,?,?,?,?,?)
                ON CONFLICT (action_run_id) DO NOTHING
                """,
                [run_id, action_id, request_id, "BUSY", "", "处理中", now, None],
            )
            connection.commit()
        finally:
            connection.close()
        return self.get(run_id)

    def finish(
        self,
        action_run_id: str,
        *,
        state: str,
        result_ref: str,
        user_message: str,
        now: datetime,
    ) -> dict[str, Any]:
        normalized = state.strip().upper()
        if normalized not in ACTION_TERMINAL_STATES:
            raise ValueError("UI_ACTION_STATE_INVALID")
        if not user_message.strip():
            raise ValueError("UI_ACTION_MESSAGE_REQUIRED")
        if normalized == "SUCCESS" and not result_ref.strip():
            raise ValueError("UI_ACTION_RESULT_REF_REQUIRED")
        safe_message = user_message.replace("\n", " ")[:500]
        connection = duckdb.connect(self.db_path)
        try:
            row = connection.execute(
                "SELECT state FROM ui_action_audit WHERE action_run_id=?",
                [action_run_id],
            ).fetchone()
            if row is None:
                raise KeyError(action_run_id)
            if row[0] in ACTION_TERMINAL_STATES:
                return self.get(action_run_id)
            connection.execute(
                """
                UPDATE ui_action_audit
                SET state=?, result_ref=?, user_message=?, completed_at=?
                WHERE action_run_id=?
                """,
                [normalized, result_ref, safe_message, now, action_run_id],
            )
            connection.commit()
        finally:
            connection.close()
        return self.get(action_run_id)

    def get(self, action_run_id: str) -> dict[str, Any] | None:
        connection = duckdb.connect(self.db_path, read_only=True)
        try:
            row = connection.execute(
                "SELECT * FROM ui_action_audit WHERE action_run_id=?",
                [action_run_id],
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        return {
            "action_run_id": str(row[0]),
            "action_id": str(row[1]),
            "request_id": str(row[2]),
            "state": str(row[3]),
            "result_ref": str(row[4] or ""),
            "user_message": str(row[5]),
            "started_at": row[6],
            "completed_at": row[7],
        }


def explain_order(db_path: str | Path, plan_id: str) -> dict[str, Any]:
    """Return a layered, read-only explanation with explicit missing sections."""
    normalized_plan_id = plan_id.strip()
    if not normalized_plan_id:
        return {
            "status": "ERROR",
            "plan_id": "",
            "error_code": "PLAN_ID_REQUIRED",
        }
    try:
        traces = order_traces(db_path, plan_ids={normalized_plan_id})
    except FileNotFoundError:
        return {
            "status": "EMPTY",
            "plan_id": normalized_plan_id,
            "missing": ["audit_database"],
        }
    except (ValueError, duckdb.Error) as exc:
        return {
            "status": "ERROR",
            "plan_id": normalized_plan_id,
            "error_code": type(exc).__name__,
        }
    if not traces:
        return {
            "status": "EMPTY",
            "plan_id": normalized_plan_id,
            "missing": ["plan"],
        }
    trace = traces[0]
    order = trace.get("order") or {}
    plan = trace.get("plan") or {}
    evidence = order.get("evidence_refs_json") or order.get("evidence_refs") or "[]"
    try:
        evidence_refs = json.loads(evidence) if isinstance(evidence, str) else evidence
    except json.JSONDecodeError:
        evidence_refs = []
    sections = {
        "sources_and_snapshots": evidence_refs,
        "research": {
            "decision_id": trace.get("decision_id", ""),
            "ai_run_id": order.get("ai_run_id", ""),
        },
        "plan_versions": {
            "plan_id": normalized_plan_id,
            "candidate_plan_id": order.get("candidate_plan_id", ""),
            "final_plan_id": order.get("final_plan_id", ""),
            "position_plan_id": order.get("position_plan_id", ""),
            "plan": plan,
        },
        "risk": trace.get("risk_events") or [],
        "order": order,
        "fills": trace.get("fills") or [],
    }
    missing = [name for name, value in sections.items() if not value]
    return {
        "status": "SUCCESS",
        "plan_id": normalized_plan_id,
        "sections": sections,
        "missing": missing,
    }


def render_order_explanation_html(explanation: dict[str, Any]) -> str:
    """Render one explanation without trusting database text as markup."""
    status = str(explanation.get("status", "ERROR")).upper()
    plan_id = html.escape(str(explanation.get("plan_id", "")))
    if status == "EMPTY":
        missing = ", ".join(
            html.escape(str(item)) for item in explanation.get("missing", [])
        )
        return (
            '<div class="qa-empty">暂无可解释订单'
            + (f" · 缺少 {missing}" if missing else "")
            + "</div>"
        )
    if status != "SUCCESS":
        code = html.escape(str(explanation.get("error_code", "ORDER_TRACE_ERROR")))
        return f'<div class="qa-error">订单解释失败 · {code}</div>'
    labels = {
        "sources_and_snapshots": "来源与快照",
        "research": "研究结论",
        "plan_versions": "计划版本",
        "risk": "风险裁决",
        "order": "订单意图",
        "fills": "成交事实",
    }
    parts = [
        '<div class="qa-note">',
        f"<b>Plan</b> <code>{plan_id}</code>",
    ]
    for key, label in labels.items():
        payload = explanation.get("sections", {}).get(key)
        encoded = html.escape(
            json.dumps(payload, ensure_ascii=False, default=str, indent=2)
        )
        parts.append(
            f"<details><summary>{label}</summary><pre>{encoded}</pre></details>"
        )
    if explanation.get("missing"):
        missing = ", ".join(
            html.escape(str(item)) for item in explanation["missing"]
        )
        parts.append(f"<div>缺失部分：{missing}</div>")
    parts.append("</div>")
    return "".join(parts)


def button_contract_manifest() -> list[dict[str, Any]]:
    return [asdict(item) for item in BUTTON_ACTIONS]
