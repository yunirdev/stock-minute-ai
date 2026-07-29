"""Read-only order explanations and durable NiceGUI action contracts."""

from __future__ import annotations

import hashlib
import html
import json
import queue
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

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
        ("overview.span_change", "切换权益曲线跨度", "overview", False, False),
        ("system.start", "启动引擎", "system", True, False),
        ("system.stop", "停止", "system", True, False),
        ("discord.send", "发送 Discord 简报", "discord", True, True),
        ("discord.stock_analysis", "发送个股分析", "discord", True, True),
        ("factors.run", "分析", "research", False, False),
        ("backtest.run", "运行回测", "research", False, False),
        ("pool.rebuild_all", "全部重建", "pool", False, False),
        ("pool.full_scan", "全市场扫描", "pool", False, False),
        ("pool.long_term", "更新长期池", "pool", False, False),
        ("pool.decision", "更新决策池", "pool", False, False),
        ("pool.clear_filter", "清除筛选", "pool", False, False),
        ("risk.trigger", "触发急停", "risk", True, False),
        ("risk.clear", "解除急停", "risk", True, False),
        ("maintenance.run", "运行维护分析", "maintenance", False, False),
    ]
    return tuple(ButtonActionContract(*values) for values in raw)


BUTTON_ACTIONS = _contracts()
BUTTON_ACTIONS_BY_ID = {item.action_id: item for item in BUTTON_ACTIONS}
ACTION_TERMINAL_STATES = {"SUCCESS", "EMPTY", "ERROR"}


class ButtonActionAuditStore:
    """DuckDB-backed audit trail.

    All access goes through `self._db_lock` (a reentrant lock): DuckDB file
    locking on Windows can raise IOException when two connections from
    different threads in the *same process* open the file even briefly out
    of turn, which is easy to hit once actions run on a background worker
    while the UI polls `get()`/`recent()` concurrently. Serializing access
    in-process avoids that; it does not change the on-disk isolation
    DuckDB already provides across processes.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._db_lock = threading.RLock()
        with self._db_lock:
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
        run_id = (
            "ui-action-"
            + hashlib.sha256(f"{action_id}|{request_id}".encode()).hexdigest()[:24]
        )
        with self._db_lock:
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
        with self._db_lock:
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
        with self._db_lock:
            connection = duckdb.connect(self.db_path)
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

    def recent(
        self,
        limit: int = 100,
        category: str | None = None,
        action_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Most recent action runs, newest first, for the activity feed.

        Category/action_id filters are applied against the static
        BUTTON_ACTIONS registry (not stored per-row), since categories are
        a property of the action definition, not the audit record.
        """
        candidate_ids = {
            item.action_id
            for item in BUTTON_ACTIONS
            if (category is None or item.category == category)
            and (action_id is None or item.action_id == action_id)
        }
        if not candidate_ids:
            return []
        with self._db_lock:
            connection = duckdb.connect(self.db_path, read_only=True)
            try:
                placeholders = ",".join("?" for _ in candidate_ids)
                rows = connection.execute(
                    f"""
                    SELECT action_run_id, action_id, request_id, state, result_ref,
                           user_message, started_at, completed_at
                    FROM ui_action_audit
                    WHERE action_id IN ({placeholders})
                    ORDER BY started_at DESC
                    LIMIT ?
                    """,
                    [*candidate_ids, limit],
                ).fetchall()
            except duckdb.Error:
                return []
            finally:
                connection.close()
        out = []
        for row in rows:
            contract = BUTTON_ACTIONS_BY_ID.get(row[1])
            started = row[6]
            completed = row[7]
            duration = (
                (completed - started).total_seconds()
                if started is not None and completed is not None
                else None
            )
            out.append(
                {
                    "action_run_id": str(row[0]),
                    "action_id": str(row[1]),
                    "request_id": str(row[2]),
                    "state": str(row[3]),
                    "result_ref": str(row[4] or ""),
                    "user_message": str(row[5]),
                    "started_at": started,
                    "completed_at": completed,
                    "duration_seconds": duration,
                    "label": contract.label if contract else row[1],
                    "category": contract.category if contract else "",
                }
            )
        return out

    def average_duration_seconds(
        self, action_id: str, sample: int = 20
    ) -> float | None:
        """Average wall-clock duration of the last *sample* terminal runs.

        Used only for ETA display in the task queue panel; returns None when
        there is no history yet (caller falls back to a fixed default).
        """
        with self._db_lock:
            connection = duckdb.connect(self.db_path, read_only=True)
            try:
                rows = connection.execute(
                    """
                    SELECT started_at, completed_at
                    FROM ui_action_audit
                    WHERE action_id = ? AND completed_at IS NOT NULL
                    ORDER BY started_at DESC
                    LIMIT ?
                    """,
                    [action_id, sample],
                ).fetchall()
            except duckdb.Error:
                return None
            finally:
                connection.close()
        durations = [
            (completed - started).total_seconds()
            for started, completed in rows
            if started is not None and completed is not None
        ]
        durations = [d for d in durations if d >= 0]
        if not durations:
            return None
        return sum(durations) / len(durations)


@dataclass(frozen=True)
class ActionJobResult:
    state: str = "SUCCESS"
    user_message: str = "动作已完成"
    result_ref: str = ""


class AsyncButtonActionRunner:
    """Run UI actions one at a time on a single shared worker queue.

    Multiple job-type actions triggered close together (e.g. a Discord push
    fired right after a pool rebuild) previously ran on separate threads in
    parallel. That could contend for the same DuckDB file and made "how long
    until my click actually happens" impossible to answer. Now there is one
    background worker draining a FIFO queue, so tasks execute strictly in
    arrival order and a queue position + ETA can be computed for each.
    """

    def __init__(self, store: ButtonActionAuditStore) -> None:
        self._store = store
        self._lock = threading.Lock()
        self._active: dict[str, str] = {}  # action_id -> run_id (queued or running)
        self._order: list[tuple[str, str, datetime]] = []  # (run_id, action_id, enqueued_at)
        self._current: tuple[str, str, datetime] | None = None  # (run_id, action_id, started_at)
        self._queue: "queue.Queue[tuple[str, str, Callable[[], Any]]]" = queue.Queue()
        self._worker = threading.Thread(target=self._drain, daemon=True)
        self._worker.start()

    def start(
        self,
        action_id: str,
        work: Callable[[], Any],
    ) -> dict[str, Any]:
        with self._lock:
            active_run_id = self._active.get(action_id)
            if active_run_id:
                return self._store.get(active_run_id)
            request_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc)
            started = self._store.begin(action_id, request_id, now=now)
            run_id = started["action_run_id"]
            self._active[action_id] = run_id
            self._order.append((run_id, action_id, now))
        self._queue.put((run_id, action_id, work))
        return started

    def _drain(self) -> None:
        while True:
            run_id, action_id, work = self._queue.get()
            with self._lock:
                self._order = [
                    item for item in self._order if item[0] != run_id
                ]
                self._current = (run_id, action_id, datetime.now(timezone.utc))
            try:
                result = self._normalize_result(work())
                result_ref = result.result_ref
                if result.state == "SUCCESS" and not result_ref:
                    result_ref = f"ui-job:{action_id}:{run_id}"
                self._store.finish(
                    run_id,
                    state=result.state,
                    result_ref=result_ref,
                    user_message=result.user_message,
                    now=datetime.now(timezone.utc),
                )
            except Exception as exc:
                self._store.finish(
                    run_id,
                    state="ERROR",
                    result_ref="",
                    user_message=f"操作失败: {type(exc).__name__}",
                    now=datetime.now(timezone.utc),
                )
            finally:
                with self._lock:
                    if self._active.get(action_id) == run_id:
                        self._active.pop(action_id, None)
                    self._current = None
                self._queue.task_done()

    def queue_snapshot(self) -> dict[str, Any]:
        """In-memory view of what's running/queued right now, for the UI.

        Not persisted — this is ephemeral scheduling state, distinct from
        the durable BUSY/terminal audit trail in ButtonActionAuditStore.
        """
        with self._lock:
            current = self._current
            queued = list(self._order)
        return {
            "current": (
                {"run_id": current[0], "action_id": current[1], "started_at": current[2]}
                if current
                else None
            ),
            "queued": [
                {"run_id": run_id, "action_id": action_id, "enqueued_at": enqueued_at}
                for run_id, action_id, enqueued_at in queued
            ],
        }

    @staticmethod
    def _normalize_result(result: Any) -> ActionJobResult:
        if isinstance(result, ActionJobResult):
            return result
        if result is False:
            return ActionJobResult(state="EMPTY", user_message="没有可用结果")
        if isinstance(result, (list, dict, tuple, set)) and not result:
            return ActionJobResult(state="EMPTY", user_message="没有可用结果")
        return ActionJobResult()


def _position_lifecycle_sections(
    db_path: str | Path, position_plan_id: str
) -> dict[str, Any]:
    """Read-only PositionPlan/InvalidationEvent/PositionAdjustment lookup.

    Missing tables/db degrade to empty lists rather than raising, matching
    the rest of explain_order's fail-soft behavior.
    """
    empty = {"position_plan": {}, "invalidation_events": [], "adjustments": []}
    if not position_plan_id:
        return empty
    try:
        connection = duckdb.connect(str(db_path), read_only=True)
    except duckdb.Error:
        return empty
    try:
        position_plan: dict[str, Any] = {}
        try:
            row = connection.execute(
                """
                SELECT h.position_plan_id, h.symbol, h.side, h.status,
                       h.current_version, v.open_quantity, v.average_entry_price,
                       v.stop_loss, v.take_profit, v.change_reason, v.created_at
                FROM position_plan_heads h
                LEFT JOIN position_plan_versions v
                  ON v.position_plan_id = h.position_plan_id
                 AND v.version_id = h.current_version_id
                WHERE h.position_plan_id = ?
                """,
                [position_plan_id],
            ).fetchdf()
            if not row.empty:
                position_plan = row.iloc[0].to_dict()
        except duckdb.Error:
            position_plan = {}

        invalidation_events: list[dict[str, Any]] = []
        try:
            df = connection.execute(
                """
                SELECT event_id, event_type, source, as_of, facts_json
                FROM invalidation_events
                WHERE position_plan_id = ?
                ORDER BY as_of DESC
                """,
                [position_plan_id],
            ).fetchdf()
            invalidation_events = df.to_dict("records")
        except duckdb.Error:
            invalidation_events = []

        adjustments: list[dict[str, Any]] = []
        try:
            df = connection.execute(
                """
                SELECT adjustment_id, action, status, quantity,
                       previous_stop_loss, new_stop_loss, order_plan_id, created_at
                FROM position_adjustments
                WHERE position_plan_id = ?
                ORDER BY created_at DESC
                """,
                [position_plan_id],
            ).fetchdf()
            adjustments = df.to_dict("records")
        except duckdb.Error:
            adjustments = []
    finally:
        connection.close()
    return {
        "position_plan": position_plan,
        "invalidation_events": invalidation_events,
        "adjustments": adjustments,
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
    position_plan_id = order.get("position_plan_id", "")
    lifecycle = _position_lifecycle_sections(db_path, position_plan_id)
    sections["position_plan"] = lifecycle["position_plan"]
    sections["invalidation_events"] = lifecycle["invalidation_events"]
    sections["adjustments"] = lifecycle["adjustments"]
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
        "position_plan": "持仓计划版本",
        "invalidation_events": "失效事件",
        "adjustments": "自动调整",
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
        missing = ", ".join(html.escape(str(item)) for item in explanation["missing"])
        parts.append(f"<div>缺失部分：{missing}</div>")
    parts.append("</div>")
    return "".join(parts)


def button_contract_manifest() -> list[dict[str, Any]]:
    return [asdict(item) for item in BUTTON_ACTIONS]


_DEFAULT_JOB_DURATION_SECONDS = 15.0


def queue_view(
    runner: "AsyncButtonActionRunner",
    store: ButtonActionAuditStore,
    default_duration: float = _DEFAULT_JOB_DURATION_SECONDS,
) -> dict[str, Any]:
    """Enrich the runner's in-memory queue snapshot with labels and ETAs.

    Each item's `eta_seconds` is the estimated time until that specific task
    finishes (cumulative over everything ahead of it); `total_seconds` is the
    ETA of the last item, i.e. time until the whole queue drains. Estimates
    fall back to `default_duration` for actions with no completed history yet.
    """
    snapshot = runner.queue_snapshot()
    now = datetime.now(timezone.utc)
    items: list[dict[str, Any]] = []
    cumulative = 0.0

    current = snapshot["current"]
    if current is not None:
        avg = store.average_duration_seconds(current["action_id"]) or default_duration
        elapsed = max((now - current["started_at"]).total_seconds(), 0.0)
        remaining = max(avg - elapsed, 1.0)
        contract = BUTTON_ACTIONS_BY_ID.get(current["action_id"])
        items.append(
            {
                "run_id": current["run_id"],
                "action_id": current["action_id"],
                "label": contract.label if contract else current["action_id"],
                "category": contract.category if contract else "",
                "status": "RUNNING",
                "elapsed_seconds": elapsed,
                "eta_seconds": remaining,
            }
        )
        cumulative = remaining

    for queued in snapshot["queued"]:
        avg = store.average_duration_seconds(queued["action_id"]) or default_duration
        cumulative += avg
        contract = BUTTON_ACTIONS_BY_ID.get(queued["action_id"])
        items.append(
            {
                "run_id": queued["run_id"],
                "action_id": queued["action_id"],
                "label": contract.label if contract else queued["action_id"],
                "category": contract.category if contract else "",
                "status": "QUEUED",
                "elapsed_seconds": 0.0,
                "eta_seconds": cumulative,
            }
        )

    return {"items": items, "total_seconds": cumulative}
