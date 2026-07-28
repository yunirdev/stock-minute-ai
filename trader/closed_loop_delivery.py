"""Frozen complete-loop evidence and Paper delivery sign-off."""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb

REQUIRED_STAGE_REFS = {
    "research_snapshot_id",
    "research_run_id",
    "candidate_plan_id",
    "final_plan_id",
    "risk_event_id",
    "order_intent_id",
    "fill_id",
    "position_plan_id",
    "episode_snapshot_id",
    "review_id",
    "strategy_candidate_id",
}
REQUIRED_SCENARIOS = {
    "buy_partial": "FILLED",
    "sell": "FILLED",
    "risk_rejected": "REJECTED_WITHOUT_ORDER",
    "unknown_before_restart": "UNKNOWN",
    "unknown_after_recovery": "CANCELED",
    "restart_resubmissions": 0,
}


class ClosedLoopDeliveryStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        connection = duckdb.connect(self.db_path)
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS closed_loop_delivery_evidence (
                    evidence_id TEXT PRIMARY KEY,
                    evidence_type TEXT,
                    stage_refs_json TEXT,
                    scenarios_json TEXT,
                    metrics_json TEXT,
                    recovery_json TEXT,
                    button_report_json TEXT,
                    limitations_json TEXT,
                    accepted BOOLEAN,
                    created_at TIMESTAMPTZ
                )
                """
            )
            connection.commit()
        finally:
            connection.close()

    def record(
        self,
        *,
        evidence_type: str,
        stage_refs: dict[str, str],
        scenarios: dict[str, Any],
        metrics: dict[str, float],
        recovery: dict[str, bool],
        button_report: dict[str, Any],
        limitations: list[str],
        created_at: datetime,
    ) -> dict[str, Any]:
        if created_at.tzinfo is None or created_at.utcoffset() is None:
            raise ValueError("CLOSED_LOOP_TIME_TZ_REQUIRED")
        evidence = evidence_type.strip().upper()
        if evidence not in {"ISOLATED_PAPER", "REAL_PAPER"}:
            raise ValueError("CLOSED_LOOP_EVIDENCE_TYPE_INVALID")
        missing = REQUIRED_STAGE_REFS - set(stage_refs)
        if missing or any(not str(stage_refs[key]).strip() for key in REQUIRED_STAGE_REFS):
            raise ValueError("CLOSED_LOOP_STAGE_REFERENCE_MISSING")
        if any(scenarios.get(key) != value for key, value in REQUIRED_SCENARIOS.items()):
            raise ValueError("CLOSED_LOOP_SCENARIO_INCOMPLETE")
        required_metrics = {
            "data_coverage",
            "research_success_rate",
            "plan_count",
            "order_success_rate",
            "fill_rate",
            "slippage",
            "max_drawdown",
            "realized_pnl",
        }
        if required_metrics - set(metrics):
            raise ValueError("CLOSED_LOOP_METRICS_MISSING")
        if not all(math.isfinite(float(value)) for value in metrics.values()):
            raise ValueError("CLOSED_LOOP_METRICS_NON_FINITE")
        required_recovery = {"API", "DATABASE", "RUNTIME"}
        recovery_ok = all(recovery.get(key) is True for key in required_recovery)
        buttons_ok = (
            button_report.get("action_count") == 31
            and button_report.get("success_covered") == 31
            and button_report.get("empty_covered") == 31
            and button_report.get("error_covered") == 31
            and button_report.get("busy_covered") == 31
        )
        accepted = recovery_ok and buttons_ok and bool(limitations)
        payload = {
            "evidence_type": evidence,
            "stage_refs": stage_refs,
            "scenarios": scenarios,
            "metrics": metrics,
            "recovery": recovery,
            "button_report": button_report,
            "limitations": limitations,
            "accepted": accepted,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        evidence_id = "closed-loop-" + hashlib.sha256(
            canonical.encode()
        ).hexdigest()[:24]
        connection = duckdb.connect(self.db_path)
        try:
            connection.execute(
                """
                INSERT INTO closed_loop_delivery_evidence VALUES
                (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT (evidence_id) DO NOTHING
                """,
                [
                    evidence_id,
                    evidence,
                    json.dumps(stage_refs, sort_keys=True),
                    json.dumps(scenarios, sort_keys=True),
                    json.dumps(metrics, sort_keys=True),
                    json.dumps(recovery, sort_keys=True),
                    json.dumps(button_report, sort_keys=True),
                    json.dumps(limitations),
                    accepted,
                    created_at,
                ],
            )
            connection.commit()
        finally:
            connection.close()
        return self.get(evidence_id)

    def get(self, evidence_id: str) -> dict[str, Any] | None:
        connection = duckdb.connect(self.db_path, read_only=True)
        try:
            row = connection.execute(
                """
                SELECT * FROM closed_loop_delivery_evidence
                WHERE evidence_id=?
                """,
                [evidence_id],
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        return {
            "evidence_id": str(row[0]),
            "evidence_type": str(row[1]),
            "stage_refs": json.loads(row[2]),
            "scenarios": json.loads(row[3]),
            "metrics": json.loads(row[4]),
            "recovery": json.loads(row[5]),
            "button_report": json.loads(row[6]),
            "limitations": json.loads(row[7]),
            "accepted": bool(row[8]),
            "created_at": row[9],
        }


def render_call_graph(stage_refs: dict[str, str]) -> str:
    return " -> ".join(
        (
            stage_refs["research_snapshot_id"],
            stage_refs["research_run_id"],
            stage_refs["candidate_plan_id"],
            stage_refs["final_plan_id"],
            stage_refs["risk_event_id"],
            stage_refs["order_intent_id"],
            stage_refs["fill_id"],
            stage_refs["position_plan_id"],
            stage_refs["episode_snapshot_id"],
            stage_refs["review_id"],
            stage_refs["strategy_candidate_id"],
        )
    )
