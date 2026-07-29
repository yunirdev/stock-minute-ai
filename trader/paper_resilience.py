"""Frozen fault-drill evidence for the complete Alpaca Paper path."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb

SCENARIO_SPECS = {
    "MISSING_DATA": {"outcome": "BLOCKED", "submit_count": 0},
    "TIMEOUT": {"outcome": "BLOCKED", "submit_count": 0},
    "RESTART": {"outcome": "RECOVERED", "submit_count": 0},
    "PARTIAL_FILL": {"outcome": "RECOVERED", "submit_count": 1},
    "MARKET_CLOSED": {"outcome": "NO_ACTION", "submit_count": 0},
    "KILL_SWITCH": {"outcome": "BLOCKED", "submit_count": 0},
}
_EVIDENCE_TYPES = {"REAL", "SYNTHETIC"}


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


class PaperResilienceStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        connection = duckdb.connect(self.db_path)
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS paper_resilience_drills (
                    drill_id TEXT PRIMARY KEY,
                    drill_run_id TEXT,
                    evidence_type TEXT,
                    scenario TEXT,
                    expected_outcome TEXT,
                    actual_outcome TEXT,
                    submit_count INTEGER,
                    unexpected_submit_count INTEGER,
                    audit_ref TEXT,
                    recovery_ref TEXT,
                    error_code TEXT,
                    passed BOOLEAN,
                    content_hash TEXT,
                    created_at TIMESTAMPTZ,
                    UNIQUE(drill_run_id, evidence_type, scenario)
                )
                """
            )
            connection.commit()
        finally:
            connection.close()

    def record(
        self,
        *,
        drill_run_id: str,
        evidence_type: str,
        scenario: str,
        actual_outcome: str,
        submit_count: int,
        unexpected_submit_count: int,
        audit_ref: str,
        recovery_ref: str,
        error_code: str,
        created_at: datetime,
    ) -> dict[str, Any]:
        run_id = drill_run_id.strip()
        evidence = evidence_type.strip().upper()
        normalized_scenario = scenario.strip().upper()
        actual = actual_outcome.strip().upper()
        if not run_id:
            raise ValueError("PAPER_RESILIENCE_RUN_ID_REQUIRED")
        if evidence not in _EVIDENCE_TYPES:
            raise ValueError("PAPER_RESILIENCE_EVIDENCE_TYPE_INVALID")
        if normalized_scenario not in SCENARIO_SPECS:
            raise ValueError("PAPER_RESILIENCE_SCENARIO_INVALID")
        if created_at.tzinfo is None or created_at.utcoffset() is None:
            raise ValueError("PAPER_RESILIENCE_TIME_TZ_REQUIRED")
        if (
            not isinstance(submit_count, int)
            or not isinstance(unexpected_submit_count, int)
            or submit_count < 0
            or unexpected_submit_count < 0
        ):
            raise ValueError("PAPER_RESILIENCE_SUBMIT_COUNT_INVALID")
        if not audit_ref.strip() or not recovery_ref.strip():
            raise ValueError("PAPER_RESILIENCE_REFERENCE_REQUIRED")
        spec = SCENARIO_SPECS[normalized_scenario]
        passed = (
            actual == spec["outcome"]
            and submit_count == spec["submit_count"]
            and unexpected_submit_count == 0
        )
        payload = {
            "drill_run_id": run_id,
            "evidence_type": evidence,
            "scenario": normalized_scenario,
            "expected_outcome": spec["outcome"],
            "actual_outcome": actual,
            "submit_count": submit_count,
            "unexpected_submit_count": unexpected_submit_count,
            "audit_ref": audit_ref.strip(),
            "recovery_ref": recovery_ref.strip(),
            "error_code": error_code.strip()[:200],
            "passed": passed,
        }
        content_hash = hashlib.sha256(_canonical(payload).encode()).hexdigest()
        drill_id = "paper-resilience-" + content_hash[:24]
        connection = duckdb.connect(self.db_path)
        try:
            existing = connection.execute(
                """
                SELECT content_hash FROM paper_resilience_drills
                WHERE drill_run_id=? AND evidence_type=? AND scenario=?
                """,
                [run_id, evidence, normalized_scenario],
            ).fetchone()
            if existing is not None and str(existing[0]) != content_hash:
                raise ValueError("PAPER_RESILIENCE_DRILL_REWRITE")
            connection.execute(
                """
                INSERT INTO paper_resilience_drills VALUES
                (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(drill_id) DO NOTHING
                """,
                [
                    drill_id,
                    run_id,
                    evidence,
                    normalized_scenario,
                    spec["outcome"],
                    actual,
                    submit_count,
                    unexpected_submit_count,
                    audit_ref.strip(),
                    recovery_ref.strip(),
                    error_code.strip()[:200],
                    passed,
                    content_hash,
                    created_at,
                ],
            )
            connection.commit()
        finally:
            connection.close()
        return {**payload, "drill_id": drill_id}

    def build_report(
        self,
        *,
        drill_run_id: str,
        evidence_type: str,
    ) -> dict[str, Any]:
        run_id = drill_run_id.strip()
        evidence = evidence_type.strip().upper()
        if evidence not in _EVIDENCE_TYPES:
            raise ValueError("PAPER_RESILIENCE_EVIDENCE_TYPE_INVALID")
        connection = duckdb.connect(self.db_path, read_only=True)
        try:
            rows = connection.execute(
                """
                SELECT drill_id, scenario, expected_outcome, actual_outcome,
                       submit_count, unexpected_submit_count, audit_ref,
                       recovery_ref, error_code, passed
                FROM paper_resilience_drills
                WHERE drill_run_id=? AND evidence_type=?
                ORDER BY scenario
                """,
                [run_id, evidence],
            ).fetchall()
        finally:
            connection.close()
        by_scenario = {str(row[1]): row for row in rows}
        missing = sorted(set(SCENARIO_SPECS) - set(by_scenario))
        failed = sorted(
            scenario
            for scenario, row in by_scenario.items()
            if not bool(row[9])
        )
        unexpected_submits = sum(int(row[5]) for row in rows)
        return {
            "drill_run_id": run_id,
            "evidence_type": evidence,
            "required_scenarios": sorted(SCENARIO_SPECS),
            "observed_scenarios": sorted(by_scenario),
            "missing_scenarios": missing,
            "failed_scenarios": failed,
            "unexpected_submit_count": unexpected_submits,
            "drill_ids": [str(row[0]) for row in rows],
            "passed": not missing and not failed and unexpected_submits == 0,
        }
