"""Immutable scheduled-session evidence for long-running Paper maturity."""
from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import duckdb

_EVIDENCE_TYPES = {"REAL", "SYNTHETIC"}


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


class PaperMaturityStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        connection = duckdb.connect(self.db_path)
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS paper_maturity_schedule (
                    evidence_type TEXT,
                    session_date DATE,
                    schedule_version TEXT,
                    registered_at TIMESTAMPTZ,
                    PRIMARY KEY(evidence_type, session_date)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS paper_maturity_observations (
                    observation_id TEXT PRIMARY KEY,
                    evidence_type TEXT,
                    session_date DATE,
                    reports_complete BOOLEAN,
                    unexplained_duplicate_orders INTEGER,
                    plan_rewrites INTEGER,
                    state_differences INTEGER,
                    unresolved_failures INTEGER,
                    evidence_refs_json TEXT,
                    content_hash TEXT,
                    observed_at TIMESTAMPTZ,
                    UNIQUE(evidence_type, session_date)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS paper_maturity_reports (
                    report_id TEXT PRIMARY KEY,
                    evidence_type TEXT,
                    through_date DATE,
                    required_sessions INTEGER,
                    report_json TEXT,
                    created_at TIMESTAMPTZ
                )
                """
            )
            connection.commit()
        finally:
            connection.close()

    def register_session(
        self,
        *,
        evidence_type: str,
        session_date: date,
        schedule_version: str,
        registered_at: datetime,
    ) -> None:
        evidence = self._evidence_type(evidence_type)
        version = schedule_version.strip()
        self._aware(registered_at, "PAPER_MATURITY_REGISTER_TIME_TZ_REQUIRED")
        if not version:
            raise ValueError("PAPER_MATURITY_SCHEDULE_VERSION_REQUIRED")
        connection = duckdb.connect(self.db_path)
        try:
            existing = connection.execute(
                """
                SELECT schedule_version FROM paper_maturity_schedule
                WHERE evidence_type=? AND session_date=?
                """,
                [evidence, session_date],
            ).fetchone()
            if existing is not None and str(existing[0]) != version:
                raise ValueError("PAPER_MATURITY_SCHEDULE_REWRITE")
            connection.execute(
                """
                INSERT INTO paper_maturity_schedule VALUES (?,?,?,?)
                ON CONFLICT(evidence_type, session_date) DO NOTHING
                """,
                [evidence, session_date, version, registered_at],
            )
            connection.commit()
        finally:
            connection.close()

    def record_observation(
        self,
        *,
        evidence_type: str,
        session_date: date,
        reports_complete: bool,
        unexplained_duplicate_orders: int,
        plan_rewrites: int,
        state_differences: int,
        unresolved_failures: int,
        evidence_refs: dict[str, str],
        observed_at: datetime,
    ) -> str:
        evidence = self._evidence_type(evidence_type)
        self._aware(observed_at, "PAPER_MATURITY_OBSERVED_TIME_TZ_REQUIRED")
        counts = {
            "unexplained_duplicate_orders": unexplained_duplicate_orders,
            "plan_rewrites": plan_rewrites,
            "state_differences": state_differences,
            "unresolved_failures": unresolved_failures,
        }
        if any(not isinstance(value, int) or value < 0 for value in counts.values()):
            raise ValueError("PAPER_MATURITY_COUNT_INVALID")
        required_refs = {
            "daily_report_ref",
            "research_quality_ref",
            "position_quality_ref",
        }
        if required_refs - set(evidence_refs) or any(
            not str(evidence_refs[key]).strip() for key in required_refs
        ):
            raise ValueError("PAPER_MATURITY_EVIDENCE_REF_MISSING")
        connection = duckdb.connect(self.db_path)
        try:
            scheduled = connection.execute(
                """
                SELECT 1 FROM paper_maturity_schedule
                WHERE evidence_type=? AND session_date=?
                """,
                [evidence, session_date],
            ).fetchone()
            if scheduled is None:
                raise ValueError("PAPER_MATURITY_SESSION_NOT_SCHEDULED")
            payload = {
                "evidence_type": evidence,
                "session_date": session_date.isoformat(),
                "reports_complete": bool(reports_complete),
                **counts,
                "evidence_refs": dict(sorted(evidence_refs.items())),
            }
            content_hash = hashlib.sha256(_canonical(payload).encode()).hexdigest()
            observation_id = "paper-maturity-" + content_hash[:24]
            existing = connection.execute(
                """
                SELECT content_hash FROM paper_maturity_observations
                WHERE evidence_type=? AND session_date=?
                """,
                [evidence, session_date],
            ).fetchone()
            if existing is not None and str(existing[0]) != content_hash:
                raise ValueError("PAPER_MATURITY_OBSERVATION_REWRITE")
            connection.execute(
                """
                INSERT INTO paper_maturity_observations VALUES
                (?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(observation_id) DO NOTHING
                """,
                [
                    observation_id,
                    evidence,
                    session_date,
                    bool(reports_complete),
                    unexplained_duplicate_orders,
                    plan_rewrites,
                    state_differences,
                    unresolved_failures,
                    _canonical(evidence_refs),
                    content_hash,
                    observed_at,
                ],
            )
            connection.commit()
            return observation_id
        finally:
            connection.close()

    def build_report(
        self,
        *,
        evidence_type: str,
        through_date: date,
        required_sessions: int = 60,
        created_at: datetime,
        persist: bool = True,
    ) -> dict[str, Any]:
        evidence = self._evidence_type(evidence_type)
        self._aware(created_at, "PAPER_MATURITY_REPORT_TIME_TZ_REQUIRED")
        if required_sessions < 1:
            raise ValueError("PAPER_MATURITY_REQUIRED_SESSIONS_INVALID")
        connection = duckdb.connect(self.db_path, read_only=True)
        try:
            schedule_rows = connection.execute(
                """
                SELECT session_date FROM paper_maturity_schedule
                WHERE evidence_type=? AND session_date<=?
                ORDER BY session_date DESC LIMIT ?
                """,
                [evidence, through_date, required_sessions],
            ).fetchall()
            scheduled_dates = list(reversed([row[0] for row in schedule_rows]))
            observation_rows = connection.execute(
                """
                SELECT session_date, observation_id, reports_complete,
                       unexplained_duplicate_orders, plan_rewrites,
                       state_differences, unresolved_failures
                FROM paper_maturity_observations
                WHERE evidence_type=? AND session_date<=?
                """,
                [evidence, through_date],
            ).fetchall()
        finally:
            connection.close()
        observations = {row[0]: row for row in observation_rows}
        failures: list[dict[str, str]] = []
        observation_ids: list[str] = []
        for session in scheduled_dates:
            row = observations.get(session)
            if row is None:
                failures.append(
                    {"session_date": session.isoformat(), "reason": "MISSING_OBSERVATION"}
                )
                continue
            observation_ids.append(str(row[1]))
            reasons = []
            if not bool(row[2]):
                reasons.append("REPORTS_INCOMPLETE")
            for index, reason in (
                (3, "UNEXPLAINED_DUPLICATE_ORDER"),
                (4, "PLAN_REWRITE"),
                (5, "STATE_DIFFERENCE"),
                (6, "UNRESOLVED_FAILURE"),
            ):
                if int(row[index]) > 0:
                    reasons.append(reason)
            failures.extend(
                {"session_date": session.isoformat(), "reason": reason}
                for reason in reasons
            )
        if len(scheduled_dates) < required_sessions:
            failures.append(
                {"session_date": "", "reason": "INSUFFICIENT_SCHEDULED_SESSIONS"}
            )
        report = {
            "evidence_type": evidence,
            "through_date": through_date.isoformat(),
            "required_sessions": required_sessions,
            "scheduled_sessions": len(scheduled_dates),
            "observed_sessions": len(observation_ids),
            "failures": failures,
            "observation_ids": observation_ids,
            "ready": len(scheduled_dates) == required_sessions and not failures,
        }
        report_id = "paper-maturity-report-" + hashlib.sha256(
            _canonical(report).encode()
        ).hexdigest()[:24]
        report["report_id"] = report_id
        if persist:
            connection = duckdb.connect(self.db_path)
            try:
                connection.execute(
                    """
                    INSERT INTO paper_maturity_reports VALUES (?,?,?,?,?,?)
                    ON CONFLICT(report_id) DO NOTHING
                    """,
                    [
                        report_id,
                        evidence,
                        through_date,
                        required_sessions,
                        _canonical(report),
                        created_at,
                    ],
                )
                connection.commit()
            finally:
                connection.close()
        return report

    @staticmethod
    def _evidence_type(value: str) -> str:
        normalized = value.strip().upper()
        if normalized not in _EVIDENCE_TYPES:
            raise ValueError("PAPER_MATURITY_EVIDENCE_TYPE_INVALID")
        return normalized

    @staticmethod
    def _aware(value: datetime, code: str) -> None:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(code)
