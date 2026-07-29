"""Runtime-owned daily evidence capture for natural Paper maturity."""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import duckdb

from .market_calendar import market_holidays
from .paper_maturity import PaperMaturityStore

logger = logging.getLogger(__name__)
_NEW_YORK = ZoneInfo("America/New_York")


class ProductionEvidenceCoordinator:
    """Register every scheduled session and finalize immutable REAL evidence."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        schedule_version: str = "simple-nyse-v1",
        cutoff_hour_et: int = 20,
    ) -> None:
        self.db_path = str(db_path)
        self.schedule_version = schedule_version
        self.cutoff_hour_et = cutoff_hour_et
        self.store = PaperMaturityStore(db_path)

    def tick(
        self,
        *,
        now: datetime,
        research_run: Any | None,
        position_report: dict[str, Any] | None,
        reconciliation_blocked: bool,
    ) -> dict[str, Any]:
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("PRODUCTION_EVIDENCE_TIME_TZ_REQUIRED")
        et_now = now.astimezone(_NEW_YORK)
        self._ensure_schedule_through(et_now.date(), now)
        finalized = []
        for session_date in self._unobserved_sessions(et_now.date()):
            if session_date == et_now.date() and et_now.hour < self.cutoff_hour_et:
                continue
            current = session_date == et_now.date()
            research_matches = (
                current
                and research_run is not None
                and str(getattr(research_run, "trading_date", ""))
                == session_date.isoformat()
            )
            research_ok = (
                research_matches
                and str(getattr(research_run, "status", "")).upper()
                == "COMPLETED"
                and int(getattr(research_run, "failed_symbols", 0) or 0) == 0
            )
            position_ok = (
                current
                and position_report is not None
                and position_report.get("observation_kind") == "REAL"
                and int(position_report.get("position_mismatches", 0)) == 0
                and int(position_report.get("silent_rewrites", 0)) == 0
                and int(position_report.get("duplicate_adjustments", 0)) == 0
            )
            research_ref = (
                str(getattr(research_run, "run_id", ""))
                if research_matches
                else f"missing-research:{session_date.isoformat()}"
            )
            position_ref = (
                self._report_ref(position_report)
                if current and position_report is not None
                else f"missing-position:{session_date.isoformat()}"
            )
            duplicates = self._unexplained_duplicate_orders(session_date)
            plan_rewrites = int(
                (position_report or {}).get("silent_rewrites", 0)
            ) if current else 0
            state_differences = (
                int((position_report or {}).get("position_mismatches", 0))
                + int(bool(reconciliation_blocked))
                if current
                else 1
            )
            unresolved = (
                int(not research_ok)
                + int(not position_ok)
                + int(bool(reconciliation_blocked))
            )
            observation_id = self.store.record_observation(
                evidence_type="REAL",
                session_date=session_date,
                reports_complete=research_ok and position_ok,
                unexplained_duplicate_orders=duplicates,
                plan_rewrites=plan_rewrites,
                state_differences=state_differences,
                unresolved_failures=unresolved,
                evidence_refs={
                    "daily_report_ref": f"runtime-day:{session_date.isoformat()}",
                    "research_quality_ref": research_ref,
                    "position_quality_ref": position_ref,
                },
                observed_at=now,
            )
            finalized.append(observation_id)
        report = self.store.build_report(
            evidence_type="REAL",
            through_date=et_now.date(),
            required_sessions=60,
            created_at=now,
        )
        return {"finalized_observation_ids": finalized, "report": report}

    def _ensure_schedule_through(self, through_date: date, now: datetime) -> None:
        connection = duckdb.connect(self.db_path, read_only=True)
        try:
            row = connection.execute(
                """
                SELECT max(session_date) FROM paper_maturity_schedule
                WHERE evidence_type='REAL'
                """
            ).fetchone()
        finally:
            connection.close()
        start = (row[0] + timedelta(days=1)) if row and row[0] else through_date
        current = start
        while current <= through_date:
            if self._is_scheduled_session(current):
                self.store.register_session(
                    evidence_type="REAL",
                    session_date=current,
                    schedule_version=self.schedule_version,
                    registered_at=now,
                )
            current += timedelta(days=1)

    def _unobserved_sessions(self, through_date: date) -> list[date]:
        connection = duckdb.connect(self.db_path, read_only=True)
        try:
            rows = connection.execute(
                """
                SELECT s.session_date
                FROM paper_maturity_schedule AS s
                LEFT JOIN paper_maturity_observations AS o
                  ON o.evidence_type=s.evidence_type
                 AND o.session_date=s.session_date
                WHERE s.evidence_type='REAL'
                  AND s.session_date<=?
                  AND o.observation_id IS NULL
                ORDER BY s.session_date
                """,
                [through_date],
            ).fetchall()
        finally:
            connection.close()
        return [row[0] for row in rows]

    def _unexplained_duplicate_orders(self, session_date: date) -> int:
        connection = duckdb.connect(self.db_path, read_only=True)
        try:
            table = connection.execute(
                """
                SELECT 1 FROM information_schema.tables
                WHERE table_name='order_intents'
                """
            ).fetchone()
            if table is None:
                return 0
            return int(
                connection.execute(
                    """
                    SELECT coalesce(sum(duplicate_count - 1), 0)
                    FROM (
                        SELECT count(*) AS duplicate_count
                        FROM order_intents
                        WHERE CAST(updated_at AS DATE)=?
                        GROUP BY idempotency_key
                        HAVING count(*) > 1
                    )
                    """,
                    [session_date],
                ).fetchone()[0]
                or 0
            )
        finally:
            connection.close()

    @staticmethod
    def _is_scheduled_session(value: date) -> bool:
        return value.weekday() < 5 and value not in market_holidays(value.year)

    @staticmethod
    def _report_ref(report: dict[str, Any]) -> str:
        canonical = json.dumps(
            report,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return "position-report-" + hashlib.sha256(
            canonical.encode()
        ).hexdigest()[:24]
