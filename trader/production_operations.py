"""Scheduled, non-destructive backups for the production Paper runtime."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import duckdb

from .operational_recovery import RecoveryManager

_EASTERN = ZoneInfo("America/New_York")


class ProductionOperationsCoordinator:
    """Create one verified backup after each completed US trading session."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        ai_db_path: str | Path | None = None,
        backup_root: str | Path | None = None,
    ) -> None:
        self.db_path = Path(db_path).resolve()
        self.ai_db_path = Path(ai_db_path).resolve() if ai_db_path else None
        root = Path(__file__).resolve().parents[1]
        self.backup_root = Path(backup_root or root / "backups").resolve()
        self._recovery = RecoveryManager()
        connection = duckdb.connect(str(self.db_path))
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS operational_backup_runs (
                    session_date DATE PRIMARY KEY,
                    status TEXT,
                    manifest_id TEXT,
                    manifest_path TEXT,
                    source_count INTEGER,
                    attempt_count INTEGER,
                    error_code TEXT,
                    completed_at TIMESTAMPTZ
                )
                """
            )
            connection.commit()
        finally:
            connection.close()

    def tick(self, *, now: datetime) -> dict[str, Any]:
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("PRODUCTION_OPERATIONS_TIME_TZ_REQUIRED")
        eastern = now.astimezone(_EASTERN)
        session_date = eastern.date()
        if eastern.hour < 20:
            return {"status": "NOT_DUE", "session_date": session_date.isoformat()}
        existing = self._get(session_date.isoformat())
        if existing and existing["status"] == "SUCCESS":
            return existing

        attempt = int(existing["attempt_count"]) + 1 if existing else 1
        sources = [self.db_path]
        if self.ai_db_path and self.ai_db_path != self.db_path:
            sources.append(self.ai_db_path)
        sources = [source for source in sources if source.is_file()]
        if not sources:
            return self._record(
                session_date=session_date.isoformat(),
                status="FAILED",
                manifest_id="",
                manifest_path="",
                source_count=0,
                attempt_count=attempt,
                error_code="BACKUP_SOURCE_MISSING",
                completed_at=now,
            )

        destination = (
            self.backup_root
            / session_date.isoformat()
            / f"attempt-{attempt:02d}"
        )
        try:
            for source in sources:
                connection = duckdb.connect(str(source))
                try:
                    connection.execute("CHECKPOINT")
                finally:
                    connection.close()
            manifest = self._recovery.create_backup(
                sources,
                destination=destination,
                created_at=now,
            )
        except Exception as exc:
            return self._record(
                session_date=session_date.isoformat(),
                status="FAILED",
                manifest_id="",
                manifest_path="",
                source_count=len(sources),
                attempt_count=attempt,
                error_code=type(exc).__name__,
                completed_at=now,
            )
        return self._record(
            session_date=session_date.isoformat(),
            status="SUCCESS",
            manifest_id=str(manifest["manifest_id"]),
            manifest_path=str(manifest["manifest_path"]),
            source_count=len(sources),
            attempt_count=attempt,
            error_code="",
            completed_at=now,
        )

    def _record(
        self,
        *,
        session_date: str,
        status: str,
        manifest_id: str,
        manifest_path: str,
        source_count: int,
        attempt_count: int,
        error_code: str,
        completed_at: datetime,
    ) -> dict[str, Any]:
        connection = duckdb.connect(str(self.db_path))
        try:
            connection.execute(
                """
                INSERT OR REPLACE INTO operational_backup_runs
                VALUES (?,?,?,?,?,?,?,?)
                """,
                [
                    session_date,
                    status,
                    manifest_id,
                    manifest_path,
                    source_count,
                    attempt_count,
                    error_code,
                    completed_at,
                ],
            )
            connection.commit()
        finally:
            connection.close()
        return self._get(session_date) or {}

    def _get(self, session_date: str) -> dict[str, Any] | None:
        connection = duckdb.connect(str(self.db_path), read_only=True)
        try:
            row = connection.execute(
                """
                SELECT session_date, status, manifest_id, manifest_path,
                       source_count, attempt_count, error_code, completed_at
                FROM operational_backup_runs
                WHERE session_date=?
                """,
                [session_date],
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        return {
            "session_date": row[0].isoformat(),
            "status": str(row[1]),
            "manifest_id": str(row[2] or ""),
            "manifest_path": str(row[3] or ""),
            "source_count": int(row[4]),
            "attempt_count": int(row[5]),
            "error_code": str(row[6] or ""),
            "completed_at": row[7],
        }
