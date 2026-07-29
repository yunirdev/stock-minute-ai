"""Local, dependency-free backend bug collection on top of DuckDB."""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

import duckdb

_SECRET = re.compile(
    r"(?i)(api[_-]?key|secret|token|password|authorization)\s*[:=]\s*[^,\s]+"
)
_SECRET_KEYS = {
    "api_key",
    "secret_key",
    "api_secret",
    "password",
    "token",
    "authorization",
}
_WRITE_LOCK = threading.RLock()


def _redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _redact(item)
            for key, item in value.items()
            if str(key).lower() not in _SECRET_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    return _SECRET.sub(r"\1=<redacted>", str(value)) if value is not None else None


@dataclass(frozen=True)
class BugIssue:
    fingerprint: str
    error_type: str
    message: str
    component: str
    severity: str
    status: str
    occurrence_count: int
    first_seen: datetime
    last_seen: datetime


class BugReporter:
    """Append sanitized occurrences and maintain a deduplicated issue index."""

    def __init__(self, db_path: str, component: str = "trader") -> None:
        self.db_path = db_path
        self.component = component
        self._init_db()

    def _connect(self):
        return duckdb.connect(self.db_path)

    @contextmanager
    def _connection(self):
        con = self._connect()
        try:
            yield con
        finally:
            con.close()

    def _write(self, operation: Callable[[Any], Any]) -> Any:
        """Serialize local writers and retry short cross-process DuckDB conflicts."""
        last_error: duckdb.Error | None = None
        for attempt in range(5):
            try:
                with _WRITE_LOCK:
                    con = self._connect()
                    try:
                        con.execute("BEGIN TRANSACTION")
                        result = operation(con)
                        con.execute("COMMIT")
                        return result
                    except Exception:
                        try:
                            con.execute("ROLLBACK")
                        except duckdb.Error:
                            pass
                        raise
                    finally:
                        con.close()
            except duckdb.Error as exc:
                last_error = exc
                if attempt < 4:
                    time.sleep(0.05 * (attempt + 1))
        assert last_error is not None
        raise last_error

    def _init_db(self) -> None:
        def create_tables(con) -> None:
            con.execute("""
                CREATE TABLE IF NOT EXISTS bug_issues (
                    fingerprint TEXT PRIMARY KEY, error_type TEXT, message TEXT,
                    component TEXT, severity TEXT, status TEXT DEFAULT 'OPEN',
                    occurrence_count INTEGER DEFAULT 0, first_seen TIMESTAMPTZ,
                    last_seen TIMESTAMPTZ
                )
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS bug_events (
                    event_id TEXT PRIMARY KEY, fingerprint TEXT, ts TIMESTAMPTZ,
                    error_type TEXT, message TEXT, traceback TEXT, component TEXT,
                    operation TEXT, severity TEXT, context_json TEXT, run_id TEXT,
                    symbol TEXT, plan_id TEXT, intent_id TEXT
                )
            """)

        self._write(create_tables)

    def capture_exception(
        self,
        exc: BaseException,
        *,
        operation: str = "",
        context: Mapping[str, Any] | None = None,
        severity: str = "ERROR",
        run_id: str = "",
        symbol: str = "",
        plan_id: str = "",
        intent_id: str = "",
    ) -> str:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        return self.capture_message(
            f"{type(exc).__name__}: {exc}",
            traceback_text=tb,
            operation=operation,
            context=context,
            severity=severity,
            run_id=run_id,
            symbol=symbol,
            plan_id=plan_id,
            intent_id=intent_id,
            error_type=type(exc).__name__,
        )

    def capture_message(
        self,
        message: str,
        *,
        traceback_text: str = "",
        operation: str = "",
        context: Mapping[str, Any] | None = None,
        severity: str = "ERROR",
        run_id: str = "",
        symbol: str = "",
        plan_id: str = "",
        intent_id: str = "",
        error_type: str = "Error",
    ) -> str:
        safe_message = str(_redact(message))[:2000]
        safe_traceback = str(_redact(traceback_text))[:12000]
        fingerprint = hashlib.sha256(
            f"{self.component}|{error_type}|{safe_message}".encode()
        ).hexdigest()[:32]
        now = datetime.now(timezone.utc)
        event_id = hashlib.sha256(
            f"{fingerprint}|{now.isoformat()}".encode()
        ).hexdigest()[:32]

        def store(con) -> None:
            con.execute(
                "INSERT INTO bug_events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [
                    event_id,
                    fingerprint,
                    now,
                    error_type,
                    safe_message,
                    safe_traceback,
                    self.component,
                    operation,
                    severity,
                    json.dumps(_redact(context or {}), default=str),
                    run_id,
                    symbol,
                    plan_id,
                    intent_id,
                ],
            )
            con.execute(
                """
                INSERT INTO bug_issues VALUES (?,?,?,?,?,?,1,?,?)
                ON CONFLICT (fingerprint) DO UPDATE SET
                    last_seen=excluded.last_seen,
                    occurrence_count=bug_issues.occurrence_count+1,
                    error_type=excluded.error_type,
                    message=excluded.message,
                    severity=excluded.severity,
                    status='OPEN'
                """,
                [
                    fingerprint,
                    error_type,
                    safe_message,
                    self.component,
                    severity,
                    "OPEN",
                    now,
                    now,
                ],
            )

        self._write(store)
        return fingerprint

    def unresolved(self, limit: int = 100) -> list[BugIssue]:
        with self._connection() as con:
            rows = con.execute(
                "SELECT fingerprint,error_type,message,component,severity,status,"
                "occurrence_count,first_seen,last_seen FROM bug_issues "
                "WHERE status='OPEN' ORDER BY last_seen DESC LIMIT ?",
                [limit],
            ).fetchall()
        return [BugIssue(*row) for row in rows]

    def recent_events(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connection() as con:
            rows = con.execute(
                "SELECT * FROM bug_events ORDER BY ts DESC LIMIT ?", [limit]
            ).fetchall()
            columns = [row[0] for row in con.execute("DESCRIBE bug_events").fetchall()]
        return [dict(zip(columns, row)) for row in rows]

    def resolve(self, fingerprint: str) -> bool:
        return self._set_status(fingerprint, "RESOLVED")

    def ignore(self, fingerprint: str) -> bool:
        return self._set_status(fingerprint, "IGNORED")

    def reopen(self, fingerprint: str) -> bool:
        return self._set_status(fingerprint, "OPEN")

    def _set_status(self, fingerprint: str, status: str) -> bool:
        def update(con) -> bool:
            exists = con.execute(
                "SELECT 1 FROM bug_issues WHERE fingerprint=?", [fingerprint]
            ).fetchone()
            if not exists:
                return False
            con.execute(
                "UPDATE bug_issues SET status=? WHERE fingerprint=?",
                [status, fingerprint],
            )
            return True

        return bool(self._write(update))
