"""Durable research quota, batching, retry, timeout, and resume control."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import duckdb

from .focus_pool import FocusPoolStore

_ACTIONABLE = {"PENDING", "RETRY", "RUNNING"}


def _aware(value: datetime, code: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(code)


@dataclass(frozen=True)
class ResearchEstimate:
    symbol: str
    estimated_cost: float
    estimated_seconds: float

    def normalized(self) -> ResearchEstimate:
        symbol = self.symbol.strip().upper()
        if not symbol:
            raise ValueError("RESEARCH_ESTIMATE_SYMBOL_REQUIRED")
        values = (self.estimated_cost, self.estimated_seconds)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("RESEARCH_ESTIMATE_NON_FINITE")
        if self.estimated_cost < 0 or self.estimated_seconds <= 0:
            raise ValueError("RESEARCH_ESTIMATE_INVALID")
        return ResearchEstimate(
            symbol=symbol,
            estimated_cost=float(self.estimated_cost),
            estimated_seconds=float(self.estimated_seconds),
        )


@dataclass(frozen=True)
class ResearchBudgetPolicy:
    max_symbols: int
    max_estimated_cost: float
    max_runtime_seconds: float
    batch_size: int
    max_retries: int
    attempt_timeout_seconds: float

    def __post_init__(self) -> None:
        if (
            self.max_symbols < 1
            or self.batch_size < 1
            or self.max_retries < 0
        ):
            raise ValueError("RESEARCH_BUDGET_COUNT_INVALID")
        numeric = (
            self.max_estimated_cost,
            self.max_runtime_seconds,
            self.attempt_timeout_seconds,
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("RESEARCH_BUDGET_NON_FINITE")
        if min(numeric) <= 0:
            raise ValueError("RESEARCH_BUDGET_LIMIT_INVALID")


class ResearchBudgetStore:
    """A durable queue planned only from an immutable focus pool."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self.pools = FocusPoolStore(db_path)
        self._migrate()

    def _connect(self, *, read_only: bool = False):
        return duckdb.connect(self.db_path, read_only=read_only)

    def _migrate(self) -> None:
        connection = self._connect()
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS research_budget_runs (
                    budget_run_id TEXT PRIMARY KEY,
                    pool_id TEXT,
                    trading_date TEXT,
                    status TEXT,
                    policy_json TEXT,
                    planned_count INTEGER,
                    deferred_count INTEGER,
                    estimated_cost DOUBLE,
                    estimated_seconds DOUBLE,
                    started_at TIMESTAMPTZ,
                    updated_at TIMESTAMPTZ
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS research_budget_items (
                    work_id TEXT PRIMARY KEY,
                    budget_run_id TEXT,
                    symbol TEXT,
                    priority INTEGER,
                    estimated_cost DOUBLE,
                    estimated_seconds DOUBLE,
                    status TEXT,
                    attempts INTEGER,
                    last_error TEXT,
                    attempt_started_at TIMESTAMPTZ,
                    completed_at TIMESTAMPTZ,
                    actual_cost DOUBLE,
                    actual_seconds DOUBLE,
                    deferral_reason TEXT,
                    UNIQUE (budget_run_id, symbol)
                )
                """
            )
            connection.commit()
        finally:
            connection.close()

    def plan(
        self,
        *,
        pool_id: str,
        trading_date: str,
        estimates: Iterable[ResearchEstimate],
        policy: ResearchBudgetPolicy,
        started_at: datetime,
    ) -> dict[str, Any]:
        _aware(started_at, "RESEARCH_BUDGET_START_TZ_REQUIRED")
        date_value = trading_date.strip()
        if not date_value:
            raise ValueError("RESEARCH_BUDGET_DATE_REQUIRED")
        pool = self.pools.get(pool_id)
        if pool is None:
            raise ValueError("RESEARCH_BUDGET_POOL_NOT_FOUND")
        included = sorted(
            (
                decision
                for decision in pool["decisions"]
                if decision["included"]
            ),
            key=lambda decision: (decision["rank"], decision["symbol"]),
        )
        by_symbol: dict[str, ResearchEstimate] = {}
        for raw in estimates:
            if not isinstance(raw, ResearchEstimate):
                raise ValueError("RESEARCH_ESTIMATE_INVALID")
            estimate = raw.normalized()
            existing = by_symbol.get(estimate.symbol)
            if existing is not None and existing != estimate:
                raise ValueError("RESEARCH_ESTIMATE_DUPLICATE_CONFLICT")
            by_symbol[estimate.symbol] = estimate
        included_symbols = {item["symbol"] for item in included}
        if not included_symbols.issubset(by_symbol):
            raise ValueError("RESEARCH_ESTIMATE_MISSING")
        if set(by_symbol) - included_symbols:
            raise ValueError("RESEARCH_ESTIMATE_OUTSIDE_FOCUS_POOL")

        selected_count = 0
        selected_cost = 0.0
        selected_seconds = 0.0
        planned: list[dict[str, Any]] = []
        for decision in included:
            estimate = by_symbol[decision["symbol"]]
            deferral = ""
            if selected_count >= policy.max_symbols:
                deferral = "SYMBOL_QUOTA"
            elif (
                selected_cost + estimate.estimated_cost
                > policy.max_estimated_cost
            ):
                deferral = "COST_QUOTA"
            elif (
                selected_seconds + estimate.estimated_seconds
                > policy.max_runtime_seconds
            ):
                deferral = "TIME_QUOTA"
            else:
                selected_count += 1
                selected_cost += estimate.estimated_cost
                selected_seconds += estimate.estimated_seconds
            planned.append(
                {
                    "symbol": estimate.symbol,
                    "priority": int(decision["rank"]),
                    "estimated_cost": estimate.estimated_cost,
                    "estimated_seconds": estimate.estimated_seconds,
                    "status": "DEFERRED" if deferral else "PENDING",
                    "deferral_reason": deferral,
                }
            )
        if selected_count == 0:
            raise ValueError("RESEARCH_BUDGET_SELECTS_NOTHING")
        payload = {
            "pool_id": pool_id,
            "trading_date": date_value,
            "policy": asdict(policy),
            "items": planned,
        }
        budget_run_id = "research-budget-" + self._digest(payload, 24)
        connection = self._connect()
        try:
            connection.execute("BEGIN TRANSACTION")
            connection.execute(
                """
                INSERT INTO research_budget_runs VALUES
                (?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT (budget_run_id) DO NOTHING
                """,
                [
                    budget_run_id,
                    pool_id,
                    date_value,
                    "PENDING",
                    json.dumps(asdict(policy), sort_keys=True),
                    selected_count,
                    len(planned) - selected_count,
                    selected_cost,
                    selected_seconds,
                    started_at,
                    started_at,
                ],
            )
            for item in planned:
                work_id = "research-work-" + self._digest(
                    {
                        "budget_run_id": budget_run_id,
                        "symbol": item["symbol"],
                    },
                    24,
                )
                connection.execute(
                    """
                    INSERT INTO research_budget_items VALUES
                    (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT (work_id) DO NOTHING
                    """,
                    [
                        work_id,
                        budget_run_id,
                        item["symbol"],
                        item["priority"],
                        item["estimated_cost"],
                        item["estimated_seconds"],
                        item["status"],
                        0,
                        "",
                        None,
                        None,
                        0.0,
                        0.0,
                        item["deferral_reason"],
                    ],
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        run = self.get_run(budget_run_id)
        if run is None:  # pragma: no cover
            raise RuntimeError("RESEARCH_BUDGET_PERSIST_FAILED")
        return run

    def claim_batch(
        self,
        budget_run_id: str,
        *,
        now: datetime,
    ) -> list[dict[str, Any]]:
        _aware(now, "RESEARCH_BUDGET_CLAIM_TZ_REQUIRED")
        connection = self._connect()
        try:
            connection.execute("BEGIN TRANSACTION")
            run = connection.execute(
                "SELECT * FROM research_budget_runs WHERE budget_run_id=?",
                [budget_run_id],
            ).fetchone()
            if run is None:
                raise KeyError(budget_run_id)
            policy = ResearchBudgetPolicy(**json.loads(run[4]))
            self._recover_timeouts(
                connection,
                budget_run_id,
                policy,
                now,
            )
            actual_cost = float(
                connection.execute(
                    """
                    SELECT coalesce(sum(actual_cost), 0)
                    FROM research_budget_items
                    WHERE budget_run_id=?
                    """,
                    [budget_run_id],
                ).fetchone()[0]
            )
            elapsed = (now - run[9]).total_seconds()
            deferral_reason = ""
            if actual_cost >= policy.max_estimated_cost:
                deferral_reason = "ACTUAL_COST_QUOTA"
            elif elapsed >= policy.max_runtime_seconds:
                deferral_reason = "RUNTIME_QUOTA"
            if deferral_reason:
                connection.execute(
                    """
                    UPDATE research_budget_items
                    SET status='DEFERRED', deferral_reason=?
                    WHERE budget_run_id=? AND status IN ('PENDING', 'RETRY')
                    """,
                    [deferral_reason, budget_run_id],
                )
                connection.commit()
                self._refresh_run(budget_run_id, now=now)
                return []
            rows = connection.execute(
                """
                SELECT work_id FROM research_budget_items
                WHERE budget_run_id=? AND status IN ('PENDING', 'RETRY')
                ORDER BY priority, symbol
                LIMIT ?
                """,
                [budget_run_id, policy.batch_size],
            ).fetchall()
            for work_id, in rows:
                connection.execute(
                    """
                    UPDATE research_budget_items
                    SET status='RUNNING', attempts=attempts+1,
                        attempt_started_at=?, last_error=''
                    WHERE work_id=?
                    """,
                    [now, work_id],
                )
            connection.execute(
                """
                UPDATE research_budget_runs
                SET status='RUNNING', updated_at=?
                WHERE budget_run_id=?
                """,
                [now, budget_run_id],
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        return [
            item
            for work_id, in rows
            if (item := self.get_item(str(work_id))) is not None
        ]

    def finish(
        self,
        work_id: str,
        *,
        success: bool,
        error_code: str = "",
        actual_cost: float,
        actual_seconds: float,
        now: datetime,
    ) -> dict[str, Any]:
        _aware(now, "RESEARCH_BUDGET_FINISH_TZ_REQUIRED")
        cost = float(actual_cost)
        seconds = float(actual_seconds)
        if (
            not math.isfinite(cost)
            or not math.isfinite(seconds)
            or cost < 0
            or seconds < 0
        ):
            raise ValueError("RESEARCH_BUDGET_ACTUALS_INVALID")
        connection = self._connect()
        try:
            connection.execute("BEGIN TRANSACTION")
            row = connection.execute(
                """
                SELECT budget_run_id, status, attempts
                FROM research_budget_items WHERE work_id=?
                """,
                [work_id],
            ).fetchone()
            if row is None:
                raise KeyError(work_id)
            if row[1] != "RUNNING":
                raise ValueError("RESEARCH_WORK_NOT_RUNNING")
            policy_row = connection.execute(
                """
                SELECT policy_json FROM research_budget_runs
                WHERE budget_run_id=?
                """,
                [row[0]],
            ).fetchone()
            policy = ResearchBudgetPolicy(**json.loads(policy_row[0]))
            if success:
                status = "COMPLETED"
                error = ""
                completed_at = now
            else:
                error = error_code.strip() or "RESEARCH_ATTEMPT_FAILED"
                status = "RETRY" if int(row[2]) <= policy.max_retries else "FAILED"
                completed_at = now if status == "FAILED" else None
            connection.execute(
                """
                UPDATE research_budget_items
                SET status=?, last_error=?, completed_at=?,
                    actual_cost=actual_cost+?,
                    actual_seconds=actual_seconds+?,
                    attempt_started_at=NULL
                WHERE work_id=?
                """,
                [status, error, completed_at, cost, seconds, work_id],
            )
            connection.commit()
            budget_run_id = str(row[0])
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        self._refresh_run(budget_run_id, now=now)
        item = self.get_item(work_id)
        if item is None:  # pragma: no cover
            raise RuntimeError("RESEARCH_WORK_PERSIST_FAILED")
        return item

    def get_run(self, budget_run_id: str) -> dict[str, Any] | None:
        connection = self._connect(read_only=True)
        try:
            row = connection.execute(
                "SELECT * FROM research_budget_runs WHERE budget_run_id=?",
                [budget_run_id],
            ).fetchone()
            items = (
                connection.execute(
                    """
                    SELECT work_id FROM research_budget_items
                    WHERE budget_run_id=?
                    ORDER BY priority, symbol
                    """,
                    [budget_run_id],
                ).fetchall()
                if row is not None
                else []
            )
        finally:
            connection.close()
        if row is None:
            return None
        return {
            "budget_run_id": str(row[0]),
            "pool_id": str(row[1]),
            "trading_date": str(row[2]),
            "status": str(row[3]),
            "policy": json.loads(row[4]),
            "planned_count": int(row[5]),
            "deferred_count": int(row[6]),
            "estimated_cost": float(row[7]),
            "estimated_seconds": float(row[8]),
            "started_at": row[9],
            "updated_at": row[10],
            "items": [
                item
                for work_id, in items
                if (item := self.get_item(str(work_id))) is not None
            ],
        }

    def get_item(self, work_id: str) -> dict[str, Any] | None:
        connection = self._connect(read_only=True)
        try:
            row = connection.execute(
                "SELECT * FROM research_budget_items WHERE work_id=?",
                [work_id],
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        return {
            "work_id": str(row[0]),
            "budget_run_id": str(row[1]),
            "symbol": str(row[2]),
            "priority": int(row[3]),
            "estimated_cost": float(row[4]),
            "estimated_seconds": float(row[5]),
            "status": str(row[6]),
            "attempts": int(row[7]),
            "last_error": str(row[8] or ""),
            "attempt_started_at": row[9],
            "completed_at": row[10],
            "actual_cost": float(row[11]),
            "actual_seconds": float(row[12]),
            "deferral_reason": str(row[13] or ""),
        }

    def _recover_timeouts(
        self,
        connection,
        budget_run_id: str,
        policy: ResearchBudgetPolicy,
        now: datetime,
    ) -> None:
        rows = connection.execute(
            """
            SELECT work_id, attempts, attempt_started_at
            FROM research_budget_items
            WHERE budget_run_id=? AND status='RUNNING'
            """,
            [budget_run_id],
        ).fetchall()
        for work_id, attempts, started_at in rows:
            elapsed = (now - started_at).total_seconds()
            if elapsed <= policy.attempt_timeout_seconds:
                continue
            status = "RETRY" if int(attempts) <= policy.max_retries else "FAILED"
            connection.execute(
                """
                UPDATE research_budget_items
                SET status=?, last_error='RESEARCH_ATTEMPT_TIMEOUT',
                    attempt_started_at=NULL, completed_at=?,
                    actual_seconds=actual_seconds+?
                WHERE work_id=?
                """,
                [
                    status,
                    now if status == "FAILED" else None,
                    policy.attempt_timeout_seconds,
                    work_id,
                ],
            )

    def _refresh_run(self, budget_run_id: str, *, now: datetime) -> None:
        connection = self._connect()
        try:
            counts = dict(
                connection.execute(
                    """
                    SELECT status, count(*) FROM research_budget_items
                    WHERE budget_run_id=?
                    GROUP BY status
                    """,
                    [budget_run_id],
                ).fetchall()
            )
            if any(counts.get(status, 0) for status in _ACTIONABLE):
                status = "RUNNING"
            elif counts.get("FAILED", 0):
                status = "COMPLETED_WITH_ERRORS"
            elif counts.get("DEFERRED", 0):
                status = "COMPLETED_WITH_DEFERRED"
            else:
                status = "COMPLETED"
            connection.execute(
                """
                UPDATE research_budget_runs SET status=?, updated_at=?
                WHERE budget_run_id=?
                """,
                [status, now, budget_run_id],
            )
            connection.commit()
        finally:
            connection.close()

    @staticmethod
    def _digest(payload: dict[str, Any], length: int) -> str:
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode()).hexdigest()[:length]
