"""Daily universe/focus-pool/research-budget quality evidence and gate."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import duckdb

from .focus_pool import FocusPoolStore
from .research_budget import ResearchBudgetStore
from .universe_registry import UniverseRegistryStore

_EVIDENCE_TYPES = {"REAL", "SYNTHETIC"}


def _aware(value: datetime, code: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(code)


@dataclass(frozen=True)
class UniverseResearchGate:
    required_sessions: int = 20
    min_screening_coverage: float = 1.0
    min_research_completion: float = 0.95
    max_research_failure_rate: float = 0.05
    max_cost_utilization: float = 1.0
    max_duration_utilization: float = 1.0

    def __post_init__(self) -> None:
        if self.required_sessions < 1:
            raise ValueError("UNIVERSE_REPORT_SESSION_COUNT_INVALID")
        values = (
            self.min_screening_coverage,
            self.min_research_completion,
            self.max_research_failure_rate,
            self.max_cost_utilization,
            self.max_duration_utilization,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("UNIVERSE_REPORT_GATE_NON_FINITE")
        if (
            not 0 <= self.min_screening_coverage <= 1
            or not 0 <= self.min_research_completion <= 1
            or not 0 <= self.max_research_failure_rate <= 1
            or self.max_cost_utilization <= 0
            or self.max_duration_utilization <= 0
        ):
            raise ValueError("UNIVERSE_REPORT_GATE_INVALID")


class UniverseResearchQualityStore:
    """Captures immutable daily evidence without mixing real and synthetic runs."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self.universes = UniverseRegistryStore(db_path)
        self.pools = FocusPoolStore(db_path)
        self.budgets = ResearchBudgetStore(db_path)
        self._migrate()

    def _connect(self, *, read_only: bool = False):
        return duckdb.connect(self.db_path, read_only=read_only)

    def _migrate(self) -> None:
        connection = self._connect()
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS universe_research_observations (
                    observation_id TEXT PRIMARY KEY,
                    evidence_type TEXT,
                    trading_date TEXT,
                    universe_version TEXT,
                    pool_id TEXT,
                    budget_run_id TEXT,
                    universe_asset_count INTEGER,
                    screened_asset_count INTEGER,
                    focus_member_count INTEGER,
                    planned_research_count INTEGER,
                    completed_research_count INTEGER,
                    failed_research_count INTEGER,
                    deferred_research_count INTEGER,
                    actual_cost DOUBLE,
                    cost_budget DOUBLE,
                    duration_seconds DOUBLE,
                    duration_budget_seconds DOUBLE,
                    budget_status TEXT,
                    created_at TIMESTAMPTZ,
                    UNIQUE (evidence_type, trading_date)
                )
                """
            )
            connection.commit()
        finally:
            connection.close()

    def capture(
        self,
        *,
        evidence_type: str,
        trading_date: str,
        universe_version: str,
        pool_id: str,
        budget_run_id: str,
        created_at: datetime,
    ) -> dict[str, Any]:
        evidence = evidence_type.strip().upper()
        if evidence not in _EVIDENCE_TYPES:
            raise ValueError("UNIVERSE_REPORT_EVIDENCE_TYPE_INVALID")
        _aware(created_at, "UNIVERSE_REPORT_CREATED_TZ_REQUIRED")
        try:
            session = date.fromisoformat(trading_date)
        except ValueError as exc:
            raise ValueError("UNIVERSE_REPORT_DATE_INVALID") from exc
        if session.weekday() >= 5:
            raise ValueError("UNIVERSE_REPORT_NON_TRADING_WEEKDAY")
        universe = self.universes.get_version(universe_version)
        pool = self.pools.get(pool_id)
        budget = self.budgets.get_run(budget_run_id)
        if universe is None or pool is None or budget is None:
            raise ValueError("UNIVERSE_REPORT_REFERENCE_NOT_FOUND")
        if pool["universe_version"] != universe_version:
            raise ValueError("UNIVERSE_REPORT_POOL_UNIVERSE_MISMATCH")
        if budget["pool_id"] != pool_id:
            raise ValueError("UNIVERSE_REPORT_BUDGET_POOL_MISMATCH")
        if budget["trading_date"] != trading_date:
            raise ValueError("UNIVERSE_REPORT_BUDGET_DATE_MISMATCH")
        if budget["status"] in {"PENDING", "RUNNING"}:
            raise ValueError("UNIVERSE_REPORT_BUDGET_NOT_TERMINAL")
        if pool["as_of"].date() > session or universe["as_of"].date() > session:
            raise ValueError("UNIVERSE_REPORT_REFERENCE_FROM_FUTURE")
        if created_at < budget["updated_at"]:
            raise ValueError("UNIVERSE_REPORT_BEFORE_BUDGET_UPDATE")
        screened = len(pool["decisions"])
        statuses: dict[str, int] = {}
        actual_cost = 0.0
        for item in budget["items"]:
            statuses[item["status"]] = statuses.get(item["status"], 0) + 1
            actual_cost += item["actual_cost"]
        policy = budget["policy"]
        duration = max(
            0.0,
            (budget["updated_at"] - budget["started_at"]).total_seconds(),
        )
        payload = {
            "evidence_type": evidence,
            "trading_date": trading_date,
            "universe_version": universe_version,
            "pool_id": pool_id,
            "budget_run_id": budget_run_id,
            "universe_asset_count": universe["asset_count"],
            "screened_asset_count": screened,
            "focus_member_count": pool["member_count"],
            "planned_research_count": budget["planned_count"],
            "completed_research_count": statuses.get("COMPLETED", 0),
            "failed_research_count": statuses.get("FAILED", 0),
            "deferred_research_count": statuses.get("DEFERRED", 0),
            "actual_cost": actual_cost,
            "cost_budget": float(policy["max_estimated_cost"]),
            "duration_seconds": duration,
            "duration_budget_seconds": float(policy["max_runtime_seconds"]),
            "budget_status": budget["status"],
        }
        observation_id = "universe-observation-" + self._digest(payload, 24)
        connection = self._connect()
        try:
            existing = connection.execute(
                """
                SELECT observation_id FROM universe_research_observations
                WHERE evidence_type=? AND trading_date=?
                """,
                [evidence, trading_date],
            ).fetchone()
            if existing is not None and str(existing[0]) != observation_id:
                raise ValueError("UNIVERSE_REPORT_DATE_CONFLICT")
            connection.execute(
                """
                INSERT INTO universe_research_observations VALUES
                (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT (observation_id) DO NOTHING
                """,
                [
                    observation_id,
                    evidence,
                    trading_date,
                    universe_version,
                    pool_id,
                    budget_run_id,
                    payload["universe_asset_count"],
                    payload["screened_asset_count"],
                    payload["focus_member_count"],
                    payload["planned_research_count"],
                    payload["completed_research_count"],
                    payload["failed_research_count"],
                    payload["deferred_research_count"],
                    payload["actual_cost"],
                    payload["cost_budget"],
                    payload["duration_seconds"],
                    payload["duration_budget_seconds"],
                    payload["budget_status"],
                    created_at,
                ],
            )
            connection.commit()
        finally:
            connection.close()
        observation = self.get(observation_id)
        if observation is None:  # pragma: no cover
            raise RuntimeError("UNIVERSE_REPORT_PERSIST_FAILED")
        return observation

    def report(
        self,
        *,
        evidence_type: str,
        through_date: str,
        gate: UniverseResearchGate | None = None,
    ) -> dict[str, Any]:
        evidence = evidence_type.strip().upper()
        if evidence not in _EVIDENCE_TYPES:
            raise ValueError("UNIVERSE_REPORT_EVIDENCE_TYPE_INVALID")
        policy = gate or UniverseResearchGate()
        date.fromisoformat(through_date)
        connection = self._connect(read_only=True)
        try:
            ids = connection.execute(
                """
                SELECT observation_id
                FROM universe_research_observations
                WHERE evidence_type=? AND trading_date<=?
                ORDER BY trading_date DESC
                LIMIT ?
                """,
                [evidence, through_date, policy.required_sessions],
            ).fetchall()
        finally:
            connection.close()
        observations = [
            item
            for observation_id, in reversed(ids)
            if (item := self.get(str(observation_id))) is not None
        ]
        failures: list[dict[str, str]] = []
        daily = []
        for item in observations:
            planned = item["planned_research_count"]
            coverage = self._ratio(
                item["screened_asset_count"],
                item["universe_asset_count"],
            )
            completion = self._ratio(
                item["completed_research_count"],
                planned,
            )
            failure_rate = self._ratio(
                item["failed_research_count"],
                planned,
            )
            cost_utilization = self._ratio(
                item["actual_cost"],
                item["cost_budget"],
            )
            duration_utilization = self._ratio(
                item["duration_seconds"],
                item["duration_budget_seconds"],
            )
            reasons = []
            if coverage < policy.min_screening_coverage:
                reasons.append("SCREENING_COVERAGE")
            if completion < policy.min_research_completion:
                reasons.append("RESEARCH_COMPLETION")
            if failure_rate > policy.max_research_failure_rate:
                reasons.append("RESEARCH_FAILURE_RATE")
            if cost_utilization > policy.max_cost_utilization:
                reasons.append("RESEARCH_COST")
            if duration_utilization > policy.max_duration_utilization:
                reasons.append("RESEARCH_DURATION")
            if item["budget_status"] not in {
                "COMPLETED",
                "COMPLETED_WITH_DEFERRED",
            }:
                reasons.append("BUDGET_STATUS")
            for reason in reasons:
                failures.append(
                    {
                        "trading_date": item["trading_date"],
                        "reason": reason,
                    }
                )
            daily.append(
                {
                    "trading_date": item["trading_date"],
                    "screening_coverage": coverage,
                    "research_completion": completion,
                    "research_failure_rate": failure_rate,
                    "cost_utilization": cost_utilization,
                    "duration_utilization": duration_utilization,
                    "reasons": reasons,
                }
            )
        if len(observations) < policy.required_sessions:
            failures.append(
                {
                    "trading_date": "",
                    "reason": "INSUFFICIENT_TRADING_SESSIONS",
                }
            )
        return {
            "evidence_type": evidence,
            "through_date": through_date,
            "required_sessions": policy.required_sessions,
            "observed_sessions": len(observations),
            "passed": not failures,
            "failures": failures,
            "daily": daily,
            "gate": asdict(policy),
            "observation_ids": [
                item["observation_id"] for item in observations
            ],
        }

    def get(self, observation_id: str) -> dict[str, Any] | None:
        connection = self._connect(read_only=True)
        try:
            row = connection.execute(
                """
                SELECT * FROM universe_research_observations
                WHERE observation_id=?
                """,
                [observation_id],
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        return {
            "observation_id": str(row[0]),
            "evidence_type": str(row[1]),
            "trading_date": str(row[2]),
            "universe_version": str(row[3]),
            "pool_id": str(row[4]),
            "budget_run_id": str(row[5]),
            "universe_asset_count": int(row[6]),
            "screened_asset_count": int(row[7]),
            "focus_member_count": int(row[8]),
            "planned_research_count": int(row[9]),
            "completed_research_count": int(row[10]),
            "failed_research_count": int(row[11]),
            "deferred_research_count": int(row[12]),
            "actual_cost": float(row[13]),
            "cost_budget": float(row[14]),
            "duration_seconds": float(row[15]),
            "duration_budget_seconds": float(row[16]),
            "budget_status": str(row[17]),
            "created_at": row[18],
        }

    @staticmethod
    def _ratio(numerator: float, denominator: float) -> float:
        return float(numerator) / float(denominator) if denominator else 0.0

    @staticmethod
    def _digest(payload: dict[str, Any], length: int) -> str:
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode()).hexdigest()[:length]
