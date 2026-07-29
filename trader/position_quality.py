"""Daily, replayable broker/local/PositionPlan consistency evidence."""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import duckdb

from .models import Position
from .position_plans import PositionPlanStore

_NEW_YORK = ZoneInfo("America/New_York")
_OBSERVATION_KINDS = {"REAL", "SYNTHETIC"}


def _position_map(items: Iterable[Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for item in items:
        symbol = getattr(item, "symbol", None) or (
            item.get("symbol") if isinstance(item, dict) else None
        )
        quantity = getattr(item, "qty", None)
        if quantity is None and isinstance(item, dict):
            quantity = item.get("qty")
        if symbol:
            values[str(symbol).strip().upper()] = float(quantity or 0.0)
    return dict(sorted(values.items()))


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


class PositionQualityStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._migrate()

    def _connect(self, *, read_only: bool = False):
        return duckdb.connect(self.db_path, read_only=read_only)

    def _migrate(self) -> None:
        connection = self._connect()
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS position_quality_observations (
                    observation_id TEXT PRIMARY KEY,
                    observation_kind TEXT,
                    trading_date DATE,
                    observed_at TIMESTAMPTZ,
                    broker_positions_json TEXT,
                    local_positions_json TEXT,
                    plan_positions_json TEXT,
                    mismatches_json TEXT,
                    version_errors_json TEXT,
                    duplicate_adjustments INTEGER,
                    ok BOOLEAN
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS position_quality_reports (
                    report_id TEXT PRIMARY KEY,
                    observation_kind TEXT,
                    as_of DATE,
                    required_days INTEGER,
                    report_json TEXT,
                    created_at TIMESTAMPTZ
                )
                """
            )
            connection.commit()
        finally:
            connection.close()

    def capture(
        self,
        *,
        broker_positions: Iterable[Any],
        local_positions: Iterable[Position],
        observed_at: datetime,
        observation_kind: str = "REAL",
    ) -> str:
        plan_store = PositionPlanStore(self.db_path)
        plan_positions = {
            plan.symbol: plan.open_quantity
            for plan in plan_store.recover_open()
        }
        return self.record_snapshot(
            trading_date=observed_at.astimezone(_NEW_YORK).date(),
            observed_at=observed_at,
            broker_positions=_position_map(broker_positions),
            local_positions=_position_map(local_positions),
            plan_positions=dict(sorted(plan_positions.items())),
            version_errors=self._version_errors(),
            duplicate_adjustments=self._duplicate_adjustments(),
            observation_kind=observation_kind,
        )

    def record_snapshot(
        self,
        *,
        trading_date: date,
        observed_at: datetime,
        broker_positions: dict[str, float],
        local_positions: dict[str, float],
        plan_positions: dict[str, float],
        version_errors: list[str],
        duplicate_adjustments: int,
        observation_kind: str,
    ) -> str:
        kind = observation_kind.strip().upper()
        if kind not in _OBSERVATION_KINDS:
            raise ValueError("POSITION_QUALITY_KIND_INVALID")
        if observed_at.tzinfo is None or observed_at.utcoffset() is None:
            raise ValueError("POSITION_QUALITY_TIME_TZ_REQUIRED")
        if duplicate_adjustments < 0:
            raise ValueError("POSITION_QUALITY_DUPLICATE_COUNT_INVALID")
        broker = self._validated_map(broker_positions)
        local = self._validated_map(local_positions)
        plans = self._validated_map(plan_positions)
        mismatches = self._mismatches(broker, local, plans)
        errors = sorted(
            {
                str(error).strip()
                for error in version_errors
                if str(error).strip()
            }
        )
        payload = {
            "kind": kind,
            "date": trading_date.isoformat(),
            "broker": broker,
            "local": local,
            "plans": plans,
            "mismatches": mismatches,
            "version_errors": errors,
            "duplicate_adjustments": duplicate_adjustments,
        }
        observation_id = "position-quality-" + hashlib.sha256(
            _canonical(payload).encode()
        ).hexdigest()[:24]
        connection = self._connect()
        try:
            connection.execute(
                """
                INSERT INTO position_quality_observations VALUES
                (?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT (observation_id) DO NOTHING
                """,
                [
                    observation_id,
                    kind,
                    trading_date,
                    observed_at,
                    _canonical(broker),
                    _canonical(local),
                    _canonical(plans),
                    _canonical(mismatches),
                    _canonical(errors),
                    duplicate_adjustments,
                    not mismatches
                    and not errors
                    and duplicate_adjustments == 0,
                ],
            )
            connection.commit()
        finally:
            connection.close()
        return observation_id

    def build_report(
        self,
        *,
        as_of: date,
        required_days: int = 30,
        observation_kind: str = "REAL",
        persist: bool = True,
    ) -> dict[str, Any]:
        if required_days < 1:
            raise ValueError("POSITION_QUALITY_REQUIRED_DAYS_INVALID")
        kind = observation_kind.strip().upper()
        if kind not in _OBSERVATION_KINDS:
            raise ValueError("POSITION_QUALITY_KIND_INVALID")
        connection = self._connect(read_only=True)
        try:
            rows = connection.execute(
                """
                SELECT trading_date, observed_at, mismatches_json,
                       version_errors_json, duplicate_adjustments, ok
                FROM (
                    SELECT *, row_number() OVER (
                        PARTITION BY trading_date
                        ORDER BY observed_at DESC, observation_id DESC
                    ) AS row_number
                    FROM position_quality_observations
                    WHERE observation_kind=? AND trading_date<=?
                )
                WHERE row_number=1
                ORDER BY trading_date DESC
                LIMIT ?
                """,
                [kind, as_of, required_days],
            ).fetchall()
        finally:
            connection.close()
        selected = list(reversed(rows))
        failed_dates = [
            row[0].isoformat()
            for row in selected
            if not bool(row[5])
        ]
        mismatch_count = sum(
            len(json.loads(row[2] or "[]"))
            for row in selected
        )
        silent_rewrites = sum(
            len(json.loads(row[3] or "[]"))
            for row in selected
        )
        duplicate_adjustments = sum(int(row[4] or 0) for row in selected)
        observed_days = len(selected)
        report = {
            "observation_kind": kind,
            "as_of": as_of.isoformat(),
            "required_days": required_days,
            "observed_days": observed_days,
            "passed_days": observed_days - len(failed_dates),
            "failed_dates": failed_dates,
            "position_mismatches": mismatch_count,
            "silent_rewrites": silent_rewrites,
            "duplicate_adjustments": duplicate_adjustments,
            "ready": (
                observed_days >= required_days
                and not failed_dates
                and mismatch_count == 0
                and silent_rewrites == 0
                and duplicate_adjustments == 0
            ),
        }
        if persist:
            self._persist_report(report)
        return report

    def _persist_report(self, report: dict[str, Any]) -> None:
        payload = _canonical(report)
        report_id = "position-report-" + hashlib.sha256(
            payload.encode()
        ).hexdigest()[:24]
        connection = self._connect()
        try:
            connection.execute(
                """
                INSERT INTO position_quality_reports VALUES
                (?,?,?,?,?,?)
                ON CONFLICT (report_id) DO NOTHING
                """,
                [
                    report_id,
                    report["observation_kind"],
                    date.fromisoformat(report["as_of"]),
                    report["required_days"],
                    payload,
                    datetime.now(timezone.utc),
                ],
            )
            connection.commit()
        finally:
            connection.close()

    def _version_errors(self) -> list[str]:
        connection = self._connect(read_only=True)
        try:
            versions = connection.execute(
                """
                SELECT position_plan_id, version_id, version,
                       parent_version_id, symbol, side,
                       source_trade_plan_id, initial_fill_id,
                       initial_entry_price, initial_quantity,
                       change_reason
                FROM position_plan_versions
                ORDER BY position_plan_id, version
                """
            ).fetchall()
            fill_versions = {
                str(row[0])
                for row in connection.execute(
                    """
                    SELECT version_id FROM position_plan_fill_events
                    """
                ).fetchall()
            }
            has_adjustments = connection.execute(
                """
                SELECT 1 FROM information_schema.tables
                WHERE table_name='position_adjustments'
                """
            ).fetchone()
            adjustment_versions = (
                {
                    str(row[0])
                    for row in connection.execute(
                        """
                        SELECT to_version_id FROM position_adjustments
                        """
                    ).fetchall()
                }
                if has_adjustments
                else set()
            )
        finally:
            connection.close()
        grouped: dict[str, list[tuple]] = defaultdict(list)
        for row in versions:
            grouped[str(row[0])].append(row)
        errors: list[str] = []
        for plan_id, rows in grouped.items():
            baseline = rows[0][4:10]
            previous_id = ""
            for expected, row in enumerate(rows, start=1):
                version_id = str(row[1])
                version = int(row[2])
                parent = str(row[3] or "")
                reason = str(row[10] or "")
                if version != expected:
                    errors.append(
                        f"{plan_id}:VERSION_SEQUENCE:{version}"
                    )
                if (expected == 1 and parent) or (
                    expected > 1 and parent != previous_id
                ):
                    errors.append(
                        f"{plan_id}:PARENT_LINK:{version}"
                    )
                if row[4:10] != baseline:
                    errors.append(
                        f"{plan_id}:BASELINE_REWRITE:{version}"
                    )
                if expected > 1:
                    if reason.startswith("INVALIDATION:"):
                        if version_id not in adjustment_versions:
                            errors.append(
                                f"{plan_id}:SILENT_INVALIDATION:{version}"
                            )
                    elif reason in {
                        "ADDITIONAL_FILL",
                        "REDUCTION_FILL",
                        "CLOSE_FILL",
                    }:
                        if version_id not in fill_versions:
                            errors.append(
                                f"{plan_id}:SILENT_FILL:{version}"
                            )
                    else:
                        errors.append(
                            f"{plan_id}:UNKNOWN_CHANGE:{version}"
                        )
                previous_id = version_id
        return sorted(set(errors))

    def _duplicate_adjustments(self) -> int:
        connection = self._connect(read_only=True)
        try:
            table = connection.execute(
                """
                SELECT 1 FROM information_schema.tables
                WHERE table_name='position_adjustments'
                """
            ).fetchone()
            if table is None:
                return 0
            return int(
                connection.execute(
                    """
                    SELECT coalesce(sum(count_value - 1), 0)
                    FROM (
                        SELECT count(*) AS count_value
                        FROM position_adjustments
                        GROUP BY event_id
                        HAVING count(*) > 1
                    )
                    """
                ).fetchone()[0]
                or 0
            )
        finally:
            connection.close()

    @staticmethod
    def _validated_map(values: dict[str, float]) -> dict[str, float]:
        normalized: dict[str, float] = {}
        for symbol, quantity in values.items():
            key = str(symbol).strip().upper()
            value = float(quantity)
            if not key or not math.isfinite(value) or value < 0:
                raise ValueError("POSITION_QUALITY_POSITION_INVALID")
            normalized[key] = value
        return dict(sorted(normalized.items()))

    @staticmethod
    def _mismatches(
        broker: dict[str, float],
        local: dict[str, float],
        plans: dict[str, float],
    ) -> list[str]:
        mismatches: list[str] = []
        for symbol in sorted(set(broker) | set(local) | set(plans)):
            values = (
                broker.get(symbol, 0.0),
                local.get(symbol, 0.0),
                plans.get(symbol, 0.0),
            )
            if max(values) - min(values) > 1e-8:
                mismatches.append(
                    f"{symbol}:broker={values[0]:g},"
                    f"local={values[1]:g},plan={values[2]:g}"
                )
        return mismatches
