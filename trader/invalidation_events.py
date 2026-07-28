"""Validated, source-backed invalidation facts for PositionPlan."""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

import duckdb

from .models import (
    InvalidationEvent,
    InvalidationEventType,
    InvalidationSource,
    PositionPlan,
    PositionPlanStatus,
    Side,
    utc_now,
)

_ALLOWED_SOURCES = {
    InvalidationEventType.PRICE_STOP: {
        InvalidationSource.MARKET_DATA,
    },
    InvalidationEventType.BROKER_RESTRICTION: {
        InvalidationSource.BROKER,
    },
    InvalidationEventType.CORPORATE_ACTION: {
        InvalidationSource.CORPORATE_ACTION_DATA,
    },
    InvalidationEventType.TRADING_RESTRICTION: {
        InvalidationSource.BROKER,
        InvalidationSource.EXCHANGE,
    },
    InvalidationEventType.STRATEGY_INVALIDATED: {
        InvalidationSource.STRATEGY_ENGINE,
    },
}

_MAX_AGE = {
    InvalidationEventType.PRICE_STOP: timedelta(minutes=15),
    InvalidationEventType.BROKER_RESTRICTION: timedelta(days=1),
    InvalidationEventType.CORPORATE_ACTION: timedelta(days=30),
    InvalidationEventType.TRADING_RESTRICTION: timedelta(days=1),
    InvalidationEventType.STRATEGY_INVALIDATED: timedelta(days=1),
}


def build_invalidation_event(
    *,
    plan: PositionPlan,
    event_type: InvalidationEventType,
    source: InvalidationSource,
    source_event_id: str,
    as_of: datetime,
    observed_at: datetime,
    facts: Mapping[str, Any],
    evidence_refs: tuple[str, ...],
) -> InvalidationEvent:
    """Build a deterministic event ID from canonical source facts."""
    facts_json = json.dumps(
        dict(facts),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    rule_id = event_type.value
    raw = "|".join(
        (
            plan.position_plan_id,
            plan.version_id,
            event_type.value,
            source.value,
            source_event_id.strip(),
            rule_id,
            as_of.astimezone(timezone.utc).isoformat(),
            facts_json,
        )
    )
    event_id = "invalidation-" + hashlib.sha256(
        raw.encode()
    ).hexdigest()[:24]
    return InvalidationEvent(
        event_id=event_id,
        position_plan_id=plan.position_plan_id,
        position_plan_version_id=plan.version_id,
        symbol=plan.symbol,
        event_type=event_type,
        source=source,
        source_event_id=source_event_id,
        rule_id=rule_id,
        as_of=as_of,
        observed_at=observed_at,
        facts_json=facts_json,
        evidence_refs=evidence_refs,
    )


class InvalidationEventValidator:
    def validate(
        self,
        event: InvalidationEvent,
        *,
        plan: PositionPlan,
        received_at: datetime | None = None,
    ) -> dict[str, Any]:
        received = received_at or utc_now()
        if received.tzinfo is None or received.utcoffset() is None:
            raise ValueError("INVALIDATION_RECEIVED_AT_TZ_REQUIRED")
        if plan.status == PositionPlanStatus.CLOSED:
            raise ValueError("INVALIDATION_POSITION_PLAN_CLOSED")
        if (
            event.position_plan_id != plan.position_plan_id
            or event.position_plan_version_id != plan.version_id
            or event.symbol != plan.symbol
        ):
            raise ValueError("INVALIDATION_POSITION_PLAN_MISMATCH")
        if event.rule_id != event.event_type.value:
            raise ValueError("INVALIDATION_RULE_TYPE_MISMATCH")
        if event.rule_id not in plan.invalidation_rules:
            raise ValueError("INVALIDATION_RULE_NOT_CONFIGURED")
        if event.source not in _ALLOWED_SOURCES[event.event_type]:
            raise ValueError("INVALIDATION_SOURCE_NOT_ALLOWED")
        if event.as_of > event.observed_at or event.observed_at > received:
            raise ValueError("INVALIDATION_EVENT_TIME_ORDER_INVALID")
        if event.as_of < plan.created_at:
            raise ValueError("INVALIDATION_EVENT_PREDATES_PLAN")
        if received - event.as_of > _MAX_AGE[event.event_type]:
            raise ValueError("INVALIDATION_EVENT_STALE")
        facts = self._facts(event.facts_json)
        self._validate_facts(event.event_type, facts, plan)
        expected = build_invalidation_event(
            plan=plan,
            event_type=event.event_type,
            source=event.source,
            source_event_id=event.source_event_id,
            as_of=event.as_of,
            observed_at=event.observed_at,
            facts=facts,
            evidence_refs=event.evidence_refs,
        )
        if event.event_id != expected.event_id:
            raise ValueError("INVALIDATION_EVENT_ID_MISMATCH")
        return facts

    @staticmethod
    def _facts(value: str) -> dict[str, Any]:
        try:
            facts = json.loads(value)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("INVALIDATION_FACTS_INVALID") from exc
        if not isinstance(facts, dict) or not facts:
            raise ValueError("INVALIDATION_FACTS_INVALID")
        if json.dumps(
            facts,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ) != value:
            raise ValueError("INVALIDATION_FACTS_NOT_CANONICAL")
        return facts

    @staticmethod
    def _validate_facts(
        event_type: InvalidationEventType,
        facts: dict[str, Any],
        plan: PositionPlan,
    ) -> None:
        if event_type == InvalidationEventType.PRICE_STOP:
            try:
                trigger = float(facts["trigger_price"])
                threshold = float(facts["threshold_price"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("INVALIDATION_PRICE_FACTS_INVALID") from exc
            if (
                not math.isfinite(trigger)
                or not math.isfinite(threshold)
                or trigger <= 0
                or threshold <= 0
                or not math.isclose(
                    threshold,
                    plan.stop_loss,
                    rel_tol=1e-9,
                    abs_tol=1e-8,
                )
            ):
                raise ValueError("INVALIDATION_PRICE_FACTS_INVALID")
            triggered = (
                trigger <= threshold
                if plan.side == Side.BUY
                else trigger >= threshold
            )
            if not triggered:
                raise ValueError("INVALIDATION_PRICE_NOT_TRIGGERED")
            return
        if event_type == InvalidationEventType.STRATEGY_INVALIDATED:
            if (
                not str(facts.get("evaluation_id", "")).strip()
                or facts.get("valid") is not False
            ):
                raise ValueError("INVALIDATION_STRATEGY_FACTS_INVALID")
            return
        if (
            facts.get("active") is not True
            or not str(facts.get("fact_code", "")).strip()
        ):
            raise ValueError("INVALIDATION_EXTERNAL_FACTS_INVALID")


class InvalidationEventStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self.validator = InvalidationEventValidator()
        self._migrate()

    def _connect(self, *, read_only: bool = False):
        return duckdb.connect(self.db_path, read_only=read_only)

    def _migrate(self) -> None:
        connection = self._connect()
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS invalidation_events (
                    event_id TEXT PRIMARY KEY,
                    dedupe_key TEXT UNIQUE,
                    position_plan_id TEXT,
                    position_plan_version_id TEXT,
                    symbol TEXT,
                    event_type TEXT,
                    source TEXT,
                    source_event_id TEXT,
                    rule_id TEXT,
                    as_of TIMESTAMPTZ,
                    observed_at TIMESTAMPTZ,
                    facts_json TEXT,
                    evidence_refs_json TEXT,
                    recorded_at TIMESTAMPTZ
                )
                """
            )
            connection.commit()
        finally:
            connection.close()

    def record(
        self,
        event: InvalidationEvent,
        *,
        plan: PositionPlan,
        received_at: datetime | None = None,
    ) -> bool:
        recorded_at = received_at or utc_now()
        self.validator.validate(
            event,
            plan=plan,
            received_at=recorded_at,
        )
        dedupe_key = self._dedupe_key(event)
        connection = self._connect()
        try:
            connection.execute("BEGIN TRANSACTION")
            existing = connection.execute(
                """
                SELECT event_id FROM invalidation_events
                WHERE dedupe_key=?
                """,
                [dedupe_key],
            ).fetchone()
            if existing:
                if str(existing[0]) != event.event_id:
                    raise ValueError("INVALIDATION_EVENT_SOURCE_CONFLICT")
                connection.rollback()
                return False
            connection.execute(
                """
                INSERT INTO invalidation_events VALUES
                (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                [
                    event.event_id,
                    dedupe_key,
                    event.position_plan_id,
                    event.position_plan_version_id,
                    event.symbol,
                    event.event_type.value,
                    event.source.value,
                    event.source_event_id,
                    event.rule_id,
                    event.as_of,
                    event.observed_at,
                    event.facts_json,
                    json.dumps(
                        list(event.evidence_refs),
                        separators=(",", ":"),
                    ),
                    recorded_at,
                ],
            )
            connection.commit()
            return True
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def list_for_plan(
        self,
        position_plan_id: str,
    ) -> list[InvalidationEvent]:
        connection = self._connect(read_only=True)
        try:
            rows = connection.execute(
                """
                SELECT event_id, position_plan_id,
                       position_plan_version_id, symbol, event_type,
                       source, source_event_id, rule_id, as_of,
                       observed_at, facts_json, evidence_refs_json
                FROM invalidation_events
                WHERE position_plan_id=?
                ORDER BY as_of, event_id
                """,
                [position_plan_id],
            ).fetchall()
        finally:
            connection.close()
        return [
            InvalidationEvent(
                event_id=str(row[0]),
                position_plan_id=str(row[1]),
                position_plan_version_id=str(row[2]),
                symbol=str(row[3]),
                event_type=InvalidationEventType(str(row[4])),
                source=InvalidationSource(str(row[5])),
                source_event_id=str(row[6]),
                rule_id=str(row[7]),
                as_of=row[8],
                observed_at=row[9],
                facts_json=str(row[10]),
                evidence_refs=tuple(json.loads(row[11] or "[]")),
            )
            for row in rows
        ]

    @staticmethod
    def _dedupe_key(event: InvalidationEvent) -> str:
        raw = "|".join(
            (
                event.position_plan_id,
                event.position_plan_version_id,
                event.event_type.value,
                event.source.value,
                event.source_event_id,
                event.rule_id,
            )
        )
        return hashlib.sha256(raw.encode()).hexdigest()
