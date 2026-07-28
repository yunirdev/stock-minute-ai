"""Deterministic PositionPlan adjustments derived from invalidation facts."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import duckdb

from .models import (
    InvalidationEvent,
    InvalidationEventType,
    PositionAdjustment,
    PositionAdjustmentAction,
    PositionAdjustmentStatus,
    PositionPlan,
    PositionPlanStatus,
    Side,
    TradePlan,
)
from .position_plans import PositionPlanStore


class PositionAdjustmentEvaluator:
    def evaluate(
        self,
        event: InvalidationEvent,
        *,
        plan: PositionPlan,
        limit_price: float | None,
    ) -> tuple[PositionAdjustment, PositionPlan, TradePlan | None]:
        facts = json.loads(event.facts_json)
        action = self._action(event, facts)
        quantity = self._quantity(action, facts, plan)
        price = self._limit_price(action, limit_price)
        new_stop = self._new_stop(action, facts, plan)
        new_status = {
            PositionAdjustmentAction.EXIT: PositionPlanStatus.EXIT_PENDING,
            PositionAdjustmentAction.REDUCE: PositionPlanStatus.REDUCING,
            PositionAdjustmentAction.TIGHTEN_STOP: PositionPlanStatus.ACTIVE,
        }[action]
        adjustment_id = self._stable_id(
            "adjustment",
            event.event_id,
            plan.version_id,
            action.value,
        )
        version = plan.version + 1
        version_id = self._stable_id(
            "position-version",
            plan.position_plan_id,
            str(version),
            event.event_id,
        )
        order_plan_id = (
            self._stable_id(
                "adjustment-order",
                adjustment_id,
                version_id,
            )
            if action != PositionAdjustmentAction.TIGHTEN_STOP
            else ""
        )
        next_plan = PositionPlan(
            **{
                **plan.__dict__,
                "version_id": version_id,
                "version": version,
                "parent_version_id": plan.version_id,
                "status": new_status,
                "stop_loss": new_stop,
                "change_reason": f"INVALIDATION:{event.event_id}",
                "created_at": event.observed_at,
            }
        )
        adjustment = PositionAdjustment(
            adjustment_id=adjustment_id,
            event_id=event.event_id,
            position_plan_id=plan.position_plan_id,
            from_version_id=plan.version_id,
            to_version_id=version_id,
            action=action,
            status=(
                PositionAdjustmentStatus.COMPLETED
                if action == PositionAdjustmentAction.TIGHTEN_STOP
                else PositionAdjustmentStatus.PLANNED
            ),
            quantity=quantity,
            limit_price=price,
            previous_stop_loss=plan.stop_loss,
            new_stop_loss=new_stop,
            order_plan_id=order_plan_id,
            created_at=event.observed_at,
        )
        order_plan = (
            self._order_plan(adjustment, next_plan)
            if order_plan_id
            else None
        )
        return adjustment, next_plan, order_plan

    @staticmethod
    def _action(
        event: InvalidationEvent,
        facts: dict,
    ) -> PositionAdjustmentAction:
        if event.event_type != InvalidationEventType.STRATEGY_INVALIDATED:
            return PositionAdjustmentAction.EXIT
        requested = str(facts.get("requested_action", "EXIT")).upper()
        try:
            return PositionAdjustmentAction(requested)
        except ValueError as exc:
            raise ValueError("POSITION_ADJUSTMENT_ACTION_INVALID") from exc

    @staticmethod
    def _quantity(
        action: PositionAdjustmentAction,
        facts: dict,
        plan: PositionPlan,
    ) -> float:
        if action == PositionAdjustmentAction.TIGHTEN_STOP:
            return 0.0
        if action == PositionAdjustmentAction.EXIT:
            return plan.open_quantity
        try:
            quantity = float(facts["quantity"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("POSITION_ADJUSTMENT_QUANTITY_INVALID") from exc
        if (
            not math.isfinite(quantity)
            or quantity <= 0
            or quantity >= plan.open_quantity
        ):
            raise ValueError("POSITION_ADJUSTMENT_QUANTITY_INVALID")
        return quantity

    @staticmethod
    def _limit_price(
        action: PositionAdjustmentAction,
        value: float | None,
    ) -> float:
        if action == PositionAdjustmentAction.TIGHTEN_STOP:
            return 0.0
        try:
            price = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("POSITION_ADJUSTMENT_LIMIT_PRICE_INVALID") from exc
        if not math.isfinite(price) or price <= 0:
            raise ValueError("POSITION_ADJUSTMENT_LIMIT_PRICE_INVALID")
        return price

    @staticmethod
    def _new_stop(
        action: PositionAdjustmentAction,
        facts: dict,
        plan: PositionPlan,
    ) -> float:
        if action != PositionAdjustmentAction.TIGHTEN_STOP:
            return plan.stop_loss
        try:
            new_stop = float(facts["new_stop_loss"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("POSITION_ADJUSTMENT_STOP_INVALID") from exc
        if not math.isfinite(new_stop) or new_stop <= 0:
            raise ValueError("POSITION_ADJUSTMENT_STOP_INVALID")
        if plan.side == Side.BUY and not (
            plan.stop_loss < new_stop < plan.take_profit
        ):
            raise ValueError("LONG_STOP_MUST_TIGHTEN")
        if plan.side == Side.SELL and not (
            plan.take_profit < new_stop < plan.stop_loss
        ):
            raise ValueError("SHORT_STOP_MUST_TIGHTEN")
        return new_stop

    @staticmethod
    def _order_plan(
        adjustment: PositionAdjustment,
        plan: PositionPlan,
    ) -> TradePlan:
        side = Side.SELL if plan.side == Side.BUY else Side.BUY
        return TradePlan(
            plan_id=adjustment.order_plan_id,
            symbol=plan.symbol,
            side=side,
            action=(
                "CLOSE"
                if adjustment.action == PositionAdjustmentAction.EXIT
                else "REDUCE"
            ),
            entry_price=adjustment.limit_price,
            stop_loss=plan.stop_loss,
            take_profit=plan.take_profit,
            qty=adjustment.quantity,
            confidence=1.0,
            rationale=f"validated invalidation event {adjustment.event_id}",
            source="invalidation",
            status="READY",
            created_at=adjustment.created_at,
            metadata={
                "invalidation_event_id": adjustment.event_id,
                "position_plan_id": adjustment.position_plan_id,
                "position_plan_version_id": adjustment.to_version_id,
                "adjustment_id": adjustment.adjustment_id,
            },
        )

    @staticmethod
    def _stable_id(prefix: str, *parts: str) -> str:
        raw = "|".join(parts)
        return prefix + "-" + hashlib.sha256(raw.encode()).hexdigest()[:24]


class PositionAdjustmentStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self.evaluator = PositionAdjustmentEvaluator()
        self._migrate()

    def _connect(self, *, read_only: bool = False):
        return duckdb.connect(self.db_path, read_only=read_only)

    def _migrate(self) -> None:
        connection = self._connect()
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS position_adjustments (
                    adjustment_id TEXT PRIMARY KEY,
                    event_id TEXT UNIQUE,
                    position_plan_id TEXT,
                    from_version_id TEXT,
                    to_version_id TEXT,
                    action TEXT,
                    status TEXT,
                    quantity DOUBLE,
                    limit_price DOUBLE,
                    previous_stop_loss DOUBLE,
                    new_stop_loss DOUBLE,
                    order_plan_id TEXT UNIQUE,
                    order_intent_id TEXT,
                    order_idempotency_key TEXT,
                    created_at TIMESTAMPTZ,
                    updated_at TIMESTAMPTZ
                )
                """
            )
            connection.commit()
        finally:
            connection.close()

    def prepare(
        self,
        event: InvalidationEvent,
        *,
        plan: PositionPlan,
        limit_price: float | None,
    ) -> tuple[PositionAdjustment, TradePlan | None, bool]:
        adjustment, next_plan, order_plan = self.evaluator.evaluate(
            event,
            plan=plan,
            limit_price=limit_price,
        )
        connection = self._connect()
        try:
            connection.execute("BEGIN TRANSACTION")
            existing = self._get_by_event(connection, event.event_id)
            if existing is not None:
                connection.rollback()
                return existing, order_plan, False
            persisted = connection.execute(
                """
                SELECT 1 FROM invalidation_events WHERE event_id=?
                """,
                [event.event_id],
            ).fetchone()
            if persisted is None:
                raise ValueError("POSITION_ADJUSTMENT_EVENT_NOT_PERSISTED")
            PositionPlanStore._append_in_transaction(
                connection,
                next_plan,
                expected_version=plan.version,
            )
            connection.execute(
                """
                INSERT INTO position_adjustments VALUES
                (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                [
                    adjustment.adjustment_id,
                    adjustment.event_id,
                    adjustment.position_plan_id,
                    adjustment.from_version_id,
                    adjustment.to_version_id,
                    adjustment.action.value,
                    adjustment.status.value,
                    adjustment.quantity,
                    adjustment.limit_price,
                    adjustment.previous_stop_loss,
                    adjustment.new_stop_loss,
                    adjustment.order_plan_id or None,
                    adjustment.order_intent_id,
                    adjustment.order_idempotency_key,
                    adjustment.created_at,
                    adjustment.created_at,
                ],
            )
            connection.commit()
            return adjustment, order_plan, True
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def link_order(
        self,
        adjustment_id: str,
        *,
        order_intent_id: str,
        order_idempotency_key: str,
    ) -> PositionAdjustment:
        connection = self._connect()
        try:
            connection.execute("BEGIN TRANSACTION")
            row = connection.execute(
                """
                SELECT order_intent_id, order_idempotency_key
                FROM position_adjustments
                WHERE adjustment_id=?
                """,
                [adjustment_id],
            ).fetchone()
            if row is None:
                raise KeyError(adjustment_id)
            if row[0] or row[1]:
                if (
                    str(row[0]) != order_intent_id
                    or str(row[1]) != order_idempotency_key
                ):
                    raise ValueError("POSITION_ADJUSTMENT_ORDER_CONFLICT")
            else:
                connection.execute(
                    """
                    UPDATE position_adjustments
                    SET status=?, order_intent_id=?,
                        order_idempotency_key=?, updated_at=now()
                    WHERE adjustment_id=?
                    """,
                    [
                        PositionAdjustmentStatus.ORDER_CREATED.value,
                        order_intent_id,
                        order_idempotency_key,
                        adjustment_id,
                    ],
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        linked = self.get(adjustment_id)
        if linked is None:
            raise RuntimeError("POSITION_ADJUSTMENT_LINK_LOST")
        return linked

    def get(self, adjustment_id: str) -> PositionAdjustment | None:
        connection = self._connect(read_only=True)
        try:
            return self._get_by_id(connection, adjustment_id)
        finally:
            connection.close()

    def get_by_event(self, event_id: str) -> PositionAdjustment | None:
        connection = self._connect(read_only=True)
        try:
            return self._get_by_event(connection, event_id)
        finally:
            connection.close()

    def list_incomplete(self) -> list[PositionAdjustment]:
        connection = self._connect(read_only=True)
        try:
            rows = connection.execute(
                """
                SELECT * FROM position_adjustments
                WHERE status<>?
                ORDER BY created_at, adjustment_id
                """,
                [PositionAdjustmentStatus.COMPLETED.value],
            ).fetchall()
        finally:
            connection.close()
        return [self._from_row(row) for row in rows]

    def order_plan_for(
        self,
        adjustment: PositionAdjustment,
        *,
        plan: PositionPlan,
    ) -> TradePlan:
        if adjustment.action == PositionAdjustmentAction.TIGHTEN_STOP:
            raise ValueError("STOP_ADJUSTMENT_HAS_NO_ORDER")
        if (
            plan.position_plan_id != adjustment.position_plan_id
            or plan.version_id != adjustment.to_version_id
        ):
            raise ValueError("POSITION_ADJUSTMENT_PLAN_VERSION_MISMATCH")
        return self.evaluator._order_plan(adjustment, plan)

    def mark_completed_by_order_plan(
        self,
        order_plan_id: str,
    ) -> PositionAdjustment | None:
        connection = self._connect()
        try:
            connection.execute("BEGIN TRANSACTION")
            row = connection.execute(
                """
                SELECT adjustment_id FROM position_adjustments
                WHERE order_plan_id=?
                """,
                [order_plan_id],
            ).fetchone()
            if row is None:
                connection.rollback()
                return None
            connection.execute(
                """
                UPDATE position_adjustments
                SET status=?, updated_at=now()
                WHERE adjustment_id=?
                """,
                [
                    PositionAdjustmentStatus.COMPLETED.value,
                    str(row[0]),
                ],
            )
            connection.commit()
            adjustment_id = str(row[0])
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        completed = self.get(adjustment_id)
        if completed is None:
            raise RuntimeError("POSITION_ADJUSTMENT_COMPLETION_LOST")
        return completed

    @classmethod
    def _get_by_id(
        cls,
        connection: duckdb.DuckDBPyConnection,
        adjustment_id: str,
    ) -> PositionAdjustment | None:
        row = connection.execute(
            """
            SELECT * FROM position_adjustments WHERE adjustment_id=?
            """,
            [adjustment_id],
        ).fetchone()
        return cls._from_row(row) if row else None

    @classmethod
    def _get_by_event(
        cls,
        connection: duckdb.DuckDBPyConnection,
        event_id: str,
    ) -> PositionAdjustment | None:
        row = connection.execute(
            """
            SELECT * FROM position_adjustments WHERE event_id=?
            """,
            [event_id],
        ).fetchone()
        return cls._from_row(row) if row else None

    @staticmethod
    def _from_row(row: tuple) -> PositionAdjustment:
        return PositionAdjustment(
            adjustment_id=str(row[0]),
            event_id=str(row[1]),
            position_plan_id=str(row[2]),
            from_version_id=str(row[3]),
            to_version_id=str(row[4]),
            action=PositionAdjustmentAction(str(row[5])),
            status=PositionAdjustmentStatus(str(row[6])),
            quantity=float(row[7]),
            limit_price=float(row[8]),
            previous_stop_loss=float(row[9]),
            new_stop_loss=float(row[10]),
            order_plan_id=str(row[11] or ""),
            order_intent_id=str(row[12] or ""),
            order_idempotency_key=str(row[13] or ""),
            created_at=row[14],
        )
