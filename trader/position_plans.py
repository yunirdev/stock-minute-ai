"""Immutable PositionPlan version chains and compatible DuckDB storage."""
from __future__ import annotations

import json
import hashlib
from datetime import timezone
from pathlib import Path
from typing import Any

import duckdb

from .models import Fill, PositionPlan, PositionPlanStatus, Side, TradePlan


_ALLOWED_TRANSITIONS = {
    PositionPlanStatus.ACTIVE: {
        PositionPlanStatus.ACTIVE,
        PositionPlanStatus.REDUCING,
        PositionPlanStatus.EXIT_PENDING,
        PositionPlanStatus.CLOSED,
    },
    PositionPlanStatus.REDUCING: {
        PositionPlanStatus.ACTIVE,
        PositionPlanStatus.REDUCING,
        PositionPlanStatus.EXIT_PENDING,
        PositionPlanStatus.CLOSED,
    },
    PositionPlanStatus.EXIT_PENDING: {
        PositionPlanStatus.ACTIVE,
        PositionPlanStatus.REDUCING,
        PositionPlanStatus.EXIT_PENDING,
        PositionPlanStatus.CLOSED,
    },
    PositionPlanStatus.CLOSED: set(),
}


class PositionPlanStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._migrate()

    def _connect(self, *, read_only: bool = False):
        return duckdb.connect(self.db_path, read_only=read_only)

    def _migrate(self) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS position_plan_heads (
                    position_plan_id TEXT PRIMARY KEY,
                    symbol TEXT,
                    side TEXT,
                    source_trade_plan_id TEXT,
                    initial_fill_id TEXT,
                    initial_entry_price DOUBLE,
                    initial_quantity DOUBLE,
                    current_version INTEGER,
                    current_version_id TEXT,
                    status TEXT,
                    created_at TIMESTAMPTZ,
                    updated_at TIMESTAMPTZ
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS position_plan_fill_events (
                    fill_event_id TEXT PRIMARY KEY,
                    position_plan_id TEXT,
                    order_id TEXT,
                    intent_id TEXT,
                    cumulative_filled_qty DOUBLE,
                    applied_delta DOUBLE,
                    side TEXT,
                    avg_price DOUBLE,
                    version_id TEXT,
                    created_at TIMESTAMPTZ
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS position_plan_versions (
                    version_id TEXT PRIMARY KEY,
                    position_plan_id TEXT,
                    version INTEGER,
                    parent_version_id TEXT,
                    symbol TEXT,
                    side TEXT,
                    status TEXT,
                    source_trade_plan_id TEXT,
                    initial_fill_id TEXT,
                    initial_entry_price DOUBLE,
                    initial_quantity DOUBLE,
                    open_quantity DOUBLE,
                    average_entry_price DOUBLE,
                    stop_loss DOUBLE,
                    take_profit DOUBLE,
                    invalidation_rules_json TEXT,
                    change_reason TEXT,
                    created_at TIMESTAMPTZ,
                    UNIQUE(position_plan_id, version)
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def create(self, plan: PositionPlan) -> PositionPlan:
        if plan.version != 1 or plan.parent_version_id:
            raise ValueError("POSITION_PLAN_INITIAL_VERSION_REQUIRED")
        conn = self._connect()
        try:
            conn.execute("BEGIN TRANSACTION")
            if conn.execute(
                "SELECT 1 FROM position_plan_heads WHERE position_plan_id=?",
                [plan.position_plan_id],
            ).fetchone():
                raise ValueError("POSITION_PLAN_ALREADY_EXISTS")
            self._insert_version(conn, plan)
            conn.execute(
                """
                INSERT INTO position_plan_heads VALUES
                (?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                [
                    plan.position_plan_id,
                    plan.symbol,
                    plan.side.value,
                    plan.source_trade_plan_id,
                    plan.initial_fill_id,
                    plan.initial_entry_price,
                    plan.initial_quantity,
                    plan.version,
                    plan.version_id,
                    plan.status.value,
                    plan.created_at,
                    plan.created_at,
                ],
            )
            conn.commit()
            return plan
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def append(
        self,
        plan: PositionPlan,
        *,
        expected_version: int,
    ) -> PositionPlan:
        conn = self._connect()
        try:
            conn.execute("BEGIN TRANSACTION")
            head = conn.execute(
                """
                SELECT symbol, side, source_trade_plan_id, initial_fill_id,
                       initial_entry_price, initial_quantity, current_version,
                       current_version_id, status
                FROM position_plan_heads
                WHERE position_plan_id=?
                """,
                [plan.position_plan_id],
            ).fetchone()
            if head is None:
                raise KeyError(plan.position_plan_id)
            current_version = int(head[6])
            if current_version != expected_version:
                raise RuntimeError("POSITION_PLAN_VERSION_CONFLICT")
            if plan.version != current_version + 1:
                raise ValueError("POSITION_PLAN_VERSION_NOT_SEQUENTIAL")
            if plan.parent_version_id != str(head[7]):
                raise ValueError("POSITION_PLAN_PARENT_MISMATCH")
            immutable = (
                plan.symbol,
                plan.side.value,
                plan.source_trade_plan_id,
                plan.initial_fill_id,
                plan.initial_entry_price,
                plan.initial_quantity,
            )
            if immutable != head[:6]:
                raise ValueError("POSITION_PLAN_BASELINE_IMMUTABLE")
            previous_status = PositionPlanStatus(str(head[8]))
            if plan.status not in _ALLOWED_TRANSITIONS[previous_status]:
                raise ValueError("POSITION_PLAN_STATUS_TRANSITION_INVALID")
            self._insert_version(conn, plan)
            changed = conn.execute(
                """
                UPDATE position_plan_heads
                SET current_version=?, current_version_id=?, status=?,
                    updated_at=?
                WHERE position_plan_id=? AND current_version=?
                RETURNING position_plan_id
                """,
                [
                    plan.version,
                    plan.version_id,
                    plan.status.value,
                    plan.created_at,
                    plan.position_plan_id,
                    expected_version,
                ],
            ).fetchone()
            if changed is None:
                raise RuntimeError("POSITION_PLAN_VERSION_CONFLICT")
            conn.commit()
            return plan
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def current(self, position_plan_id: str) -> PositionPlan | None:
        conn = self._connect(read_only=True)
        try:
            cursor = conn.execute(
                """
                SELECT v.*
                FROM position_plan_heads h
                JOIN position_plan_versions v
                  ON v.version_id=h.current_version_id
                WHERE h.position_plan_id=?
                """,
                [position_plan_id],
            )
            columns = [item[0] for item in cursor.description]
            row = cursor.fetchone()
        finally:
            conn.close()
        return self._from_row(columns, row) if row else None

    def history(self, position_plan_id: str) -> list[PositionPlan]:
        conn = self._connect(read_only=True)
        try:
            cursor = conn.execute(
                """
                SELECT * FROM position_plan_versions
                WHERE position_plan_id=?
                ORDER BY version
                """,
                [position_plan_id],
            )
            columns = [item[0] for item in cursor.description]
            rows = cursor.fetchall()
        finally:
            conn.close()
        return [self._from_row(columns, row) for row in rows]

    def recover_open(self) -> list[PositionPlan]:
        conn = self._connect(read_only=True)
        try:
            ids = [
                row[0]
                for row in conn.execute(
                    """
                    SELECT position_plan_id FROM position_plan_heads
                    WHERE status<>'CLOSED'
                    ORDER BY symbol, position_plan_id
                    """
                ).fetchall()
            ]
        finally:
            conn.close()
        return [
            plan
            for plan_id in ids
            if (plan := self.current(str(plan_id))) is not None
        ]

    def current_for_symbol(self, symbol: str) -> PositionPlan | None:
        normalized = symbol.strip().upper()
        matches = [
            plan for plan in self.recover_open()
            if plan.symbol == normalized
        ]
        if len(matches) > 1:
            raise RuntimeError("POSITION_PLAN_SYMBOL_AMBIGUOUS")
        return matches[0] if matches else None

    def recover_trade_plan(self, plan_id: str) -> TradePlan | None:
        """Read an audited TradePlan needed to rebuild a missed first fill."""
        conn = self._connect(read_only=True)
        try:
            table = conn.execute(
                """
                SELECT 1 FROM information_schema.tables
                WHERE table_name='trade_plans'
                """
            ).fetchone()
            if table is None:
                return None
            row = conn.execute(
                """
                SELECT plan_id, symbol, side, action, entry_price,
                       stop_loss, take_profit, qty, confidence,
                       rationale, status, created_at
                FROM trade_plans
                WHERE plan_id=?
                """,
                [plan_id],
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        return TradePlan(
            plan_id=str(row[0]),
            symbol=str(row[1]),
            side=Side(str(row[2])),
            action=str(row[3]),
            entry_price=float(row[4]),
            stop_loss=float(row[5]),
            take_profit=float(row[6]),
            qty=float(row[7]),
            confidence=float(row[8]),
            rationale=str(row[9] or ""),
            status=str(row[10] or ""),
            created_at=row[11],
        )

    def fill_event_version(self, fill_event_id: str) -> str:
        conn = self._connect(read_only=True)
        try:
            row = conn.execute(
                """
                SELECT version_id FROM position_plan_fill_events
                WHERE fill_event_id=?
                """,
                [fill_event_id],
            ).fetchone()
        finally:
            conn.close()
        return str(row[0]) if row else ""

    def projected_cumulative_for_order(self, order_id: str) -> float:
        conn = self._connect(read_only=True)
        try:
            value = conn.execute(
                """
                SELECT coalesce(max(cumulative_filled_qty), 0)
                FROM position_plan_fill_events
                WHERE order_id=?
                """,
                [order_id],
            ).fetchone()[0]
        finally:
            conn.close()
        return float(value or 0.0)

    def record_fill_event(
        self,
        *,
        fill_event_id: str,
        plan: PositionPlan,
        fill: Fill,
        applied_delta: float,
    ) -> bool:
        conn = self._connect()
        try:
            if conn.execute(
                """
                SELECT 1 FROM position_plan_fill_events
                WHERE fill_event_id=?
                """,
                [fill_event_id],
            ).fetchone():
                return False
            conn.execute(
                """
                INSERT INTO position_plan_fill_events VALUES
                (?,?,?,?,?,?,?,?,?,?)
                """,
                [
                    fill_event_id,
                    plan.position_plan_id,
                    fill.order_id,
                    fill.intent_id,
                    fill.filled_qty,
                    applied_delta,
                    fill.side.value,
                    fill.avg_price,
                    plan.version_id,
                    fill.fill_time,
                ],
            )
            conn.commit()
            return True
        finally:
            conn.close()

    def commit_fill_projection(
        self,
        *,
        fill_event_id: str,
        plan: PositionPlan,
        fill: Fill,
        applied_delta: float,
        expected_version: int | None,
    ) -> bool:
        """Atomically advance a plan and its cumulative-fill cursor."""
        conn = self._connect()
        try:
            conn.execute("BEGIN TRANSACTION")
            if conn.execute(
                """
                SELECT 1 FROM position_plan_fill_events
                WHERE fill_event_id=?
                """,
                [fill_event_id],
            ).fetchone():
                conn.rollback()
                return False
            if expected_version is None:
                if plan.version != 1 or plan.parent_version_id:
                    raise ValueError("POSITION_PLAN_INITIAL_VERSION_REQUIRED")
                if conn.execute(
                    """
                    SELECT 1 FROM position_plan_heads
                    WHERE position_plan_id=?
                    """,
                    [plan.position_plan_id],
                ).fetchone():
                    raise ValueError("POSITION_PLAN_ALREADY_EXISTS")
                self._insert_version(conn, plan)
                conn.execute(
                    """
                    INSERT INTO position_plan_heads VALUES
                    (?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    [
                        plan.position_plan_id,
                        plan.symbol,
                        plan.side.value,
                        plan.source_trade_plan_id,
                        plan.initial_fill_id,
                        plan.initial_entry_price,
                        plan.initial_quantity,
                        plan.version,
                        plan.version_id,
                        plan.status.value,
                        plan.created_at,
                        plan.created_at,
                    ],
                )
            else:
                self._append_in_transaction(
                    conn,
                    plan,
                    expected_version=expected_version,
                )
            conn.execute(
                """
                INSERT INTO position_plan_fill_events VALUES
                (?,?,?,?,?,?,?,?,?,?)
                """,
                [
                    fill_event_id,
                    plan.position_plan_id,
                    fill.order_id,
                    fill.intent_id,
                    fill.filled_qty,
                    applied_delta,
                    fill.side.value,
                    fill.avg_price,
                    plan.version_id,
                    fill.fill_time,
                ],
            )
            conn.commit()
            return True
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _append_in_transaction(
        conn: duckdb.DuckDBPyConnection,
        plan: PositionPlan,
        *,
        expected_version: int,
    ) -> None:
        head = conn.execute(
            """
            SELECT symbol, side, source_trade_plan_id, initial_fill_id,
                   initial_entry_price, initial_quantity, current_version,
                   current_version_id, status
            FROM position_plan_heads
            WHERE position_plan_id=?
            """,
            [plan.position_plan_id],
        ).fetchone()
        if head is None:
            raise KeyError(plan.position_plan_id)
        current_version = int(head[6])
        if current_version != expected_version:
            raise RuntimeError("POSITION_PLAN_VERSION_CONFLICT")
        if plan.version != current_version + 1:
            raise ValueError("POSITION_PLAN_VERSION_NOT_SEQUENTIAL")
        if plan.parent_version_id != str(head[7]):
            raise ValueError("POSITION_PLAN_PARENT_MISMATCH")
        immutable = (
            plan.symbol,
            plan.side.value,
            plan.source_trade_plan_id,
            plan.initial_fill_id,
            plan.initial_entry_price,
            plan.initial_quantity,
        )
        if immutable != head[:6]:
            raise ValueError("POSITION_PLAN_BASELINE_IMMUTABLE")
        previous_status = PositionPlanStatus(str(head[8]))
        if plan.status not in _ALLOWED_TRANSITIONS[previous_status]:
            raise ValueError("POSITION_PLAN_STATUS_TRANSITION_INVALID")
        PositionPlanStore._insert_version(conn, plan)
        changed = conn.execute(
            """
            UPDATE position_plan_heads
            SET current_version=?, current_version_id=?, status=?,
                updated_at=?
            WHERE position_plan_id=? AND current_version=?
            RETURNING position_plan_id
            """,
            [
                plan.version,
                plan.version_id,
                plan.status.value,
                plan.created_at,
                plan.position_plan_id,
                expected_version,
            ],
        ).fetchone()
        if changed is None:
            raise RuntimeError("POSITION_PLAN_VERSION_CONFLICT")

    @staticmethod
    def _insert_version(
        conn: duckdb.DuckDBPyConnection,
        plan: PositionPlan,
    ) -> None:
        conn.execute(
            """
            INSERT INTO position_plan_versions VALUES
            (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            [
                plan.version_id,
                plan.position_plan_id,
                plan.version,
                plan.parent_version_id,
                plan.symbol,
                plan.side.value,
                plan.status.value,
                plan.source_trade_plan_id,
                plan.initial_fill_id,
                plan.initial_entry_price,
                plan.initial_quantity,
                plan.open_quantity,
                plan.average_entry_price,
                plan.stop_loss,
                plan.take_profit,
                json.dumps(
                    list(plan.invalidation_rules),
                    separators=(",", ":"),
                ),
                plan.change_reason,
                plan.created_at.astimezone(timezone.utc),
            ],
        )

    @staticmethod
    def _from_row(
        columns: list[str],
        row: tuple[Any, ...],
    ) -> PositionPlan:
        value = dict(zip(columns, row, strict=True))
        return PositionPlan(
            position_plan_id=str(value["position_plan_id"]),
            version_id=str(value["version_id"]),
            version=int(value["version"]),
            parent_version_id=str(value["parent_version_id"] or ""),
            symbol=str(value["symbol"]),
            side=Side(str(value["side"])),
            status=PositionPlanStatus(str(value["status"])),
            source_trade_plan_id=str(value["source_trade_plan_id"]),
            initial_fill_id=str(value["initial_fill_id"]),
            initial_entry_price=float(value["initial_entry_price"]),
            initial_quantity=float(value["initial_quantity"]),
            open_quantity=float(value["open_quantity"]),
            average_entry_price=float(value["average_entry_price"]),
            stop_loss=float(value["stop_loss"]),
            take_profit=float(value["take_profit"]),
            invalidation_rules=tuple(
                json.loads(value["invalidation_rules_json"] or "[]")
            ),
            change_reason=str(value["change_reason"]),
            created_at=value["created_at"],
        )


class PositionPlanFillProjector:
    """Project confirmed incremental fills into one durable plan chain."""

    def __init__(self, store: PositionPlanStore) -> None:
        self.store = store

    def apply(
        self,
        *,
        fill: Fill,
        applied_delta: float | None,
        trade_plan: TradePlan | None,
    ) -> PositionPlan | None:
        if applied_delta is None:
            applied_delta = max(
                float(fill.filled_qty)
                - self.store.projected_cumulative_for_order(fill.order_id),
                0.0,
            )
        if applied_delta <= 0:
            return self.store.current_for_symbol(fill.symbol)
        event_id = self._fill_event_id(fill)
        existing_version = self.store.fill_event_version(event_id)
        if existing_version:
            current = self.store.current_for_symbol(fill.symbol)
            if current is None:
                raise RuntimeError("POSITION_PLAN_FILL_EVENT_ORPHANED")
            return current
        # 多空对称：开新仓看 trade_plan.side（BUY=开多，SELL=开空）；已有仓位
        # 时看这笔成交跟现有仓位是同向还是反向——同向是加仓（多头再买/空头
        # 再卖），反向是减仓/平仓（多头卖出/空头买入回补）。原来这里硬编码
        # "fill.side==BUY 才能开新仓，否则必须已经有仓位"，SELL 在没有持仓
        # 时开空单会直接走进 else 分支报 POSITION_PLAN_OPEN_POSITION_REQUIRED
        # 崩掉。
        current = self.store.current_for_symbol(fill.symbol)
        if current is None:
            if trade_plan is None:
                raise RuntimeError("POSITION_PLAN_TRADE_PLAN_REQUIRED")
            if fill.side != trade_plan.side:
                raise ValueError("POSITION_PLAN_INITIAL_FILL_SIDE_INVALID")
            projected = self._initial(fill, applied_delta, trade_plan)
            expected_version = None
        elif fill.side == current.side:
            projected = self._increase(current, fill, applied_delta)
            expected_version = current.version
        else:
            projected = self._reduce(current, fill, applied_delta)
            expected_version = current.version
        self.store.commit_fill_projection(
            fill_event_id=event_id,
            plan=projected,
            fill=fill,
            applied_delta=applied_delta,
            expected_version=expected_version,
        )
        return projected

    @staticmethod
    def _fill_event_id(fill: Fill) -> str:
        raw = (
            f"{fill.order_id}|{fill.intent_id}|"
            f"{float(fill.filled_qty):.8f}|{fill.side.value}"
        )
        return "position-fill-" + hashlib.sha256(
            raw.encode()
        ).hexdigest()[:24]

    @staticmethod
    def _version_id(
        position_plan_id: str,
        version: int,
        fill: Fill,
    ) -> str:
        raw = (
            f"{position_plan_id}|{version}|{fill.order_id}|"
            f"{float(fill.filled_qty):.8f}|{fill.side.value}"
        )
        return "position-version-" + hashlib.sha256(
            raw.encode()
        ).hexdigest()[:24]

    def _initial(
        self,
        fill: Fill,
        delta: float,
        trade_plan: TradePlan,
    ) -> PositionPlan:
        if trade_plan.symbol.strip().upper() != fill.symbol.strip().upper():
            raise ValueError("POSITION_PLAN_FILL_SYMBOL_MISMATCH")
        # trade_plan.side==fill.side 已经在 apply() 里校验过了；这里不再要求
        # 必须是 BUY——SELL 开新仓（做空）走的是同一条路径，PositionPlan 的
        # side 字段（models.py 里 BUY/SELL 两个方向的价格顺序校验本来就都
        # 写好了，SHORT_POSITION_PLAN_PRICE_ORDER_INVALID 那支一直都在，只
        # 是从来没人真正创建过 side=SELL 的 PositionPlan）如实记录方向。
        raw = f"{trade_plan.plan_id}|{fill.symbol}|{fill.order_id}"
        plan_id = "position-plan-" + hashlib.sha256(
            raw.encode()
        ).hexdigest()[:24]
        return PositionPlan(
            position_plan_id=plan_id,
            version_id=self._version_id(plan_id, 1, fill),
            version=1,
            parent_version_id="",
            symbol=fill.symbol,
            side=trade_plan.side,
            status=PositionPlanStatus.ACTIVE,
            source_trade_plan_id=trade_plan.plan_id,
            initial_fill_id=self._fill_event_id(fill),
            initial_entry_price=fill.avg_price,
            initial_quantity=delta,
            open_quantity=delta,
            average_entry_price=fill.avg_price,
            stop_loss=trade_plan.stop_loss,
            take_profit=trade_plan.take_profit,
            invalidation_rules=tuple(
                trade_plan.metadata.get("invalidation_rules")
                or (
                    "PRICE_STOP",
                    "BROKER_RESTRICTION",
                    "CORPORATE_ACTION",
                    "TRADING_RESTRICTION",
                    # STRATEGY_INVALIDATED 是追踪止损（TrailingStopEvaluator）
                    # 收紧止损用的事件类型；不在这个允许列表里，
                    # InvalidationEventValidator 会用 INVALIDATION_RULE_NOT_CONFIGURED
                    # 拒掉每一次追踪止损尝试——加进默认规则集才能让它真正生效。
                    "STRATEGY_INVALIDATED",
                )
            ),
            change_reason="INITIAL_FILL",
            created_at=fill.fill_time,
        )

    def _increase(
        self,
        current: PositionPlan,
        fill: Fill,
        delta: float,
    ) -> PositionPlan:
        quantity = current.open_quantity + delta
        average = (
            current.average_entry_price * current.open_quantity
            + fill.avg_price * delta
        ) / quantity
        return PositionPlan(
            **{
                **current.__dict__,
                "version_id": self._version_id(
                    current.position_plan_id,
                    current.version + 1,
                    fill,
                ),
                "version": current.version + 1,
                "parent_version_id": current.version_id,
                "status": PositionPlanStatus.ACTIVE,
                "open_quantity": quantity,
                "average_entry_price": average,
                "change_reason": "ADDITIONAL_FILL",
                "created_at": fill.fill_time,
            }
        )

    def _reduce(
        self,
        current: PositionPlan,
        fill: Fill,
        delta: float,
    ) -> PositionPlan:
        if delta > current.open_quantity + 1e-8:
            raise ValueError("POSITION_PLAN_REDUCTION_EXCEEDS_OPEN_QUANTITY")
        quantity = max(current.open_quantity - delta, 0.0)
        return PositionPlan(
            **{
                **current.__dict__,
                "version_id": self._version_id(
                    current.position_plan_id,
                    current.version + 1,
                    fill,
                ),
                "version": current.version + 1,
                "parent_version_id": current.version_id,
                "status": (
                    PositionPlanStatus.CLOSED
                    if quantity <= 1e-8
                    else PositionPlanStatus.REDUCING
                ),
                "open_quantity": quantity,
                "change_reason": (
                    "CLOSE_FILL"
                    if quantity <= 1e-8
                    else "REDUCTION_FILL"
                ),
                "created_at": fill.fill_time,
            }
        )
