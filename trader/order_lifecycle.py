"""Durable order identity, lifecycle state, and startup reconciliation helpers."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Iterable

import duckdb


class OrderLifecycle(StrEnum):
    CREATED = "CREATED"
    RISK_APPROVED = "RISK_APPROVED"
    PERSISTED = "PERSISTED"
    SENDING = "SENDING"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    UNKNOWN = "UNKNOWN"
    OPEN = "OPEN"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


def idempotency_key(plan_id: str, symbol: str, side: str, qty: float, limit_price: float, action: str, leg: str = "entry") -> str:
    raw = "|".join((plan_id, symbol.upper(), side.upper(), f"{qty:.8f}", f"{limit_price:.8f}", action, leg))
    return "m2-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def client_order_id(key: str) -> str:
    return "m2-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]


class OrderIntentStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        con = duckdb.connect(self.db_path)
        con.execute("""
            CREATE TABLE IF NOT EXISTS order_intents (
                intent_id TEXT PRIMARY KEY, idempotency_key TEXT UNIQUE,
                decision_id TEXT, plan_id TEXT, symbol TEXT, side TEXT,
                qty DOUBLE, limit_price DOUBLE, order_type TEXT, tif TEXT,
                state TEXT, client_order_id TEXT, broker_order_id TEXT,
                submitted_at TIMESTAMPTZ, updated_at TIMESTAMPTZ,
                filled_qty DOUBLE DEFAULT 0, remaining_qty DOUBLE,
                last_error TEXT, retry_count INTEGER DEFAULT 0,
                schema_version INTEGER DEFAULT 1
            )
        """)
        for name, sql_type, default in (
            ("candidate_plan_id", "TEXT", "''"),
            ("final_plan_id", "TEXT", "''"),
            ("final_plan_version", "INTEGER", "0"),
            ("risk_check_id", "TEXT", "''"),
            ("evidence_refs_json", "TEXT", "'[]'"),
        ):
            con.execute(
                f"""
                ALTER TABLE order_intents
                ADD COLUMN IF NOT EXISTS {name} {sql_type}
                DEFAULT {default}
                """
            )
        con.commit()
        con.close()

    def get_by_key(self, key: str) -> dict[str, Any] | None:
        con = duckdb.connect(self.db_path, read_only=True)
        row = con.execute("SELECT * FROM order_intents WHERE idempotency_key=?", [key]).fetchone()
        cols = [x[0] for x in con.execute("DESCRIBE order_intents").fetchall()]
        con.close()
        return dict(zip(cols, row)) if row else None

    def persist(self, intent, key: str, plan_id: str, state: OrderLifecycle = OrderLifecycle.PERSISTED) -> dict[str, Any]:
        existing = self.get_by_key(key)
        if existing:
            return existing
        now = datetime.now(timezone.utc)
        cid = client_order_id(key)
        con = duckdb.connect(self.db_path)
        con.execute("""
            INSERT INTO order_intents
            (intent_id,idempotency_key,decision_id,plan_id,symbol,side,qty,
             limit_price,order_type,tif,state,client_order_id,remaining_qty,
             updated_at,candidate_plan_id,final_plan_id,final_plan_version,
             risk_check_id,evidence_refs_json)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT (idempotency_key) DO NOTHING
        """, [intent.intent_id, key, getattr(intent, "decision_id", "") or intent.signal_id, getattr(intent, "plan_id", "") or plan_id, intent.symbol, intent.side.value,
               intent.qty, intent.limit_price, intent.order_type, intent.tif, state.value, cid, intent.qty, now,
               getattr(intent, "candidate_plan_id", ""), getattr(intent, "final_plan_id", ""),
               getattr(intent, "final_plan_version", 0), getattr(intent, "risk_check_id", ""),
               json.dumps(list(getattr(intent, "evidence_refs", ()) or ()))])
        con.commit()
        con.close()
        return self.get_by_key(key) or {}

    def list_all(self) -> list[dict[str, Any]]:
        con = duckdb.connect(self.db_path, read_only=True)
        rows = con.execute("SELECT * FROM order_intents").fetchall()
        cols = [x[0] for x in con.execute("DESCRIBE order_intents").fetchall()]
        con.close()
        return [dict(zip(cols, row)) for row in rows]

    def pending_buy_notional_by_symbol(self) -> dict[str, float]:
        """Return durable unfilled BUY notional reserved per symbol.

        UNKNOWN and cancel-requested orders remain reserved until the broker
        confirms a terminal state. PARTIALLY_FILLED reserves only remaining_qty.
        """
        active_states = {
            OrderLifecycle.CREATED.value,
            OrderLifecycle.RISK_APPROVED.value,
            OrderLifecycle.PERSISTED.value,
            OrderLifecycle.SENDING.value,
            OrderLifecycle.ACKNOWLEDGED.value,
            OrderLifecycle.UNKNOWN.value,
            OrderLifecycle.OPEN.value,
            OrderLifecycle.PARTIALLY_FILLED.value,
            OrderLifecycle.CANCEL_REQUESTED.value,
        }
        exposure: dict[str, float] = {}
        for row in self.list_all():
            if (
                str(row.get("side", "")).upper() != "BUY"
                or row.get("state") not in active_states
            ):
                continue
            qty = float(row.get("qty") or 0.0)
            filled_qty = float(row.get("filled_qty") or 0.0)
            remaining_raw = row.get("remaining_qty")
            remaining_qty = (
                max(qty - filled_qty, 0.0)
                if remaining_raw is None
                else max(float(remaining_raw), 0.0)
            )
            price = float(row.get("limit_price") or 0.0)
            if not (
                math.isfinite(remaining_qty)
                and math.isfinite(price)
                and remaining_qty > 0
                and price > 0
            ):
                continue
            symbol = str(row.get("symbol", "")).strip().upper()
            if symbol:
                exposure[symbol] = (
                    exposure.get(symbol, 0.0) + remaining_qty * price
                )
        return exposure

    def update(self, key: str, **fields: Any) -> None:
        if not fields:
            return
        fields["updated_at"] = datetime.now(timezone.utc)
        con = duckdb.connect(self.db_path)
        assignments = ", ".join(f"{name}=?" for name in fields)
        con.execute(f"UPDATE order_intents SET {assignments} WHERE idempotency_key=?", [*fields.values(), key])
        con.commit()
        con.close()


@dataclass
class ReconciliationReport:
    ok: bool
    open_orders: int = 0
    positions: int = 0
    fills_seen: int = 0
    unexplained_orders: list[str] = field(default_factory=list)
    unexplained_positions: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def reconcile_broker(broker, local_intents: Iterable[dict[str, Any]], local_positions: Iterable[Any] = ()) -> ReconciliationReport:
    """Compare broker facts with durable local intents; never mutates broker state."""
    try:
        broker_orders = list(getattr(broker, "get_open_orders", lambda: [])())
        positions = list(broker.get_positions())
        fills = list(getattr(broker, "get_recent_fills", lambda: [])())
    except Exception as exc:
        return ReconciliationReport(ok=False, errors=[type(exc).__name__])
    return reconcile_broker_facts(
        broker_orders,
        positions,
        fills,
        local_intents,
        local_positions,
    )


def reconcile_broker_facts(
    broker_orders: Iterable[Any],
    broker_positions: Iterable[Any],
    broker_fills: Iterable[Any],
    local_intents: Iterable[dict[str, Any]],
    local_positions: Iterable[Any] = (),
) -> ReconciliationReport:
    """Compare one already-fetched broker snapshot with durable local state."""
    broker_orders = list(broker_orders)
    positions = list(broker_positions)
    fills = list(broker_fills)
    report = ReconciliationReport(
        ok=True,
        open_orders=len(broker_orders),
        positions=len(positions),
        fills_seen=len(fills),
    )
    local_rows = list(local_intents)
    active_states = {
        OrderLifecycle.PERSISTED.value,
        OrderLifecycle.SENDING.value,
        OrderLifecycle.ACKNOWLEDGED.value,
        OrderLifecycle.UNKNOWN.value,
        OrderLifecycle.OPEN.value,
        OrderLifecycle.PARTIALLY_FILLED.value,
        OrderLifecycle.CANCEL_REQUESTED.value,
    }
    active_rows = [
        row
        for row in local_rows
        if not row.get("state") or row.get("state") in active_states
    ]
    known_clients = {
        str(row["client_order_id"])
        for row in active_rows
        if row.get("client_order_id")
    }
    known_broker_ids = {
        str(row["broker_order_id"])
        for row in active_rows
        if row.get("broker_order_id")
    }
    broker_client_ids: set[str] = set()
    broker_order_ids: set[str] = set()
    for order in broker_orders:
        cid = getattr(order, "client_order_id", None) or (
            order.get("client_order_id") if isinstance(order, dict) else None
        )
        broker_id = getattr(order, "id", None) or (
            order.get("id") if isinstance(order, dict) else None
        )
        if cid:
            broker_client_ids.add(str(cid))
        if broker_id:
            broker_order_ids.add(str(broker_id))
        if (
            (not cid or str(cid) not in known_clients)
            and (not broker_id or str(broker_id) not in known_broker_ids)
        ):
            report.unexplained_orders.append(str(cid or broker_id or "unknown"))
    for row in active_rows:
        client_match = (
            row.get("client_order_id")
            and str(row["client_order_id"]) in broker_client_ids
        )
        broker_match = (
            row.get("broker_order_id")
            and str(row["broker_order_id"]) in broker_order_ids
        )
        if not client_match and not broker_match:
            report.unexplained_orders.append(
                str(
                    row.get("client_order_id")
                    or row.get("broker_order_id")
                    or row.get("intent_id")
                )
            )

    def _position_map(items: Iterable[Any]) -> dict[str, float]:
        result: dict[str, float] = {}
        for position in items:
            symbol = getattr(position, "symbol", None) or (
                position.get("symbol") if isinstance(position, dict) else None
            )
            qty = getattr(position, "qty", None)
            if qty is None and isinstance(position, dict):
                qty = position.get("qty")
            if symbol:
                result[str(symbol).upper()] = float(qty or 0.0)
        return result

    broker_position_map = _position_map(positions)
    local_position_map = _position_map(local_positions)
    for symbol in sorted(set(broker_position_map) | set(local_position_map)):
        if symbol not in broker_position_map or symbol not in local_position_map:
            report.unexplained_positions.append(symbol)
            continue
        broker_qty = broker_position_map[symbol]
        local_qty = local_position_map[symbol]
        if abs(broker_qty - local_qty) > 1e-8:
            report.unexplained_positions.append(
                f"{symbol}:broker={broker_qty:g},local={local_qty:g}"
            )
    report.unexplained_orders = sorted(set(report.unexplained_orders))
    report.unexplained_positions = sorted(set(report.unexplained_positions))
    report.ok = not report.unexplained_orders and not report.unexplained_positions and not report.errors
    return report
