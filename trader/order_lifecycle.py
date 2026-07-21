"""Durable order identity, lifecycle state, and startup reconciliation helpers."""
from __future__ import annotations

import hashlib
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
            (intent_id,idempotency_key,decision_id,plan_id,symbol,side,qty,limit_price,order_type,tif,state,client_order_id,remaining_qty,updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT (idempotency_key) DO NOTHING
        """, [intent.intent_id, key, getattr(intent, "decision_id", "") or intent.signal_id, getattr(intent, "plan_id", "") or plan_id, intent.symbol, intent.side.value,
               intent.qty, intent.limit_price, intent.order_type, intent.tif, state.value, cid, intent.qty, now])
        con.commit()
        con.close()
        return self.get_by_key(key) or {}

    def list_all(self) -> list[dict[str, Any]]:
        con = duckdb.connect(self.db_path, read_only=True)
        rows = con.execute("SELECT * FROM order_intents").fetchall()
        cols = [x[0] for x in con.execute("DESCRIBE order_intents").fetchall()]
        con.close()
        return [dict(zip(cols, row)) for row in rows]

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
    report = ReconciliationReport(ok=True)
    try:
        broker_orders = list(getattr(broker, "get_open_orders", lambda: [])())
        positions = list(broker.get_positions())
        fills = list(getattr(broker, "get_recent_fills", lambda: [])())
    except Exception as exc:
        report.ok = False
        report.errors.append(type(exc).__name__)
        return report
    report.open_orders, report.positions, report.fills_seen = len(broker_orders), len(positions), len(fills)
    local_rows = list(local_intents)
    known_clients = {row.get("client_order_id") for row in local_rows}
    broker_ids = set()
    for order in broker_orders:
        cid = getattr(order, "client_order_id", None) or (order.get("client_order_id") if isinstance(order, dict) else None)
        broker_ids.add(cid)
        if cid and cid not in known_clients:
            report.unexplained_orders.append(str(cid))
    for row in local_rows:
        if row.get("state") in {OrderLifecycle.OPEN.value, OrderLifecycle.SENDING.value, OrderLifecycle.PARTIALLY_FILLED.value} and row.get("client_order_id") not in broker_ids:
            report.unexplained_orders.append(str(row.get("client_order_id")))
    broker_symbols = {getattr(position, "symbol", position.get("symbol") if isinstance(position, dict) else "") for position in positions}
    local_symbols = {getattr(position, "symbol", position.get("symbol") if isinstance(position, dict) else "") for position in local_positions}
    report.unexplained_positions.extend(sorted(broker_symbols - local_symbols))
    report.ok = not report.unexplained_orders and not report.unexplained_positions and not report.errors
    return report




def resolve_unknown(store: OrderIntentStore, key: str, broker_order_exists: bool) -> bool:
    """Resolve UNKNOWN only after an explicit broker lookup; return whether retry is allowed."""
    row = store.get_by_key(key)
    if not row or row.get("state") != OrderLifecycle.UNKNOWN.value:
        return False
    if broker_order_exists:
        store.update(key, state=OrderLifecycle.OPEN.value)
        return False
    store.update(key, state=OrderLifecycle.PERSISTED.value, retry_count=int(row.get("retry_count") or 0) + 1)
    return True
