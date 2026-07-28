"""Deterministic, network-free Paper execution smoke harness."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from .audit import AuditLog
from .audit_query import order_traces
from .config import RiskConfig, TradingConfig
from .models import Fill, OrderStatus, Position, Side, TradePlan
from .order_lifecycle import OrderIntentStore, OrderLifecycle
from .portfolio import Portfolio
from .position_plans import PositionPlanFillProjector, PositionPlanStore
from .risk_engine import RiskEngine
from .runtime import Runtime


class IsolatedPaperBroker:
    """In-memory broker facts for smoke testing; it has no network client."""

    def __init__(self) -> None:
        self.submit_calls = 0
        self._orders: dict[str, dict] = {}
        self._positions: dict[str, Position] = {}
        self._recent_fills: dict[str, Fill] = {}

    def place_order(self, intent) -> str:
        self.submit_calls += 1
        broker_id = f"smoke-order-{self.submit_calls}"
        unknown = intent.plan_id.endswith("-unknown")
        if unknown:
            events: list[tuple[OrderStatus, float]] = []
        elif intent.side == Side.BUY:
            events = [
                (OrderStatus.PARTIAL, min(4.0, intent.qty)),
                (OrderStatus.FILLED, intent.qty),
            ]
        else:
            events = [(OrderStatus.FILLED, intent.qty)]
        self._orders[broker_id] = {
            "intent": intent,
            "events": events,
            "open": True,
            "last_fill_qty": 0.0,
            "applied_fill_qty": 0.0,
        }
        if unknown:
            raise TimeoutError("isolated response loss after acceptance")
        return broker_id

    def get_order_status(self, broker_order_id: str) -> OrderStatus:
        order = self._orders[broker_order_id]
        if not order["events"]:
            return OrderStatus.SUBMITTED
        status, cumulative_qty = order["events"].pop(0)
        order["last_fill_qty"] = cumulative_qty
        if status in {
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
        }:
            order["open"] = False
        return status

    def get_fill(self, broker_order_id: str) -> Fill | None:
        order = self._orders[broker_order_id]
        cumulative_qty = float(order["last_fill_qty"])
        if cumulative_qty <= 0:
            return None
        intent = order["intent"]
        fill = Fill(
            order_id=broker_order_id,
            intent_id=intent.intent_id,
            symbol=intent.symbol,
            side=intent.side,
            filled_qty=cumulative_qty,
            avg_price=float(intent.limit_price),
            fill_time=datetime.now(timezone.utc),
        )
        delta = cumulative_qty - float(order["applied_fill_qty"])
        if delta > 0:
            self._apply_position_delta(intent.symbol, intent.side, delta)
            order["applied_fill_qty"] = cumulative_qty
        self._recent_fills[broker_order_id] = fill
        return fill

    def _apply_position_delta(
        self,
        symbol: str,
        side: Side,
        qty: float,
    ) -> None:
        current = self._positions.get(
            symbol,
            Position(symbol, qty=0.0, avg_entry_px=100.0),
        )
        next_qty = current.qty + qty if side == Side.BUY else current.qty - qty
        if next_qty <= 1e-8:
            self._positions.pop(symbol, None)
        else:
            self._positions[symbol] = Position(
                symbol,
                qty=next_qty,
                avg_entry_px=100.0,
            )

    def get_open_orders(self) -> list[dict]:
        return [
            {
                "id": broker_id,
                "client_order_id": order["intent"].client_order_id,
                "status": "new",
            }
            for broker_id, order in self._orders.items()
            if order["open"]
        ]

    def get_positions(self) -> list[Position]:
        return list(self._positions.values())

    def get_recent_fills(self) -> list[Fill]:
        return list(self._recent_fills.values())

    def resolve_as_canceled(self, plan_id: str) -> str:
        for broker_id, order in self._orders.items():
            if order["intent"].plan_id == plan_id:
                order["events"] = [(OrderStatus.CANCELLED, 0.0)]
                return broker_id
        raise KeyError(plan_id)


def _runtime(
    config: TradingConfig,
    broker: IsolatedPaperBroker,
    audit: AuditLog,
) -> Runtime:
    runtime = Runtime.__new__(Runtime)
    runtime._cfg = config
    runtime._broker = broker
    runtime._audit = audit
    runtime._risk = RiskEngine(config)
    runtime._order_store = OrderIntentStore(config.db_path)
    runtime._position_plan_store = PositionPlanStore(config.db_path)
    runtime._position_plan_projector = PositionPlanFillProjector(
        runtime._position_plan_store
    )
    runtime._portfolio = Portfolio(config)
    runtime._kill = SimpleNamespace(engaged=lambda: False)
    runtime._reconciliation_blocked = False
    runtime._open_orders = {}
    runtime._live_plans = {}
    runtime._monitor_plans = {}
    runtime._signal_store = SimpleNamespace(
        mark_exit=lambda *_, **__: None,
        apply_fill=lambda *_, **__: None,
    )
    runtime._notifier = SimpleNamespace(send=lambda *_: True)
    runtime._bug_reporter = SimpleNamespace(
        capture_exception=lambda *_, **__: None
    )
    return runtime


def _plan(
    run_id: str,
    scenario: str,
    *,
    side: Side,
    action: str,
    qty: float,
) -> TradePlan:
    return TradePlan(
        plan_id=f"smoke-{run_id}-{scenario}",
        symbol="AAPL" if scenario != "unknown" else "MSFT",
        side=side,
        action=action,
        entry_price=100.0,
        stop_loss=99.0 if side == Side.BUY else 101.0,
        take_profit=102.0 if side == Side.BUY else 98.0,
        qty=qty,
        status="READY",
        rationale=f"isolated paper smoke: {scenario}",
        metadata={"decision_id": f"smoke-decision-{run_id}-{scenario}"},
    )


def _safe_smoke_path(db_path: str | Path) -> Path:
    path = Path(db_path).resolve()
    if "smoke" not in path.stem.lower():
        raise ValueError("SMOKE_DB_NAME_MUST_CONTAIN_SMOKE")
    if path.name.lower() in {"trade.duckdb", "ai_states.duckdb"}:
        raise ValueError("PROTECTED_DATABASE_PATH")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def run_smoke(db_path: str | Path) -> dict:
    """Run all isolated scenarios and return a machine-readable report."""
    path = _safe_smoke_path(db_path)
    run_id = uuid4().hex[:12]
    config = TradingConfig(
        db_path=str(path),
        broker_type="alpaca_paper",
        auto_trade_paper=True,
        risk=RiskConfig(
            max_position_pct=0.20,
            max_trade_risk_pct=0.005,
        ),
    )
    audit = AuditLog(config)
    broker = IsolatedPaperBroker()
    runtime = _runtime(config, broker, audit)
    plans = {
        "buy_partial": _plan(
            run_id,
            "buy-partial",
            side=Side.BUY,
            action="OPEN",
            qty=10.0,
        ),
        "sell": _plan(
            run_id,
            "sell",
            side=Side.SELL,
            action="CLOSE",
            qty=10.0,
        ),
        "rejected": _plan(
            run_id,
            "rejected",
            side=Side.BUY,
            action="OPEN",
            qty=600.0,
        ),
        "unknown": _plan(
            run_id,
            "unknown",
            side=Side.BUY,
            action="OPEN",
            qty=10.0,
        ),
    }

    for scenario in ("buy_partial", "sell", "rejected", "unknown"):
        plan = plans[scenario]
        audit.log_trade_plan(plan)
        runtime._execute_via_pipeline(
            plan,
            100_000.0,
            runtime._portfolio.positions,
        )
        if scenario == "buy_partial":
            runtime._poll_orders()
            runtime._poll_orders()
        elif scenario == "sell":
            runtime._poll_orders()

    unknown_plan = plans["unknown"]
    unknown_before = next(
        row
        for row in runtime._order_store.list_all()
        if row["plan_id"] == unknown_plan.plan_id
    )
    calls_before_restart = broker.submit_calls
    restarted = _runtime(config, broker, audit)
    restarted._run_reconciliation()
    if restarted._reconciliation_blocked:
        raise AssertionError("restart reconciliation unexpectedly blocked")
    restarted._execute_via_pipeline(
        unknown_plan,
        100_000.0,
        restarted._portfolio.positions,
    )
    if broker.submit_calls != calls_before_restart:
        raise AssertionError("UNKNOWN order was resubmitted after restart")
    unknown_broker_id = broker.resolve_as_canceled(unknown_plan.plan_id)
    restarted._poll_orders()

    rows = {
        row["plan_id"]: row
        for row in restarted._order_store.list_all()
        if row["plan_id"] in {plan.plan_id for plan in plans.values()}
    }
    expected_states = {
        plans["buy_partial"].plan_id: OrderLifecycle.FILLED.value,
        plans["sell"].plan_id: OrderLifecycle.FILLED.value,
        plans["unknown"].plan_id: OrderLifecycle.CANCELED.value,
    }
    for plan_id, expected in expected_states.items():
        if rows[plan_id]["state"] != expected:
            raise AssertionError(
                f"{plan_id} state={rows[plan_id]['state']} expected={expected}"
            )
    if plans["rejected"].plan_id in rows:
        raise AssertionError("risk-rejected plan created an order intent")
    if unknown_before["state"] != OrderLifecycle.UNKNOWN.value:
        raise AssertionError("response-loss scenario was not persisted UNKNOWN")
    if unknown_broker_id in restarted._open_orders:
        raise AssertionError("canceled UNKNOWN recovery order remained open")

    plan_ids = {plan.plan_id for plan in plans.values()}
    traces = order_traces(path, plan_ids=plan_ids)
    traces_by_plan = {trace["plan_id"]: trace for trace in traces}
    for plan in plans.values():
        trace = traces_by_plan[plan.plan_id]
        if trace["plan"] is None or not trace["risk_events"]:
            raise AssertionError(f"incomplete plan/risk trace: {plan.plan_id}")
        if trace["order"] is not None:
            order = trace["order"]
            if not order["idempotency_key"] or order["order_type"] != "LMT":
                raise AssertionError(
                    f"incomplete idempotency/LMT trace: {plan.plan_id}"
                )

    rejected_trace = traces_by_plan[plans["rejected"].plan_id]
    if not any(
        event["verdict"] == "BLOCKED"
        for event in rejected_trace["risk_events"]
    ):
        raise AssertionError("rejected plan lacks BLOCKED risk evidence")

    return {
        "ok": True,
        "network_used": False,
        "broker_type": config.broker_type,
        "db_path": str(path),
        "run_id": run_id,
        "scenarios": {
            "buy_partial": expected_states[plans["buy_partial"].plan_id],
            "sell": expected_states[plans["sell"].plan_id],
            "risk_rejected": "REJECTED_WITHOUT_ORDER",
            "unknown_before_restart": unknown_before["state"],
            "unknown_after_recovery": expected_states[unknown_plan.plan_id],
            "restart_resubmissions": broker.submit_calls - calls_before_restart,
        },
        "plan_ids": sorted(plan_ids),
        "trace_count": len(traces),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run BUY/SELL/reject/partial/UNKNOWN/restart against an "
            "in-memory broker with no network access."
        )
    )
    parser.add_argument(
        "--db",
        required=True,
        help="Isolated DuckDB path whose filename contains 'smoke'",
    )
    return parser.parse_args()


def main() -> None:
    report = run_smoke(_parse_args().db)
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
