from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from trader.audit import AuditLog
from trader.config import RiskConfig, TradingConfig
from trader.invalidation_events import (
    InvalidationEventStore,
    build_invalidation_event,
)
from trader.models import (
    Fill,
    InvalidationEventType,
    InvalidationSource,
    OrderIntent,
    PositionAdjustmentAction,
    PositionAdjustmentStatus,
    PositionPlan,
    PositionPlanStatus,
    Side,
    TradePlan,
)
from trader.order_lifecycle import (
    OrderIntentStore,
    OrderLifecycle,
    idempotency_key,
)
from trader.portfolio import Portfolio
from trader.position_adjustments import PositionAdjustmentStore
from trader.position_plans import PositionPlanFillProjector, PositionPlanStore
from trader.risk_engine import RiskEngine
from trader.runtime import Runtime

NOW = datetime(2026, 7, 27, 16, 0, tzinfo=timezone.utc)


def _config(path) -> TradingConfig:
    return TradingConfig(
        db_path=str(path),
        broker_type="alpaca_paper",
        auto_trade_paper=True,
        risk=RiskConfig(
            max_position_pct=0.20,
            max_trade_risk_pct=0.005,
        ),
    )


def _plan() -> PositionPlan:
    return PositionPlan(
        position_plan_id="position-plan-adjust",
        version_id="position-version-adjust-1",
        version=1,
        parent_version_id="",
        symbol="AAPL",
        side=Side.BUY,
        status=PositionPlanStatus.ACTIVE,
        source_trade_plan_id="trade-plan-entry",
        initial_fill_id="fill-entry",
        initial_entry_price=100,
        initial_quantity=10,
        open_quantity=10,
        average_entry_price=100,
        stop_loss=95,
        take_profit=115,
        invalidation_rules=(
            "PRICE_STOP",
            "STRATEGY_INVALIDATED",
        ),
        change_reason="INITIAL_FILL",
        created_at=NOW - timedelta(minutes=1),
    )


def _event(
    plan,
    *,
    event_type=InvalidationEventType.PRICE_STOP,
    source=InvalidationSource.MARKET_DATA,
    source_id="source-event-1",
    facts=None,
):
    return build_invalidation_event(
        plan=plan,
        event_type=event_type,
        source=source,
        source_event_id=source_id,
        as_of=NOW,
        observed_at=NOW,
        facts=facts
        or {
            "trigger_price": 94.9,
            "threshold_price": 95,
        },
        evidence_refs=("evidence-1",),
    )


def _persist_event(path, plan, event):
    event_store = InvalidationEventStore(path)
    event_store.record(event, plan=plan, received_at=NOW)


class _AuditCapture:
    def __init__(self):
        self.reports = []

    def log_reconciliation(self, report):
        self.reports.append(report)

    def log_plan_risk_event(self, *_args, **_kwargs):
        return None


def _runtime(config, broker):
    runtime = Runtime.__new__(Runtime)
    runtime._cfg = config
    runtime._broker = broker
    runtime._risk = RiskEngine(config)
    runtime._order_store = OrderIntentStore(config.db_path)
    runtime._position_plan_store = PositionPlanStore(config.db_path)
    runtime._position_plan_projector = PositionPlanFillProjector(
        runtime._position_plan_store
    )
    runtime._invalidation_event_store = InvalidationEventStore(
        config.db_path
    )
    runtime._position_adjustment_store = PositionAdjustmentStore(
        config.db_path
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
    runtime._bug_reporter = SimpleNamespace(
        capture_exception=lambda *_, **__: None
    )
    runtime._audit = _AuditCapture()
    return runtime


def _seed_open_position(path):
    config = _config(path)
    portfolio = Portfolio(config)
    portfolio.apply_fill(
        Fill(
            order_id="entry-order",
            intent_id="entry-intent",
            symbol="AAPL",
            side=Side.BUY,
            filled_qty=10,
            avg_price=100,
            fill_time=NOW - timedelta(minutes=1),
        )
    )
    plan = PositionPlanStore(path).create(_plan())
    return config, plan


def test_exit_event_atomically_versions_plan_and_prepares_order(tmp_path):
    path = tmp_path / "trade.duckdb"
    plan_store = PositionPlanStore(path)
    plan = plan_store.create(_plan())
    event = _event(plan)
    _persist_event(path, plan, event)
    store = PositionAdjustmentStore(path)

    adjustment, order_plan, created = store.prepare(
        event,
        plan=plan,
        limit_price=94.8,
    )

    assert created
    assert adjustment.action == PositionAdjustmentAction.EXIT
    assert adjustment.quantity == 10
    assert adjustment.status == PositionAdjustmentStatus.PLANNED
    assert order_plan is not None
    assert order_plan.action == "CLOSE"
    assert order_plan.side == Side.SELL
    assert order_plan.qty == 10
    current = plan_store.current(plan.position_plan_id)
    assert current is not None
    assert current.status == PositionPlanStatus.EXIT_PENDING
    assert current.version == 2

    duplicate, duplicate_order, created = store.prepare(
        event,
        plan=plan,
        limit_price=94.8,
    )
    assert not created
    assert duplicate == adjustment
    assert duplicate_order == order_plan
    assert len(plan_store.history(plan.position_plan_id)) == 2


def test_reduce_and_stop_tightening_are_deterministic(tmp_path):
    reduce_path = tmp_path / "reduce.duckdb"
    reduce_plan_store = PositionPlanStore(reduce_path)
    reduce_plan = reduce_plan_store.create(_plan())
    reduce_event = _event(
        reduce_plan,
        event_type=InvalidationEventType.STRATEGY_INVALIDATED,
        source=InvalidationSource.STRATEGY_ENGINE,
        source_id="strategy-reduce",
        facts={
            "evaluation_id": "evaluation-reduce",
            "valid": False,
            "requested_action": "REDUCE",
            "quantity": 3,
        },
    )
    _persist_event(reduce_path, reduce_plan, reduce_event)
    reduction, order_plan, _ = PositionAdjustmentStore(
        reduce_path
    ).prepare(
        reduce_event,
        plan=reduce_plan,
        limit_price=99,
    )
    assert reduction.action == PositionAdjustmentAction.REDUCE
    assert reduction.quantity == 3
    assert order_plan is not None
    assert order_plan.action == "REDUCE"

    stop_path = tmp_path / "stop.duckdb"
    stop_plan_store = PositionPlanStore(stop_path)
    stop_plan = stop_plan_store.create(_plan())
    stop_event = _event(
        stop_plan,
        event_type=InvalidationEventType.STRATEGY_INVALIDATED,
        source=InvalidationSource.STRATEGY_ENGINE,
        source_id="strategy-stop",
        facts={
            "evaluation_id": "evaluation-stop",
            "valid": False,
            "requested_action": "TIGHTEN_STOP",
            "new_stop_loss": 101,
        },
    )
    _persist_event(stop_path, stop_plan, stop_event)
    tightened, order_plan, _ = PositionAdjustmentStore(
        stop_path
    ).prepare(
        stop_event,
        plan=stop_plan,
        limit_price=None,
    )
    assert tightened.action == PositionAdjustmentAction.TIGHTEN_STOP
    assert tightened.new_stop_loss == 101
    assert order_plan is None
    assert (
        stop_plan_store.current(stop_plan.position_plan_id).stop_loss
        == 101
    )


def test_long_stop_cannot_be_loosened(tmp_path):
    path = tmp_path / "trade.duckdb"
    plan_store = PositionPlanStore(path)
    plan = plan_store.create(_plan())
    event = _event(
        plan,
        event_type=InvalidationEventType.STRATEGY_INVALIDATED,
        source=InvalidationSource.STRATEGY_ENGINE,
        source_id="strategy-loosen",
        facts={
            "evaluation_id": "evaluation-loosen",
            "valid": False,
            "requested_action": "TIGHTEN_STOP",
            "new_stop_loss": 94,
        },
    )
    _persist_event(path, plan, event)

    with pytest.raises(ValueError, match="LONG_STOP_MUST_TIGHTEN"):
        PositionAdjustmentStore(path).prepare(
            event,
            plan=plan,
            limit_price=None,
        )
    assert plan_store.current(plan.position_plan_id) == plan


def test_runtime_event_to_order_is_end_to_end_and_duplicate_safe(tmp_path):
    path = tmp_path / "trade.duckdb"
    config = _config(path)
    plan_store = PositionPlanStore(path)
    plan = plan_store.create(_plan())
    event = _event(plan)

    class Broker:
        calls = 0

        def place_order(self, _intent):
            self.calls += 1
            return "broker-adjustment-1"

    runtime = Runtime.__new__(Runtime)
    runtime._cfg = config
    runtime._broker = Broker()
    runtime._risk = RiskEngine(config)
    runtime._order_store = OrderIntentStore(str(path))
    runtime._position_plan_store = plan_store
    runtime._invalidation_event_store = InvalidationEventStore(path)
    runtime._position_adjustment_store = PositionAdjustmentStore(path)
    runtime._portfolio = Portfolio(config)
    runtime._kill = SimpleNamespace(engaged=lambda: False)
    runtime._reconciliation_blocked = False
    runtime._open_orders = {}
    runtime._live_plans = {}
    runtime._monitor_plans = {}
    runtime._signal_store = SimpleNamespace(mark_exit=lambda *_, **__: None)
    runtime._bug_reporter = SimpleNamespace(
        capture_exception=lambda *_, **__: None
    )

    first = runtime.process_invalidation_event(
        event,
        limit_price=94.8,
        received_at=NOW,
    )
    duplicate = runtime.process_invalidation_event(
        event,
        limit_price=94.8,
        received_at=NOW,
    )

    assert first == duplicate
    assert first.status == PositionAdjustmentStatus.ORDER_CREATED
    assert first.order_intent_id
    assert runtime._broker.calls == 1
    rows = runtime._order_store.list_all()
    assert len(rows) == 1
    assert rows[0]["order_type"] == "LMT"
    assert rows[0]["side"] == "SELL"
    assert len(plan_store.history(plan.position_plan_id)) == 2


class _StatefulBroker:
    def __init__(self):
        self.submit_calls = 0
        self.intent = None
        self.broker_id = "broker-adjustment-recovery"
        self.open = True
        self.position_qty = 10.0
        self.fill_qty = 0.0

    def place_order(self, intent):
        self.submit_calls += 1
        self.intent = intent
        return self.broker_id

    def get_open_orders(self):
        if not self.open or self.intent is None:
            return []
        return [
            {
                "id": self.broker_id,
                "client_order_id": self.intent.client_order_id,
            }
        ]

    def get_positions(self):
        if self.position_qty <= 0:
            return []
        return [
            SimpleNamespace(
                symbol="AAPL",
                qty=self.position_qty,
            )
        ]

    def get_recent_fills(self):
        if self.fill_qty <= 0:
            return []
        return [
            Fill(
                order_id=self.broker_id,
                intent_id="",
                symbol="AAPL",
                side=Side.SELL,
                filled_qty=self.fill_qty,
                avg_price=94.8,
                fill_time=NOW + timedelta(minutes=1),
            )
        ]


def test_restart_recovers_open_partial_and_completed_adjustment(tmp_path):
    path = tmp_path / "trade.duckdb"
    config, plan = _seed_open_position(path)
    broker = _StatefulBroker()
    runtime = _runtime(config, broker)
    event = _event(plan)
    adjustment = runtime.process_invalidation_event(
        event,
        limit_price=94.8,
        received_at=NOW,
    )

    restarted_open = _runtime(config, broker)
    restarted_open._run_reconciliation()
    assert not restarted_open._reconciliation_blocked
    assert broker.broker_id in restarted_open._open_orders
    assert broker.submit_calls == 1

    broker.fill_qty = 4
    broker.position_qty = 6
    restarted_partial = _runtime(config, broker)
    restarted_partial._run_reconciliation()
    assert not restarted_partial._reconciliation_blocked
    current = restarted_partial._position_plan_store.current(
        plan.position_plan_id
    )
    assert current is not None
    assert current.open_quantity == 6
    assert current.status == PositionPlanStatus.REDUCING
    row = restarted_partial._order_store.list_all()[0]
    assert row["filled_qty"] == 4
    assert row["remaining_qty"] == 6
    assert row["state"] == OrderLifecycle.PARTIALLY_FILLED.value

    repeated = _runtime(config, broker)
    repeated._run_reconciliation()
    assert not repeated._reconciliation_blocked
    assert len(
        repeated._position_plan_store.history(plan.position_plan_id)
    ) == 3

    broker.fill_qty = 10
    broker.position_qty = 0
    broker.open = False
    completed_runtime = _runtime(config, broker)
    completed_runtime._run_reconciliation()
    assert not completed_runtime._reconciliation_blocked
    completed_plan = completed_runtime._position_plan_store.current(
        plan.position_plan_id
    )
    assert completed_plan is not None
    assert completed_plan.status == PositionPlanStatus.CLOSED
    assert completed_plan.open_quantity == 0
    completed_adjustment = (
        completed_runtime._position_adjustment_store.get(
            adjustment.adjustment_id
        )
    )
    assert completed_adjustment is not None
    assert completed_adjustment.status == PositionAdjustmentStatus.COMPLETED


def test_restart_creates_missing_order_for_planned_adjustment(tmp_path):
    path = tmp_path / "trade.duckdb"
    config, plan = _seed_open_position(path)
    event = _event(plan)
    _persist_event(path, plan, event)
    adjustment, _, _ = PositionAdjustmentStore(path).prepare(
        event,
        plan=plan,
        limit_price=94.8,
    )
    assert adjustment.status == PositionAdjustmentStatus.PLANNED
    broker = _StatefulBroker()

    restarted = _runtime(config, broker)
    restarted._run_reconciliation()

    assert not restarted._reconciliation_blocked
    assert broker.submit_calls == 1
    recovered = restarted._position_adjustment_store.get(
        adjustment.adjustment_id
    )
    assert recovered is not None
    assert recovered.status == PositionAdjustmentStatus.ORDER_CREATED
    assert recovered.order_intent_id


def test_restart_never_guesses_or_resubmits_unknown_adjustment(tmp_path):
    path = tmp_path / "trade.duckdb"
    config, plan = _seed_open_position(path)

    class AmbiguousBroker(_StatefulBroker):
        def place_order(self, intent):
            self.submit_calls += 1
            self.intent = intent
            raise TimeoutError("response lost")

        def get_open_orders(self):
            return []

    initial_broker = AmbiguousBroker()
    runtime = _runtime(config, initial_broker)
    adjustment = runtime.process_invalidation_event(
        _event(plan),
        limit_price=94.8,
        received_at=NOW,
    )
    row = runtime._order_store.list_all()[0]
    assert row["state"] == OrderLifecycle.UNKNOWN.value
    assert adjustment.status == PositionAdjustmentStatus.ORDER_CREATED

    restarted_broker = AmbiguousBroker()
    restarted = _runtime(config, restarted_broker)
    restarted._run_reconciliation()

    assert restarted._reconciliation_blocked
    assert restarted_broker.submit_calls == 0
    assert restarted._audit.reports[-1].unexplained_orders


def test_restart_rebuilds_missing_initial_plan_from_audited_trade_plan(
    tmp_path,
):
    path = tmp_path / "trade.duckdb"
    config = _config(path)
    trade_plan = TradePlan(
        plan_id="audited-entry-plan",
        symbol="AAPL",
        side=Side.BUY,
        action="OPEN",
        entry_price=100,
        stop_loss=95,
        take_profit=115,
        qty=5,
        status="READY",
    )
    AuditLog(config).log_trade_plan(trade_plan)
    store = OrderIntentStore(str(path))
    key = idempotency_key(
        trade_plan.plan_id,
        trade_plan.symbol,
        trade_plan.side.value,
        trade_plan.qty,
        trade_plan.entry_price,
        trade_plan.action,
    )
    intent = OrderIntent(
        intent_id="audited-entry-intent",
        signal_id=trade_plan.plan_id,
        symbol="AAPL",
        side=Side.BUY,
        qty=5,
        order_type="LMT",
        limit_price=100,
        plan_id=trade_plan.plan_id,
    )
    store.persist(intent, key, trade_plan.plan_id)
    store.update(
        key,
        state=OrderLifecycle.OPEN.value,
        broker_order_id="audited-entry-order",
    )
    fill = Fill(
        order_id="audited-entry-order",
        intent_id="",
        symbol="AAPL",
        side=Side.BUY,
        filled_qty=5,
        avg_price=100,
        fill_time=NOW,
    )
    broker = SimpleNamespace(
        get_open_orders=lambda: [],
        get_positions=lambda: [
            SimpleNamespace(symbol="AAPL", qty=5)
        ],
        get_recent_fills=lambda: [fill],
    )

    restarted = _runtime(config, broker)
    restarted._run_reconciliation()

    assert not restarted._reconciliation_blocked
    recovered = restarted._position_plan_store.current_for_symbol("AAPL")
    assert recovered is not None
    assert recovered.source_trade_plan_id == trade_plan.plan_id
    assert recovered.open_quantity == 5

    repeated = _runtime(config, broker)
    repeated._run_reconciliation()
    assert not repeated._reconciliation_blocked
    assert len(
        repeated._position_plan_store.history(
            recovered.position_plan_id
        )
    ) == 1
