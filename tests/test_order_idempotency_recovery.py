from datetime import datetime, timezone
from types import SimpleNamespace

from trader.config import RiskConfig, TradingConfig
from trader.models import Fill, OrderIntent, OrderStatus, Side, TradePlan
from trader.order_lifecycle import (
    OrderIntentStore,
    OrderLifecycle,
    client_order_id,
    idempotency_key,
)
from trader.portfolio import Portfolio
from trader.position_plans import PositionPlanFillProjector, PositionPlanStore
from trader.risk_engine import RiskEngine
from trader.runtime import Runtime


def _config(db_path: str) -> TradingConfig:
    return TradingConfig(
        db_path=db_path,
        broker_type="alpaca_paper",
        auto_trade_paper=True,
        risk=RiskConfig(
            max_position_pct=0.20,
            max_trade_risk_pct=0.005,
        ),
    )


def _plan() -> TradePlan:
    return TradePlan(
        plan_id="stable-plan",
        symbol="AAPL",
        side=Side.BUY,
        action="OPEN",
        entry_price=100.0,
        stop_loss=99.0,
        take_profit=102.0,
        qty=10.0,
    )


def _runtime_for_submit(config, store, broker) -> Runtime:
    runtime = Runtime.__new__(Runtime)
    runtime._cfg = config
    runtime._risk = RiskEngine(config)
    runtime._order_store = store
    runtime._broker = broker
    runtime._kill = SimpleNamespace(engaged=lambda: False)
    runtime._reconciliation_blocked = False
    runtime._open_orders = {}
    runtime._live_plans = {}
    runtime._bug_reporter = SimpleNamespace(capture_exception=lambda *_, **__: None)
    return runtime


def test_unknown_submission_is_durable_and_never_retried_after_restart(tmp_path):
    db_path = str(tmp_path / "trade.duckdb")
    config = _config(db_path)
    store = OrderIntentStore(db_path)

    class AmbiguousBroker:
        calls = 0

        def place_order(self, _intent):
            self.calls += 1
            raise TimeoutError("response lost after possible acceptance")

    broker = AmbiguousBroker()
    runtime = _runtime_for_submit(config, store, broker)
    runtime._execute_via_pipeline(_plan(), 100_000.0, {})
    runtime._execute_via_pipeline(_plan(), 100_000.0, {})

    restarted = _runtime_for_submit(
        config,
        OrderIntentStore(db_path),
        broker,
    )
    restarted._execute_via_pipeline(_plan(), 100_000.0, {})

    row = store.list_all()[0]
    assert broker.calls == 1
    assert row["state"] == OrderLifecycle.UNKNOWN.value
    assert row["broker_order_id"] is None
    assert row["last_error"] == "TimeoutError"


def test_successful_duplicate_plan_submits_once(tmp_path):
    db_path = str(tmp_path / "trade.duckdb")
    config = _config(db_path)
    store = OrderIntentStore(db_path)

    class Broker:
        calls = 0

        def place_order(self, _intent):
            self.calls += 1
            return "broker-1"

    broker = Broker()
    runtime = _runtime_for_submit(config, store, broker)
    runtime._execute_via_pipeline(_plan(), 100_000.0, {})
    runtime._execute_via_pipeline(_plan(), 100_000.0, {})

    assert broker.calls == 1
    assert len(store.list_all()) == 1
    assert store.list_all()[0]["state"] == OrderLifecycle.OPEN.value


def test_restart_with_sending_intent_never_resubmits(tmp_path):
    db_path = str(tmp_path / "trade.duckdb")
    config = _config(db_path)
    store = OrderIntentStore(db_path)
    plan = _plan()
    key = idempotency_key(
        plan.plan_id,
        plan.symbol,
        plan.side.value,
        plan.qty,
        plan.entry_price,
        plan.action,
    )
    intent = OrderIntent(
        intent_id="intent-sending",
        signal_id=plan.plan_id,
        symbol=plan.symbol,
        side=plan.side,
        qty=plan.qty,
        order_type="LMT",
        limit_price=plan.entry_price,
    )
    store.persist(
        intent,
        key,
        plan.plan_id,
        state=OrderLifecycle.SENDING,
    )

    broker = SimpleNamespace(
        calls=0,
        place_order=lambda _intent: setattr(broker, "calls", broker.calls + 1),
    )
    restarted = _runtime_for_submit(config, OrderIntentStore(db_path), broker)
    restarted._execute_via_pipeline(plan, 100_000.0, {})

    assert broker.calls == 0
    assert store.get_by_key(key)["state"] == OrderLifecycle.SENDING.value


def test_order_status_failure_keeps_order_open_and_nonterminal(tmp_path):
    db_path = str(tmp_path / "trade.duckdb")
    store = OrderIntentStore(db_path)
    plan = _plan()
    key = idempotency_key(
        plan.plan_id,
        plan.symbol,
        plan.side.value,
        plan.qty,
        plan.entry_price,
        plan.action,
    )
    intent = OrderIntent(
        intent_id="intent-open",
        signal_id=plan.plan_id,
        symbol=plan.symbol,
        side=plan.side,
        qty=plan.qty,
        order_type="LMT",
        limit_price=plan.entry_price,
        idempotency_key=key,
        plan_id=plan.plan_id,
    )
    store.persist(intent, key, plan.plan_id)
    store.update(
        key,
        state=OrderLifecycle.OPEN.value,
        broker_order_id="broker-open",
    )

    runtime = Runtime.__new__(Runtime)
    runtime._broker = SimpleNamespace(
        get_order_status=lambda _broker_id: OrderStatus.FAILED,
    )
    runtime._order_store = store
    runtime._open_orders = {"broker-open": intent}
    runtime._poll_orders()

    assert "broker-open" in runtime._open_orders
    assert store.get_by_key(key)["state"] == OrderLifecycle.OPEN.value


def test_get_fill_failure_keeps_order_open_and_never_applies_fill(tmp_path):
    """A transient get_fill failure (status says FILLED, fill lookup errors)
    must not mark the order done or touch the portfolio — otherwise the fill
    is silently lost forever with no retry and no visible error.
    """
    db_path = str(tmp_path / "trade.duckdb")
    store = OrderIntentStore(db_path)
    plan = _plan()
    key = idempotency_key(
        plan.plan_id, plan.symbol, plan.side.value, plan.qty,
        plan.entry_price, plan.action,
    )
    intent = OrderIntent(
        intent_id="intent-open", signal_id=plan.plan_id, symbol=plan.symbol,
        side=plan.side, qty=plan.qty, order_type="LMT",
        limit_price=plan.entry_price, idempotency_key=key, plan_id=plan.plan_id,
    )
    store.persist(intent, key, plan.plan_id)
    store.update(key, state=OrderLifecycle.OPEN.value, broker_order_id="broker-open")

    apply_fill_calls = []
    runtime = Runtime.__new__(Runtime)
    runtime._broker = SimpleNamespace(
        get_order_status=lambda _broker_id: OrderStatus.FILLED,
        get_fill=lambda _broker_id: (_ for _ in ()).throw(TimeoutError("boom")),
    )
    runtime._order_store = store
    runtime._open_orders = {"broker-open": intent}
    runtime._portfolio = SimpleNamespace(
        apply_fill=lambda fill: apply_fill_calls.append(fill)
    )
    runtime._poll_orders()

    assert "broker-open" in runtime._open_orders
    assert apply_fill_calls == []
    assert store.get_by_key(key)["state"] == OrderLifecycle.OPEN.value


def test_repeated_cumulative_partial_fills_apply_only_delta_and_finish(tmp_path):
    db_path = str(tmp_path / "trade.duckdb")
    config = _config(db_path)
    store = OrderIntentStore(db_path)
    plan = _plan()
    key = idempotency_key(
        plan.plan_id,
        plan.symbol,
        plan.side.value,
        plan.qty,
        plan.entry_price,
        plan.action,
    )
    intent = OrderIntent(
        intent_id="intent-1",
        signal_id=plan.plan_id,
        symbol=plan.symbol,
        side=plan.side,
        qty=plan.qty,
        order_type="LMT",
        limit_price=plan.entry_price,
        idempotency_key=key,
        client_order_id=client_order_id(key),
        plan_id=plan.plan_id,
    )
    store.persist(intent, key, plan.plan_id)
    store.update(
        key,
        state=OrderLifecycle.OPEN.value,
        broker_order_id="broker-1",
    )

    cumulative_fills = [4.0, 6.0, 6.0, 10.0]
    statuses = [
        OrderStatus.PARTIAL,
        OrderStatus.PARTIAL,
        OrderStatus.PARTIAL,
        OrderStatus.FILLED,
    ]

    class Broker:
        def get_order_status(self, _broker_id):
            return statuses.pop(0)

        def get_fill(self, _broker_id):
            cumulative = cumulative_fills.pop(0)
            return Fill(
                order_id="broker-1",
                intent_id="",
                symbol="AAPL",
                side=Side.BUY,
                filled_qty=cumulative,
                avg_price=100.0,
                fill_time=datetime.now(timezone.utc),
            )

    runtime = Runtime.__new__(Runtime)
    runtime._broker = Broker()
    runtime._order_store = store
    runtime._open_orders = {"broker-1": intent}
    runtime._portfolio = Portfolio(config)
    runtime._signal_store = SimpleNamespace(
        apply_fill=lambda *_: None,
    )
    runtime._risk = RiskEngine(config)
    runtime._notifier = SimpleNamespace(send=lambda *_: True)
    runtime._live_plans = {"AAPL": plan}
    runtime._monitor_plans = {"AAPL": plan}
    runtime._position_plan_store = PositionPlanStore(db_path)
    runtime._position_plan_projector = PositionPlanFillProjector(
        runtime._position_plan_store
    )

    for _ in range(4):
        runtime._poll_orders()

    row = store.get_by_key(key)
    position_plan = runtime._position_plan_store.current_for_symbol("AAPL")
    assert runtime._portfolio.positions["AAPL"].qty == 10.0
    assert position_plan is not None
    assert position_plan.open_quantity == 10.0
    assert position_plan.version == 3
    assert len(
        runtime._position_plan_store.history(
            position_plan.position_plan_id
        )
    ) == 3
    assert row["filled_qty"] == 10.0
    assert row["remaining_qty"] == 0.0
    assert row["state"] == OrderLifecycle.FILLED.value
    assert runtime._open_orders == {}
