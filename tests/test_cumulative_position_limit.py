from types import SimpleNamespace

from trader.allocator import EqualWeightAllocator
from trader.config import RiskConfig, TradingConfig
from trader.models import OrderIntent, Position, Side, TradePlan
from trader.order_lifecycle import OrderIntentStore, OrderLifecycle, idempotency_key
from trader.risk_engine import RiskEngine
from trader.runtime import Runtime


def _plan(*, qty: float = 0.0, action: str = "ADD") -> TradePlan:
    return TradePlan(
        plan_id="plan-aapl",
        symbol="AAPL",
        side=Side.BUY,
        action=action,
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        qty=qty,
        confidence=0.9,
    )


def _config(db_path: str, *, max_position_pct: float = 0.20) -> TradingConfig:
    return TradingConfig(
        db_path=db_path,
        broker_type="alpaca_paper",
        auto_trade_paper=True,
        risk=RiskConfig(max_position_pct=max_position_pct),
    )


def _pending_buy(
    store: OrderIntentStore,
    *,
    qty: float,
    price: float = 100.0,
    state: OrderLifecycle = OrderLifecycle.OPEN,
    filled_qty: float = 0.0,
    remaining_qty: float | None = None,
) -> str:
    key = idempotency_key("pending-plan", "AAPL", "BUY", qty, price, "ADD")
    intent = OrderIntent(
        intent_id="pending-intent",
        signal_id="pending-plan",
        symbol="AAPL",
        side=Side.BUY,
        qty=qty,
        order_type="LMT",
        limit_price=price,
    )
    store.persist(intent, key, "pending-plan", state=state)
    store.update(
        key,
        state=state.value,
        filled_qty=filled_qty,
        remaining_qty=qty - filled_qty if remaining_qty is None else remaining_qty,
    )
    return key


def test_allocator_allocates_only_remaining_symbol_capacity():
    allocator = EqualWeightAllocator(max_position_pct=0.20)
    plan = _plan()
    positions = {"AAPL": Position("AAPL", qty=100.0, avg_entry_px=100.0)}

    allocated = allocator.allocate(
        [plan],
        equity=100_000.0,
        positions=positions,
        pending_buy_notional={"AAPL": 5_000.0},
    )

    assert allocated == [plan]
    assert plan.qty == 50.0
    assert plan.target_weight == 0.20


def test_allocator_drops_buy_when_existing_position_is_at_limit():
    allocator = EqualWeightAllocator(max_position_pct=0.20)
    plan = _plan()
    positions = {"AAPL": Position("AAPL", qty=200.0, avg_entry_px=100.0)}

    assert allocator.allocate([plan], 100_000.0, positions) == []
    assert plan.qty == 0.0


def test_risk_rejects_position_plus_pending_plus_new_order_over_limit(tmp_path):
    risk = RiskEngine(_config(str(tmp_path / "trade.duckdb")))
    plan = _plan(qty=20.0)
    positions = {"AAPL": Position("AAPL", qty=150.0, avg_entry_px=100.0)}

    verdict = risk.evaluate_plan(
        plan,
        100_000.0,
        positions,
        pending_buy_notional={"AAPL": 4_000.0},
    )

    assert not verdict.approved
    assert "累计仓位" in verdict.reason


def test_cumulative_long_limit_does_not_block_position_reduction(tmp_path):
    risk = RiskEngine(_config(str(tmp_path / "trade.duckdb")))
    plan = TradePlan(
        plan_id="close-aapl",
        symbol="AAPL",
        side=Side.SELL,
        action="CLOSE",
        entry_price=100.0,
        stop_loss=105.0,
        take_profit=90.0,
        qty=300.0,
    )
    positions = {"AAPL": Position("AAPL", qty=300.0, avg_entry_px=100.0)}

    verdict = risk.evaluate_plan(
        plan,
        100_000.0,
        positions,
        pending_buy_notional={"AAPL": 50_000.0},
    )

    assert verdict.approved


def test_store_pending_exposure_uses_remaining_qty_and_survives_restart(tmp_path):
    db_path = str(tmp_path / "trade.duckdb")
    store = OrderIntentStore(db_path)
    key = _pending_buy(
        store,
        qty=100.0,
        filled_qty=40.0,
        remaining_qty=60.0,
        state=OrderLifecycle.PARTIALLY_FILLED,
    )

    assert store.pending_buy_notional_by_symbol() == {"AAPL": 6_000.0}
    assert OrderIntentStore(db_path).pending_buy_notional_by_symbol() == {
        "AAPL": 6_000.0
    }

    store.update(key, state=OrderLifecycle.CANCELED.value)
    assert store.pending_buy_notional_by_symbol() == {}


def test_execution_pipeline_rechecks_latest_pending_exposure_before_submit(
    tmp_path,
):
    db_path = str(tmp_path / "trade.duckdb")
    config = _config(db_path)
    store = OrderIntentStore(db_path)
    _pending_buy(store, qty=50.0)

    class Broker:
        submissions = 0

        def place_order(self, _intent):
            self.submissions += 1
            return "should-not-submit"

    runtime = Runtime.__new__(Runtime)
    runtime._cfg = config
    runtime._risk = RiskEngine(config)
    runtime._order_store = store
    runtime._broker = Broker()
    runtime._kill = SimpleNamespace(engaged=lambda: False)
    runtime._reconciliation_blocked = False
    runtime._live_plans = {}

    plan = _plan(qty=60.0)
    positions = {"AAPL": Position("AAPL", qty=100.0, avg_entry_px=100.0)}
    runtime._execute_via_pipeline(plan, 100_000.0, positions)
    runtime._execute_via_pipeline(plan, 100_000.0, positions)

    assert runtime._broker.submissions == 0
    assert store.pending_buy_notional_by_symbol() == {"AAPL": 5_000.0}
    assert plan.status == "REJECTED"
