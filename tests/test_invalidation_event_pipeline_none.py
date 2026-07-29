"""process_invalidation_event() must not crash when _execute_via_pipeline()
returns None (the pipeline's documented "already in flight / nothing new
to submit" outcome, e.g. a legacy idempotency-key duplicate). The method
used to reach straight for `prepared_intent.idempotency_key` without a
None check, unlike the `order_plan is not None` guard two lines above it
-- any pipeline call that legitimately returns None would crash the
invalidation-event handler instead of just skipping the order-link step.
"""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from trader.config import RiskConfig, TradingConfig
from trader.invalidation_events import InvalidationEventStore
from trader.models import (
    InvalidationEventType,
    InvalidationSource,
    PositionAdjustmentStatus,
    PositionPlan,
    PositionPlanStatus,
    Side,
)
from trader.invalidation_events import build_invalidation_event
from trader.order_lifecycle import OrderIntentStore
from trader.portfolio import Portfolio
from trader.position_adjustments import PositionAdjustmentStore
from trader.position_plans import PositionPlanStore
from trader.risk_engine import RiskEngine
from trader.runtime import Runtime

NOW = datetime(2026, 7, 27, 16, 0, tzinfo=timezone.utc)


def _config(path) -> TradingConfig:
    return TradingConfig(
        db_path=str(path),
        broker_type="alpaca_paper",
        auto_trade_paper=True,
        risk=RiskConfig(max_position_pct=0.20, max_trade_risk_pct=0.005),
    )


def _plan() -> PositionPlan:
    return PositionPlan(
        position_plan_id="position-plan-pipeline-none",
        version_id="position-version-pipeline-none-1",
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
        invalidation_rules=("PRICE_STOP", "STRATEGY_INVALIDATED"),
        change_reason="INITIAL_FILL",
        created_at=NOW - timedelta(minutes=1),
    )


def _event(plan):
    return build_invalidation_event(
        plan=plan,
        event_type=InvalidationEventType.PRICE_STOP,
        source=InvalidationSource.MARKET_DATA,
        source_event_id="source-event-pipeline-none",
        as_of=NOW,
        observed_at=NOW,
        facts={"trigger_price": 94.9, "threshold_price": 95},
        evidence_refs=("evidence-1",),
    )


def test_process_invalidation_event_survives_pipeline_returning_none(tmp_path):
    path = tmp_path / "trade.duckdb"
    config = _config(path)
    plan_store = PositionPlanStore(path)
    plan = plan_store.create(_plan())
    event = _event(plan)

    runtime = Runtime.__new__(Runtime)
    runtime._cfg = config
    runtime._broker = SimpleNamespace(place_order=lambda _intent: "unused")
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
    runtime._bug_reporter = SimpleNamespace(capture_exception=lambda *_, **__: None)

    # Simulate the pipeline's "nothing new to submit" outcome (e.g. a legacy
    # idempotency-key duplicate already in flight).
    runtime._execute_via_pipeline = lambda *_args, **_kwargs: None

    adjustment = runtime.process_invalidation_event(
        event,
        limit_price=94.8,
        received_at=NOW,
    )

    assert adjustment.status == PositionAdjustmentStatus.PLANNED
    assert adjustment.order_intent_id == ""
