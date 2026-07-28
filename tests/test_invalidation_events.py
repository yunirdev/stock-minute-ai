from datetime import datetime, timedelta, timezone

import pytest

from trader.invalidation_events import (
    InvalidationEventStore,
    build_invalidation_event,
)
from trader.models import (
    InvalidationEvent,
    InvalidationEventType,
    InvalidationSource,
    PositionPlan,
    PositionPlanStatus,
    Side,
)

NOW = datetime(2026, 7, 27, 15, 0, tzinfo=timezone.utc)


def _plan(
    *,
    version_id: str = "position-version-1",
    created_at: datetime = NOW - timedelta(minutes=2),
) -> PositionPlan:
    return PositionPlan(
        position_plan_id="position-plan-1",
        version_id=version_id,
        version=1,
        parent_version_id="",
        symbol="AAPL",
        side=Side.BUY,
        status=PositionPlanStatus.ACTIVE,
        source_trade_plan_id="trade-plan-1",
        initial_fill_id="fill-1",
        initial_entry_price=100,
        initial_quantity=10,
        open_quantity=10,
        average_entry_price=100,
        stop_loss=95,
        take_profit=115,
        invalidation_rules=(
            "PRICE_STOP",
            "BROKER_RESTRICTION",
            "CORPORATE_ACTION",
            "TRADING_RESTRICTION",
            "STRATEGY_INVALIDATED",
        ),
        change_reason="INITIAL_FILL",
        created_at=created_at,
    )


def _price_event(
    plan: PositionPlan,
    *,
    source_event_id: str = "alpaca-trade-1",
    facts: dict | None = None,
):
    return build_invalidation_event(
        plan=plan,
        event_type=InvalidationEventType.PRICE_STOP,
        source=InvalidationSource.MARKET_DATA,
        source_event_id=source_event_id,
        as_of=NOW - timedelta(seconds=5),
        observed_at=NOW,
        facts=facts
        or {
            "trigger_price": 94.9,
            "threshold_price": 95,
        },
        evidence_refs=("market-snapshot-1",),
    )


def test_valid_source_fact_is_persisted_and_duplicate_is_idempotent(
    tmp_path,
):
    plan = _plan()
    store = InvalidationEventStore(tmp_path / "trade.duckdb")
    event = _price_event(plan)

    assert store.record(event, plan=plan, received_at=NOW)
    assert not store.record(event, plan=plan, received_at=NOW)
    assert store.list_for_plan(plan.position_plan_id) == [event]


def test_same_source_identity_with_changed_facts_is_rejected(tmp_path):
    plan = _plan()
    store = InvalidationEventStore(tmp_path / "trade.duckdb")
    event = _price_event(plan)
    store.record(event, plan=plan, received_at=NOW)
    conflict = _price_event(
        plan,
        facts={
            "trigger_price": 94.8,
            "threshold_price": 95,
        },
    )

    with pytest.raises(
        ValueError,
        match="INVALIDATION_EVENT_SOURCE_CONFLICT",
    ):
        store.record(conflict, plan=plan, received_at=NOW)


@pytest.mark.parametrize(
    "event_type,source,facts",
    [
        (
            InvalidationEventType.BROKER_RESTRICTION,
            InvalidationSource.BROKER,
            {"active": True, "fact_code": "ACCOUNT_RESTRICTED"},
        ),
        (
            InvalidationEventType.CORPORATE_ACTION,
            InvalidationSource.CORPORATE_ACTION_DATA,
            {"active": True, "fact_code": "MERGER_EFFECTIVE"},
        ),
        (
            InvalidationEventType.TRADING_RESTRICTION,
            InvalidationSource.EXCHANGE,
            {"active": True, "fact_code": "HALTED"},
        ),
        (
            InvalidationEventType.STRATEGY_INVALIDATED,
            InvalidationSource.STRATEGY_ENGINE,
            {"evaluation_id": "strategy-eval-1", "valid": False},
        ),
    ],
)
def test_all_deterministic_event_types_accept_authoritative_sources(
    tmp_path,
    event_type,
    source,
    facts,
):
    plan = _plan()
    event = build_invalidation_event(
        plan=plan,
        event_type=event_type,
        source=source,
        source_event_id=f"source-{event_type.value}",
        as_of=NOW,
        observed_at=NOW,
        facts=facts,
        evidence_refs=("source-record-1",),
    )
    store = InvalidationEventStore(
        tmp_path / f"{event_type.value}.duckdb"
    )
    assert store.record(event, plan=plan, received_at=NOW)


def test_invalid_source_time_plan_and_untriggered_price_fail_closed(
    tmp_path,
):
    plan = _plan()
    store = InvalidationEventStore(tmp_path / "trade.duckdb")
    invalid_source = build_invalidation_event(
        plan=plan,
        event_type=InvalidationEventType.PRICE_STOP,
        source=InvalidationSource.BROKER,
        source_event_id="broker-price",
        as_of=NOW,
        observed_at=NOW,
        facts={"trigger_price": 94, "threshold_price": 95},
        evidence_refs=("broker-record",),
    )
    with pytest.raises(
        ValueError,
        match="INVALIDATION_SOURCE_NOT_ALLOWED",
    ):
        store.record(invalid_source, plan=plan, received_at=NOW)

    stale = _price_event(
        _plan(created_at=NOW - timedelta(hours=1)),
    )
    with pytest.raises(ValueError, match="INVALIDATION_EVENT_STALE"):
        store.record(
            stale,
            plan=_plan(created_at=NOW - timedelta(hours=1)),
            received_at=NOW + timedelta(hours=1),
        )

    other_version = _plan(version_id="position-version-other")
    with pytest.raises(
        ValueError,
        match="INVALIDATION_POSITION_PLAN_MISMATCH",
    ):
        store.record(
            _price_event(plan),
            plan=other_version,
            received_at=NOW,
        )

    with pytest.raises(
        ValueError,
        match="INVALIDATION_PRICE_NOT_TRIGGERED",
    ):
        store.record(
            _price_event(
                plan,
                facts={
                    "trigger_price": 96,
                    "threshold_price": 95,
                },
            ),
            plan=plan,
            received_at=NOW,
        )


def test_noncanonical_or_unverifiable_text_event_is_rejected(tmp_path):
    plan = _plan()
    store = InvalidationEventStore(tmp_path / "trade.duckdb")
    event = _price_event(plan)
    text_only = InvalidationEvent(
        **{
            **event.__dict__,
            "facts_json": '{"comment":"LLM says sell"}',
        }
    )
    with pytest.raises(
        ValueError,
        match="INVALIDATION_PRICE_FACTS_INVALID",
    ):
        store.record(text_only, plan=plan, received_at=NOW)
