from datetime import datetime, timedelta, timezone

import duckdb
import pytest

from trader.models import (
    Fill,
    PositionPlan,
    PositionPlanStatus,
    Side,
    TradePlan,
)
from trader.position_plans import (
    PositionPlanFillProjector,
    PositionPlanStore,
)


NOW = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)


def _plan(
    *,
    version: int = 1,
    version_id: str = "position-plan-1:v1",
    parent_version_id: str = "",
    status: PositionPlanStatus = PositionPlanStatus.ACTIVE,
    open_quantity: float = 10,
    reason: str = "INITIAL_FILL",
) -> PositionPlan:
    return PositionPlan(
        position_plan_id="position-plan-1",
        version_id=version_id,
        version=version,
        parent_version_id=parent_version_id,
        symbol="aapl",
        side=Side.BUY,
        status=status,
        source_trade_plan_id="trade-plan-1",
        initial_fill_id="fill-1",
        initial_entry_price=100,
        initial_quantity=10,
        open_quantity=open_quantity,
        average_entry_price=100,
        stop_loss=95,
        take_profit=115,
        invalidation_rules=("PRICE_STOP", "BROKER_RESTRICTION"),
        change_reason=reason,
        created_at=NOW + timedelta(minutes=version - 1),
    )


def test_position_plan_store_preserves_unrelated_legacy_data(tmp_path):
    path = tmp_path / "trade.duckdb"
    conn = duckdb.connect(str(path))
    conn.execute("CREATE TABLE legacy_orders (id TEXT PRIMARY KEY)")
    conn.execute("INSERT INTO legacy_orders VALUES ('old-order')")
    conn.close()

    PositionPlanStore(path)

    conn = duckdb.connect(str(path), read_only=True)
    try:
        assert conn.execute("SELECT id FROM legacy_orders").fetchone() == (
            "old-order",
        )
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT table_name FROM information_schema.tables"
            ).fetchall()
        }
    finally:
        conn.close()
    assert {
        "position_plan_heads",
        "position_plan_versions",
    }.issubset(tables)


def test_position_plan_version_chain_conflict_and_recovery(tmp_path):
    path = tmp_path / "trade.duckdb"
    store = PositionPlanStore(path)
    first = store.create(_plan())
    second = _plan(
        version=2,
        version_id="position-plan-1:v2",
        parent_version_id=first.version_id,
        status=PositionPlanStatus.REDUCING,
        open_quantity=6,
        reason="PARTIAL_REDUCTION",
    )
    store.append(second, expected_version=1)

    with pytest.raises(
        RuntimeError,
        match="POSITION_PLAN_VERSION_CONFLICT",
    ):
        store.append(
            _plan(
                version=3,
                version_id="position-plan-1:v3-conflict",
                parent_version_id=second.version_id,
            ),
            expected_version=1,
        )

    reopened = PositionPlanStore(path)
    assert reopened.current(first.position_plan_id) == second
    assert reopened.recover_open() == [second]
    assert reopened.history(first.position_plan_id) == [first, second]


def test_position_plan_rejects_invalid_baseline_and_transitions(tmp_path):
    store = PositionPlanStore(tmp_path / "trade.duckdb")
    first = store.create(_plan())
    with pytest.raises(
        ValueError,
        match="POSITION_PLAN_BASELINE_IMMUTABLE",
    ):
        store.append(
            PositionPlan(
                **{
                    **_plan(
                        version=2,
                        version_id="position-plan-1:v2",
                        parent_version_id=first.version_id,
                    ).__dict__,
                    "initial_entry_price": 101,
                }
            ),
            expected_version=1,
        )

    closed = _plan(
        version=2,
        version_id="position-plan-1:closed",
        parent_version_id=first.version_id,
        status=PositionPlanStatus.CLOSED,
        open_quantity=0,
        reason="CLOSED_BY_FILL",
    )
    store.append(closed, expected_version=1)
    with pytest.raises(
        ValueError,
        match="POSITION_PLAN_STATUS_TRANSITION_INVALID",
    ):
        store.append(
            _plan(
                version=3,
                version_id="position-plan-1:reopened",
                parent_version_id=closed.version_id,
                status=PositionPlanStatus.ACTIVE,
            ),
            expected_version=2,
        )


@pytest.mark.parametrize(
    "changes,error",
    [
        (
            {"status": PositionPlanStatus.CLOSED, "open_quantity": 1},
            "CLOSED_POSITION_PLAN_HAS_QUANTITY",
        ),
        (
            {"version": 2, "parent_version_id": ""},
            "POSITION_PLAN_PARENT_REQUIRED",
        ),
        (
            {"stop_loss": 105},
            "LONG_POSITION_PLAN_PRICE_ORDER_INVALID",
        ),
    ],
)
def test_position_plan_model_rejects_invalid_values(changes, error):
    with pytest.raises(ValueError, match=error):
        PositionPlan(**{**_plan().__dict__, **changes})


def _trade_plan() -> TradePlan:
    return TradePlan(
        plan_id="trade-plan-fill",
        symbol="AAPL",
        side=Side.BUY,
        action="OPEN",
        entry_price=100,
        stop_loss=95,
        take_profit=115,
        qty=10,
    )


def _fill(
    *,
    order_id: str,
    side: Side,
    cumulative: float,
    price: float,
    minute: int,
) -> Fill:
    return Fill(
        order_id=order_id,
        intent_id=f"intent-{order_id}",
        symbol="AAPL",
        side=side,
        filled_qty=cumulative,
        avg_price=price,
        fill_time=NOW + timedelta(minutes=minute),
    )


def test_fill_projector_handles_partial_duplicate_reduce_close_and_restart(
    tmp_path,
):
    path = tmp_path / "trade.duckdb"
    store = PositionPlanStore(path)
    projector = PositionPlanFillProjector(store)
    first_fill = _fill(
        order_id="buy-1",
        side=Side.BUY,
        cumulative=4,
        price=100,
        minute=1,
    )
    first = projector.apply(
        fill=first_fill,
        applied_delta=4,
        trade_plan=_trade_plan(),
    )
    assert first is not None
    assert first.version == 1
    assert first.initial_quantity == 4
    assert first.open_quantity == 4

    second_fill = _fill(
        order_id="buy-1",
        side=Side.BUY,
        cumulative=10,
        price=101,
        minute=2,
    )
    second = projector.apply(
        fill=second_fill,
        applied_delta=6,
        trade_plan=_trade_plan(),
    )
    assert second is not None
    assert second.version == 2
    assert second.initial_quantity == 4
    assert second.open_quantity == 10
    assert second.average_entry_price == pytest.approx(100.6)

    duplicate = projector.apply(
        fill=second_fill,
        applied_delta=6,
        trade_plan=_trade_plan(),
    )
    assert duplicate == second
    assert len(store.history(second.position_plan_id)) == 2

    reduced = projector.apply(
        fill=_fill(
            order_id="sell-1",
            side=Side.SELL,
            cumulative=3,
            price=108,
            minute=3,
        ),
        applied_delta=3,
        trade_plan=None,
    )
    assert reduced is not None
    assert reduced.status == PositionPlanStatus.REDUCING
    assert reduced.open_quantity == 7

    closed = projector.apply(
        fill=_fill(
            order_id="sell-2",
            side=Side.SELL,
            cumulative=7,
            price=110,
            minute=4,
        ),
        applied_delta=7,
        trade_plan=None,
    )
    assert closed is not None
    assert closed.status == PositionPlanStatus.CLOSED
    assert closed.open_quantity == 0

    reopened = PositionPlanStore(path)
    assert reopened.recover_open() == []
    assert len(reopened.history(closed.position_plan_id)) == 4


def test_fill_projector_rejects_reduction_without_open_plan(tmp_path):
    projector = PositionPlanFillProjector(
        PositionPlanStore(tmp_path / "trade.duckdb")
    )
    with pytest.raises(
        RuntimeError,
        match="POSITION_PLAN_OPEN_POSITION_REQUIRED",
    ):
        projector.apply(
            fill=_fill(
                order_id="sell-orphan",
                side=Side.SELL,
                cumulative=1,
                price=100,
                minute=1,
            ),
            applied_delta=1,
            trade_plan=None,
        )


def test_fill_projection_rolls_back_plan_and_cursor_together(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "trade.duckdb"
    store = PositionPlanStore(path)
    projector = PositionPlanFillProjector(store)
    fill = _fill(
        order_id="buy-atomic",
        side=Side.BUY,
        cumulative=2,
        price=100,
        minute=1,
    )
    original = PositionPlanStore._insert_version

    def fail_after_version_insert(conn, plan):
        original(conn, plan)
        raise RuntimeError("SIMULATED_CRASH")

    monkeypatch.setattr(
        PositionPlanStore,
        "_insert_version",
        staticmethod(fail_after_version_insert),
    )
    with pytest.raises(RuntimeError, match="SIMULATED_CRASH"):
        projector.apply(
            fill=fill,
            applied_delta=None,
            trade_plan=_trade_plan(),
        )

    assert store.recover_open() == []
    assert store.projected_cumulative_for_order(fill.order_id) == 0

    monkeypatch.setattr(
        PositionPlanStore,
        "_insert_version",
        staticmethod(original),
    )
    created = projector.apply(
        fill=fill,
        applied_delta=None,
        trade_plan=_trade_plan(),
    )
    assert created is not None
    assert created.open_quantity == 2
