from datetime import datetime, timedelta, timezone

import pytest

from trader.models import Fill, Side, TradePlan
from trader.position_plans import (
    PositionPlanFillProjector,
    PositionPlanStore,
)
from trader.trade_episodes import TradeEpisodeStore

NOW = datetime(2026, 7, 27, 15, 0, tzinfo=timezone.utc)


def _fill(order, side, cumulative, price, at):
    return Fill(
        order_id=order,
        intent_id=f"intent-{order}",
        symbol="AAPL",
        side=side,
        filled_qty=cumulative,
        avg_price=price,
        fill_time=at,
    )


def test_episode_attributes_partial_reduce_close_cross_day_and_restart(
    tmp_path,
):
    path = tmp_path / "trade.duckdb"
    projector = PositionPlanFillProjector(PositionPlanStore(path))
    trade_plan = TradePlan(
        plan_id="entry-plan",
        symbol="AAPL",
        side=Side.BUY,
        action="OPEN",
        entry_price=100,
        stop_loss=95,
        take_profit=115,
        qty=10,
    )
    first = projector.apply(
        fill=_fill("buy-1", Side.BUY, 4, 100, NOW),
        applied_delta=None,
        trade_plan=trade_plan,
    )
    assert first is not None
    projector.apply(
        fill=_fill(
            "buy-1",
            Side.BUY,
            10,
            101,
            NOW + timedelta(minutes=1),
        ),
        applied_delta=None,
        trade_plan=trade_plan,
    )
    episodes = TradeEpisodeStore(path)
    partial = episodes.sync(first.position_plan_id)
    assert partial["entry_quantity"] == 10
    assert partial["open_quantity"] == 10

    projector.apply(
        fill=_fill(
            "sell-1",
            Side.SELL,
            3,
            108,
            NOW + timedelta(days=1),
        ),
        applied_delta=None,
        trade_plan=None,
    )
    reduced = episodes.sync(first.position_plan_id)
    assert reduced["exit_quantity"] == 3
    assert reduced["open_quantity"] == 7

    projector.apply(
        fill=_fill(
            "sell-2",
            Side.SELL,
            7,
            110,
            NOW + timedelta(days=1, minutes=1),
        ),
        applied_delta=None,
        trade_plan=None,
    )
    closed = episodes.sync(first.position_plan_id)
    assert closed["status"] == "CLOSED"
    assert closed["open_quantity"] == 0
    assert closed["exit_quantity"] == 10
    assert closed["realized_pnl"] == pytest.approx(88)
    assert closed["attribution"]["cross_day"]
    assert closed["snapshot_version"] == 3

    restarted = TradeEpisodeStore(path)
    duplicate = restarted.sync(first.position_plan_id)
    assert duplicate["snapshot_id"] == closed["snapshot_id"]
    assert duplicate["snapshot_version"] == 3
    assert restarted.latest(first.position_plan_id)[
        "realized_pnl"
    ] == pytest.approx(88)
