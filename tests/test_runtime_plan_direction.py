from types import SimpleNamespace

from trader.models import Side
from trader.plan import ATRPlanner
from trader.runtime import Runtime


def test_runtime_translates_sell_decision_into_sell_atr_plan(
    sample_candidate, sample_bar,
):
    runtime = Runtime.__new__(Runtime)
    runtime._planner = ATRPlanner()
    decision = SimpleNamespace(
        params={"atr_multiplier": 1.5, "rr_ratio": 2.0},
        side=Side.SELL,
    )

    assert sample_candidate.score >= 50
    plan = runtime._make_decision_plan(
        sample_candidate,
        sample_bar,
        decision,
        current_qty=0.0,
        bars_history=[],
    )

    assert plan.side == Side.SELL
    assert plan.take_profit < plan.entry_price < plan.stop_loss
