from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from trader.ai.safety import AIScoreSnapshot
from trader.models import Bar, Fill, Side, TradePlan
from trader.signal_reports import (
    SignalState,
    SignalStore,
    build_ready_signal_report,
)


def test_signal_report_tracks_entry_and_exit(tmp_path):
    now = datetime(2026, 7, 27, 15, 0, tzinfo=timezone.utc)
    plan = TradePlan(
        "plan-1",
        "AAPL",
        Side.BUY,
        "OPEN",
        100,
        95,
        110,
        target_weight=0.1,
        qty=10,
        metadata={"decision_id": "decision-1", "strategy": "momentum"},
    )
    decision = SimpleNamespace(
        decision_id="decision-1",
        strategy="momentum",
        market_regime="bull_trend",
        valid_until=now + timedelta(minutes=15),
        reason_codes=("AI_EVIDENCE_VALID",),
    )
    snapshot = AIScoreSnapshot(
        "AAPL",
        82,
        now,
        run_id="research-1",
        source="daily_research",
        contributors=[{"agent_name": "tradingagents_graph"}],
    )
    bar = Bar("AAPL", now, 99, 102, 98, 101, 1000)
    report = build_ready_signal_report(
        plan,
        decision,
        snapshot,
        bar,
        equity=10_000,
        now=now,
        timeframe="5m",
    )
    store = SignalStore(str(tmp_path / "trade.duckdb"))
    saved, created = store.register_ready(report)
    assert created
    assert saved.state == SignalState.READY

    entered = store.apply_fill(
        plan.plan_id,
        Fill("o1", "i1", "AAPL", Side.BUY, 10, 100, now),
    )
    assert entered is not None
    assert entered.state == SignalState.ENTERED
    store.mark_exit("AAPL", plan_id="close-plan", at=now)
    closed = store.apply_fill(
        "close-plan",
        Fill("o2", "i2", "AAPL", Side.SELL, 10, 105, now),
    )
    assert closed is not None
    assert closed.state == SignalState.CLOSED
    assert closed.realized_pnl == 50
    assert closed.realized_return_pct == 5
