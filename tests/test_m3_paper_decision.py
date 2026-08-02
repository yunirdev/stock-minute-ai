from datetime import datetime, timedelta, timezone

import pytest

from trader.ai.safety import AIScoreSnapshot
from trader.paper_decision import (
    PaperDecisionService,
    StrategyStatisticsRepository,
    UniverseProvider,
)

NOW = datetime(2026, 7, 20, 16, tzinfo=timezone.utc)

def advisory(*, stale=False, stub=False, recommendation="BUY"):
    return AIScoreSnapshot(
        "AAPL", 75, NOW - timedelta(minutes=31 if stale else 1), "run-1",
        "stub" if stub else "ollama", "model-1", source="agent_manager", is_stub=stub,
        recommendation=recommendation,
    )

def decide(service, ai=None, regime="bull"):
    # Direction and eligibility are entirely AI-driven now (bull_bear_debate
    # recommendation) — strategy_statistics/votes no longer participate, so
    # candidates only need a symbol/score. `records` param removed from
    # call sites below; kept as a no-op arg on StrategyStatisticsRepository
    # only where a test still wants to prove it's ignored.
    return service.decide(
        bars={"AAPL": [1]}, positions={}, candidates=[{"symbol": "AAPL", "score": 70, "reasons": {}}],
        strategy_statistics=StrategyStatisticsRepository([]),
        ai_advisories={"AAPL": ai} if ai else {}, market_regime=regime, now=NOW,
        timeframe="5m", universe_version="u1", data_version="d1",
    )

def test_same_inputs_produce_same_serializable_decision_without_ui():
    service = PaperDecisionService()
    first = decide(service, advisory())[0]
    second = decide(service, advisory())[0]
    assert first == second
    assert first.to_dict()["decision_id"].startswith("dec-")
    assert first.side.value == "BUY"

def test_stale_universe_is_rejected():
    provider = UniverseProvider(["AAPL"], max_pool_age_minutes=30)
    pool = {"updated_at": (NOW - timedelta(minutes=31)).isoformat(), "items": [{"symbol": "AAPL"}]}
    with pytest.raises(ValueError, match="UNIVERSE_STALE"):
        provider.provide(daily_pool=pool, now=NOW)

@pytest.mark.parametrize("snapshot", [advisory(stale=True), advisory(stub=True)])
def test_invalid_ai_is_fail_closed(snapshot):
    assert decide(PaperDecisionService(), snapshot) == []

def test_hold_recommendation_produces_no_decision():
    # bull_bear_debate can land on HOLD (no decisive side) — that must not
    # trade, with or without allow_without_ai, since there's no other
    # direction source left once strategy voting is removed.
    assert decide(PaperDecisionService(), advisory(recommendation="HOLD")) == []
    assert decide(PaperDecisionService(allow_without_ai=True), advisory(recommendation="HOLD")) == []

def test_no_ai_advisory_at_all_produces_no_decision():
    # allow_without_ai only relaxes the AIScoreValidator gate's strictness;
    # it does not invent a direction when there is no advisory to read one
    # from, since strategy votes no longer supply a fallback side.
    assert decide(PaperDecisionService(allow_without_ai=True)) == []

def test_sell_recommendation_without_short_enabled_is_no_decision():
    # No held position + allow_short=False (the safety default) → AI bearish
    # calls exit an existing long downstream but must not open a new short.
    assert decide(PaperDecisionService(), advisory(recommendation="SELL")) == []

def test_sell_recommendation_maps_to_short_side_when_enabled():
    decision = decide(PaperDecisionService(allow_short=True), advisory(recommendation="SELL"))[0]
    assert decision.side.value == "SELL"
    assert decision.evidence["ai"]["recommendation"] == "SELL"

def test_strategy_statistics_no_longer_gate_the_decision():
    # Passing zero usable statistics records must not block a decision —
    # confirms decide() truly stopped depending on the removed 24-strategy
    # backtest-vote layer.
    decision = decide(PaperDecisionService(), advisory())[0]
    assert decision.strategy == "ai_consensus"
    assert decision.rejected_alternatives == ()

def test_advisory_worker_is_single_flight(tmp_path):
    from trader.paper_decision import AdvisoryWorker

    class Manager:
        def run_cycle(self, context, db_path):
            return [context, db_path]

    worker = AdvisoryWorker(Manager(), min_interval_seconds=60)
    assert worker.start("ctx", str(tmp_path / "ai.duckdb"))
    while worker.poll() is None:
        pass
    assert not worker.start("ctx", str(tmp_path / "ai.duckdb"))
    worker.close()

def test_worker_recovers_after_agent_failure(tmp_path):
    from trader.paper_decision import AdvisoryWorker

    class FailingManager:
        def run_cycle(self, context, db_path):
            raise RuntimeError("agent failed")

    worker = AdvisoryWorker(FailingManager(), min_interval_seconds=0)
    assert worker.start("ctx", str(tmp_path / "ai.duckdb"))
    while not worker._future.done():
        pass
    with pytest.raises(RuntimeError, match="agent failed"):
        worker.poll()
    assert worker.start("ctx", str(tmp_path / "ai.duckdb"))
    worker.close()

def test_statistics_evaluate_only_holdout_tail():
    import pandas as pd

    from trader.strategy_core import STRATEGY_OPTIONS
    from trader.strategy_statistics import evaluate_strategy

    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    bars = pd.DataFrame(
        {
            "timestamp_utc": [start + timedelta(minutes=index) for index in range(240)],
            "open": [100 + index * 0.1 for index in range(240)],
            "high": [101 + index * 0.1 for index in range(240)],
            "low": [99 + index * 0.1 for index in range(240)],
            "close": [100 + index * 0.1 for index in range(240)],
            "volume": [1000] * 240,
        }
    )
    record = evaluate_strategy(
        bars,
        symbol="AAPL",
        strategy=STRATEGY_OPTIONS[0],
        timeframe="5m",
        market_regime="bull_trend",
        now=NOW,
    )
    assert record is not None
    assert record.data_start > start
    assert record.data_end > record.data_start
