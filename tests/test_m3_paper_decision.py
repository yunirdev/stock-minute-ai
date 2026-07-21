from datetime import datetime, timedelta, timezone

import pytest

from trader.ai.safety import AIScoreSnapshot
from trader.paper_decision import (
    PaperDecisionService,
    StrategyStatistics,
    StrategyStatisticsRepository,
    UniverseProvider,
)


NOW = datetime(2026, 7, 20, 16, tzinfo=timezone.utc)


def stats(strategy="trend", regime="bull", net=0.2, stats_id="s1"):
    return StrategyStatistics(
        stats_id, "AAPL", strategy, "1", "5m", regime, net, 1.5, 0.1,
        100, 0.55, 0.01, 0.001, 0.001, NOW - timedelta(days=365),
        NOW - timedelta(days=1), NOW - timedelta(days=1), "v1", {"atr_multiplier": 1.5},
    )


def advisory(*, stale=False, stub=False):
    return AIScoreSnapshot(
        "AAPL", 75, NOW - timedelta(minutes=31 if stale else 1), "run-1",
        "stub" if stub else "ollama", "model-1", source="agent_manager", is_stub=stub,
    )


def decide(service, records, ai=None, regime="bull"):
    return service.decide(
        bars={"AAPL": [1]}, positions={}, candidates=[{"symbol": "AAPL", "score": 70}],
        strategy_statistics=StrategyStatisticsRepository(records),
        ai_advisories={"AAPL": ai} if ai else {}, market_regime=regime, now=NOW,
        timeframe="5m", universe_version="u1", data_version="d1",
    )


def test_same_inputs_produce_same_serializable_decision_without_ui():
    service = PaperDecisionService()
    first = decide(service, [stats()], advisory())[0]
    second = decide(service, [stats()], advisory())[0]
    assert first == second
    assert first.to_dict()["decision_id"].startswith("dec-")


def test_stale_universe_is_rejected():
    provider = UniverseProvider(["AAPL"], max_pool_age_minutes=30)
    pool = {"updated_at": (NOW - timedelta(minutes=31)).isoformat(), "items": [{"symbol": "AAPL"}]}
    with pytest.raises(ValueError, match="UNIVERSE_STALE"):
        provider.provide(daily_pool=pool, now=NOW)


@pytest.mark.parametrize("snapshot", [advisory(stale=True), advisory(stub=True)])
def test_invalid_ai_is_fail_closed(snapshot):
    assert decide(PaperDecisionService(), [stats()], snapshot) == []


def test_explicit_quant_mode_does_not_pretend_ai_participated():
    decision = decide(PaperDecisionService(allow_without_ai=True), [stats()])[0]
    assert decision.ai_advisory_run_id is None
    assert "AI_NOT_USED" in decision.reason_codes
    assert decision.evidence == {}


def test_no_reliable_statistics_means_no_subjective_strategy_choice():
    unreliable = stats()
    unreliable = StrategyStatistics(**{**unreliable.__dict__, "trade_count": 2})
    assert decide(PaperDecisionService(), [unreliable], advisory()) == []


def test_regime_selects_only_validated_matching_strategy():
    records = [stats("trend", "bull", stats_id="bull"), stats("mean_revert", "bear", stats_id="bear")]
    assert decide(PaperDecisionService(), records, advisory(), "bear")[0].strategy == "mean_revert"


def test_rejected_alternatives_have_reason_codes():
    records = [stats("trend", net=0.3, stats_id="best"), stats("mean_revert", net=0.1, stats_id="other")]
    decision = decide(PaperDecisionService(), records, advisory())[0]
    assert decision.rejected_alternatives == ({"strategy": "mean_revert", "reason_code": "LOWER_VALIDATED_NET_RETURN"},)
