from datetime import datetime, timedelta, timezone

import pytest

from trader.ai.manager import AgentManager, get_score_snapshots_from_db
from trader.ai.safety import AIScorePolicy, AIScoreSnapshot, AIScoreValidator
from trader.config import TradingConfig
from trader.models import Advisory


NOW = datetime(2026, 7, 20, 12, tzinfo=timezone.utc)


def test_auto_trade_requires_alpaca_paper():
    with pytest.raises(ValueError, match="AUTO_TRADE_REQUIRES_ALPACA_PAPER"):
        TradingConfig(auto_trade_paper=True, broker_type="alpaca_live")
    TradingConfig(auto_trade_paper=True, broker_type="alpaca_paper")


def test_manual_approval_pipeline_is_removed():
    import importlib.util

    config = TradingConfig()
    assert not hasattr(config, "execution_enabled")
    assert importlib.util.find_spec("trader.approval") is None


@pytest.mark.parametrize(
    ("snapshot", "reason"),
    [
        (None, "AI_SCORE_MISSING"),
        (AIScoreSnapshot("AAPL", 101, NOW, "r", "p", "m", source="db"), "AI_SCORE_OUT_OF_RANGE"),
        (AIScoreSnapshot("AAPL", 80, NOW - timedelta(minutes=31), "r", "p", "m", source="db"), "AI_SCORE_STALE"),
        (AIScoreSnapshot("AAPL", 80, NOW, "r", "stub", "m", source="db", is_stub=True), "AI_SCORE_STUB"),
        (AIScoreSnapshot("AAPL", 80, NOW, "r", None, "m", source="db"), "AI_SCORE_PROVIDER_MISSING"),
        (AIScoreSnapshot("AAPL", 80, NOW, "r", "p", None, source="db"), "AI_SCORE_MODEL_MISSING"),
        (AIScoreSnapshot("AAPL", 80, NOW, None, "p", "m", source="db"), "AI_SCORE_PROVENANCE_MISSING"),
        (AIScoreSnapshot("AAPL", 64, NOW, "r", "p", "m", source="db"), "AI_SCORE_BELOW_THRESHOLD"),
    ],
)
def test_invalid_scores_fail_closed(snapshot, reason):
    result = AIScoreValidator(AIScorePolicy(65, 30), lambda: NOW).validate(snapshot)
    assert not result.valid
    assert result.reason_code == reason


def test_new_advisory_round_trips_with_provenance(tmp_path):
    db_path = str(tmp_path / "ai.duckdb")
    manager = AgentManager(agents=[])
    manager._init_db(db_path)
    manager._write_advisories(
        [Advisory("adv-1", "quant", "quant", {"symbol": "AAPL", "quant_score": 80}, 0.8, created_at=NOW)],
        db_path,
    )
    snapshot = get_score_snapshots_from_db(db_path)["AAPL"]
    result = AIScoreValidator(AIScorePolicy(65, 30), lambda: NOW).validate(snapshot)
    assert result.valid
    assert snapshot.provider == "deterministic"
    assert snapshot.model == "quant:v1"
    assert snapshot.source == "agent_manager"
