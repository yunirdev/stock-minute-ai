from datetime import datetime, timezone

import pytest

from trader.episode_reviews import EpisodeReviewStore

NOW = datetime(2026, 7, 27, 18, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "outcome,error_code",
    [
        ("SUCCESS", "NONE"),
        ("RISK_REJECTED", "RISK_POSITION_LIMIT"),
        ("NO_FILL", "EXECUTION_EXPIRED"),
        ("DATA_FAILURE", "DATA_SNAPSHOT_STALE"),
        ("BROKER_FAILURE", "BROKER_API_CONNECTION"),
    ],
)
def test_all_review_outcomes_are_frozen_replayable_and_idempotent(
    tmp_path,
    outcome,
    error_code,
):
    store = EpisodeReviewStore(tmp_path / "trade.duckdb")
    values = {
        "subject_id": f"subject-{outcome}",
        "outcome": outcome,
        "error_code": error_code,
        "facts": {"snapshot_id": "snapshot-1"},
        "decision": {"decision_id": "decision-1"},
        "execution": {"order_state": "NONE"},
        "result": {
            "realized_pnl": -10 if outcome == "SUCCESS" else 0,
            "strategy_invalidated": False,
        },
        "created_at": NOW,
    }
    first = store.create(**values)
    duplicate = store.create(**values)

    assert first == duplicate
    assert first["outcome"] == outcome
    if outcome == "SUCCESS":
        assert first["result"]["realized_pnl"] == -10
        assert not first["result"]["strategy_invalidated"]


def test_review_rejects_cross_category_error_code(tmp_path):
    store = EpisodeReviewStore(tmp_path / "trade.duckdb")
    with pytest.raises(ValueError, match="TAXONOMY_INVALID"):
        store.create(
            subject_id="subject-1",
            outcome="BROKER_FAILURE",
            error_code="DATA_STALE",
            facts={"snapshot": "1"},
            decision={"decision": "1"},
            execution={"order": "none"},
            result={"pnl": 0},
            created_at=NOW,
        )
