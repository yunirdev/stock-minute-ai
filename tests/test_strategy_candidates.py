from datetime import datetime, timezone

import pytest

from trader.episode_reviews import EpisodeReviewStore
from trader.strategy_candidates import ExperimentBoundary, StrategyCandidateStore

TRAINING_START = datetime(2025, 1, 1, tzinfo=timezone.utc)
TRAINING_END = datetime(2025, 12, 31, tzinfo=timezone.utc)
HOLDOUT_START = datetime(2026, 1, 2, tzinfo=timezone.utc)
HOLDOUT_END = datetime(2026, 6, 30, tzinfo=timezone.utc)
NOW = datetime(2026, 7, 27, 18, 0, tzinfo=timezone.utc)


def _review(db_path):
    return EpisodeReviewStore(db_path).create(
        subject_id="episode-1",
        outcome="SUCCESS",
        error_code="NONE",
        facts={"snapshot_id": "snapshot-1"},
        decision={"strategy": "ema_cross"},
        execution={"fill_count": 2},
        result={"realized_pnl": -12.5},
        created_at=NOW,
    )


def _values(review_id):
    return {
        "source_review_id": review_id,
        "strategy_name": "ema_cross",
        "base_strategy_version": "production-ema-v3",
        "dataset_version": "dataset-sha256-abc",
        "code_version": "git-deadbeef",
        "parameters": {"fast": 10, "slow": 30},
        "boundary": ExperimentBoundary(
            training_start=TRAINING_START,
            training_end=TRAINING_END,
            holdout_start=HOLDOUT_START,
            holdout_end=HOLDOUT_END,
        ),
        "rationale": "Reduce the lag observed in the frozen episode review.",
        "created_at": NOW,
    }


def test_candidate_is_versioned_idempotent_and_recovers(tmp_path):
    db_path = tmp_path / "trade.duckdb"
    review = _review(db_path)
    store = StrategyCandidateStore(db_path)

    first = store.create_from_review(**_values(review["review_id"]))
    duplicate = store.create_from_review(**_values(review["review_id"]))
    recovered = StrategyCandidateStore(db_path).get(first["candidate_id"])

    assert duplicate == first
    assert recovered == first
    assert first["candidate_version"].startswith("candidate-version-")
    assert first["experiment_id"].startswith("experiment-")
    assert first["dataset_version"] == "dataset-sha256-abc"
    assert first["code_version"] == "git-deadbeef"
    assert first["parameter_version"].startswith("params-")
    assert first["boundary"]["training_end"] < first["boundary"]["holdout_start"]


def test_parameter_change_appends_child_without_rewriting_parent(tmp_path):
    db_path = tmp_path / "trade.duckdb"
    review = _review(db_path)
    store = StrategyCandidateStore(db_path)
    values = _values(review["review_id"])
    parent = store.create_from_review(**values)

    values["parameters"] = {"fast": 8, "slow": 30}
    values["parent_candidate_id"] = parent["candidate_id"]
    child = store.create_from_review(**values)

    assert child["candidate_id"] != parent["candidate_id"]
    assert child["parent_candidate_id"] == parent["candidate_id"]
    assert store.get(parent["candidate_id"]) == parent
    assert [row["candidate_id"] for row in store.list_versions("ema_cross")] == [
        parent["candidate_id"],
        child["candidate_id"],
    ]


def test_candidate_requires_a_frozen_review(tmp_path):
    db_path = tmp_path / "trade.duckdb"
    EpisodeReviewStore(db_path)
    store = StrategyCandidateStore(db_path)

    with pytest.raises(ValueError, match="REVIEW_NOT_FOUND"):
        store.create_from_review(**_values("episode-review-missing"))


@pytest.mark.parametrize(
    "boundary,error",
    [
        (
            lambda: ExperimentBoundary(
                TRAINING_START,
                TRAINING_END,
                TRAINING_END,
                HOLDOUT_END,
            ),
            "OVERLAP",
        ),
        (
            lambda: ExperimentBoundary(
                TRAINING_START.replace(tzinfo=None),
                TRAINING_END,
                HOLDOUT_START,
                HOLDOUT_END,
            ),
            "TZ_REQUIRED",
        ),
    ],
)
def test_experiment_boundary_rejects_leakage_and_naive_times(boundary, error):
    with pytest.raises(ValueError, match=error):
        boundary()


def test_candidate_rejects_unfinished_holdout_and_invalid_parameters(tmp_path):
    db_path = tmp_path / "trade.duckdb"
    review = _review(db_path)
    store = StrategyCandidateStore(db_path)
    values = _values(review["review_id"])
    values["created_at"] = datetime(2026, 2, 1, tzinfo=timezone.utc)

    with pytest.raises(ValueError, match="HOLDOUT_NOT_COMPLETE"):
        store.create_from_review(**values)

    values = _values(review["review_id"])
    values["parameters"] = {"threshold": float("nan")}
    with pytest.raises(ValueError, match="PARAMETERS_INVALID"):
        store.create_from_review(**values)


def test_parent_must_exist_and_match_strategy(tmp_path):
    db_path = tmp_path / "trade.duckdb"
    review = _review(db_path)
    store = StrategyCandidateStore(db_path)
    values = _values(review["review_id"])
    values["parent_candidate_id"] = "strategy-candidate-missing"

    with pytest.raises(ValueError, match="PARENT_NOT_FOUND"):
        store.create_from_review(**values)

    values = _values(review["review_id"])
    parent = store.create_from_review(**values)
    values["strategy_name"] = "rsi_reversal"
    values["parent_candidate_id"] = parent["candidate_id"]
    with pytest.raises(ValueError, match="PARENT_STRATEGY_MISMATCH"):
        store.create_from_review(**values)
