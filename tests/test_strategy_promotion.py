from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from trader.episode_reviews import EpisodeReviewStore
from trader.strategy_candidates import ExperimentBoundary, StrategyCandidateStore
from trader.strategy_promotion import (
    PromotionPolicy,
    StrategyMetrics,
    StrategyPromotionStore,
)

TRAINING_START = datetime(2025, 1, 1, tzinfo=timezone.utc)
TRAINING_END = datetime(2025, 12, 31, tzinfo=timezone.utc)
HOLDOUT_START = datetime(2026, 1, 2, tzinfo=timezone.utc)
HOLDOUT_END = datetime(2026, 6, 30, tzinfo=timezone.utc)
CREATED = datetime(2026, 7, 1, 18, 0, tzinfo=timezone.utc)
EVALUATED = datetime(2026, 7, 20, 18, 0, tzinfo=timezone.utc)


def _candidate(db_path):
    review = EpisodeReviewStore(db_path).create(
        subject_id="episode-1",
        outcome="SUCCESS",
        error_code="NONE",
        facts={"snapshot_id": "snapshot-1"},
        decision={"strategy": "RSI震荡战法(60买40卖)"},
        execution={"fill_count": 4},
        result={"realized_pnl": -12.5},
        created_at=CREATED,
    )
    return StrategyCandidateStore(db_path).create_from_review(
        source_review_id=review["review_id"],
        strategy_name="RSI震荡战法(60买40卖)",
        base_strategy_version="production-ma-v3",
        dataset_version="holdout-dataset-v1",
        code_version="git-deadbeef",
        parameters={"rsi_n": 5},
        boundary=ExperimentBoundary(
            training_start=TRAINING_START,
            training_end=TRAINING_END,
            holdout_start=HOLDOUT_START,
            holdout_end=HOLDOUT_END,
        ),
        rationale="Validate a review-derived candidate without changing Runtime.",
        created_at=CREATED,
    )


def _bars(start, periods=120):
    timestamps = pd.date_range(start, periods=periods, freq="D", tz="UTC")
    wave = np.sin(np.arange(periods) / 5.0) * 8
    close = 100 + wave + np.arange(periods) * 0.03
    return pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "open": close - 0.2,
            "high": close + 1.5,
            "low": close - 1.5,
            "close": close,
            "volume": np.full(periods, 10_000),
        }
    )


def _paper_metrics(*, sessions=10, trades=6, net_return=0.08):
    return StrategyMetrics(
        net_return=net_return,
        sharpe=1.2,
        max_drawdown=0.08,
        trade_count=trades,
        win_rate=0.6,
        fees=12.0,
        slippage_cost=8.0,
        bar_count=500,
        session_count=sessions,
    )


def _policy():
    return PromotionPolicy(
        min_holdout_bars=100,
        min_holdout_trades=1,
        min_replay_bars=100,
        min_replay_trades=1,
        min_paper_sessions=5,
        min_paper_trades=3,
        min_sharpe=-20,
        max_drawdown=1,
        max_drawdown_worsening=0,
        min_fee_bps=5,
        min_slippage_bps=5,
    )


def _record_bar_evidence(store, candidate):
    holdout = store.evaluate_bars(
        candidate_id=candidate["candidate_id"],
        evidence_kind="HOLDOUT",
        evidence_version="holdout-engine-v1",
        dataset_version=candidate["dataset_version"],
        bars=_bars("2026-01-02"),
        timeframe="1d",
        champion_parameters={"rsi_n": 5},
        fee_bps=5,
        slippage_bps=5,
        created_at=EVALUATED,
    )
    replay = store.evaluate_bars(
        candidate_id=candidate["candidate_id"],
        evidence_kind="HISTORICAL_REPLAY",
        evidence_version="replay-engine-v1",
        dataset_version="replay-dataset-v1",
        bars=_bars("2024-01-02"),
        timeframe="1d",
        champion_parameters={"rsi_n": 5},
        fee_bps=5,
        slippage_bps=5,
        created_at=EVALUATED,
    )
    return holdout, replay


def test_bar_evaluation_freezes_holdout_replay_costs_and_drawdown(tmp_path):
    db_path = tmp_path / "trade.duckdb"
    candidate = _candidate(db_path)
    store = StrategyPromotionStore(db_path)

    holdout, replay = _record_bar_evidence(store, candidate)

    assert holdout["candidate_metrics"].bar_count == 120
    assert holdout["candidate_metrics"].trade_count >= 1
    assert holdout["candidate_metrics"].fees > 0
    assert holdout["candidate_metrics"].slippage_cost > 0
    assert 0 <= holdout["candidate_metrics"].max_drawdown <= 1
    assert replay["window_end"] < HOLDOUT_START
    assert store.get_comparison(holdout["comparison_id"]) == holdout


def test_full_evidence_promotes_idempotently_and_rolls_back(tmp_path):
    db_path = tmp_path / "trade.duckdb"
    candidate = _candidate(db_path)
    store = StrategyPromotionStore(db_path)
    _record_bar_evidence(store, candidate)
    store.record_comparison(
        candidate_id=candidate["candidate_id"],
        evidence_kind="PAPER",
        evidence_version="paper-run-10-sessions",
        dataset_version="paper-fills-v1",
        window_start=datetime(2026, 7, 2, tzinfo=timezone.utc),
        window_end=datetime(2026, 7, 15, tzinfo=timezone.utc),
        champion_version=candidate["base_strategy_version"],
        candidate_metrics=_paper_metrics(),
        champion_metrics=_paper_metrics(net_return=0.07),
        fee_bps=5,
        slippage_bps=5,
        created_at=EVALUATED,
    )

    promoted = store.decide(
        candidate["candidate_id"],
        policy=_policy(),
        created_at=EVALUATED + timedelta(hours=1),
    )
    duplicate = store.decide(
        candidate["candidate_id"],
        policy=_policy(),
        created_at=EVALUATED + timedelta(hours=2),
    )

    assert promoted["event_type"] == "PROMOTED"
    assert promoted["reasons"] == []
    assert promoted["rollback_version"] == candidate["base_strategy_version"]
    assert duplicate == promoted
    assert store.current_champion(candidate["strategy_name"]) == (
        candidate["candidate_version"]
    )

    rolled_back = store.rollback(
        promoted["event_id"],
        reason="Paper drawdown breached the post-promotion recovery condition.",
        created_at=EVALUATED + timedelta(days=1),
    )
    recovered = StrategyPromotionStore(db_path)

    assert rolled_back["event_type"] == "ROLLED_BACK"
    assert rolled_back["to_version"] == candidate["base_strategy_version"]
    assert recovered.current_champion(candidate["strategy_name"]) == (
        candidate["base_strategy_version"]
    )
    assert recovered.rollback(
        promoted["event_id"],
        reason="Paper drawdown breached the post-promotion recovery condition.",
        created_at=EVALUATED + timedelta(days=2),
    ) == rolled_back


def test_one_profitable_paper_trade_is_rejected_with_explicit_reasons(tmp_path):
    db_path = tmp_path / "trade.duckdb"
    candidate = _candidate(db_path)
    store = StrategyPromotionStore(db_path)
    store.record_comparison(
        candidate_id=candidate["candidate_id"],
        evidence_kind="PAPER",
        evidence_version="paper-single-trade",
        dataset_version="paper-fills-v1",
        window_start=datetime(2026, 7, 2, tzinfo=timezone.utc),
        window_end=datetime(2026, 7, 3, tzinfo=timezone.utc),
        champion_version=candidate["base_strategy_version"],
        candidate_metrics=_paper_metrics(
            sessions=1,
            trades=1,
            net_return=1.0,
        ),
        champion_metrics=_paper_metrics(
            sessions=1,
            trades=1,
            net_return=0.0,
        ),
        fee_bps=5,
        slippage_bps=5,
        created_at=EVALUATED,
    )

    decision = store.decide(
        candidate["candidate_id"],
        policy=_policy(),
        created_at=EVALUATED + timedelta(hours=1),
    )

    assert decision["event_type"] == "REJECTED"
    assert "MISSING_HOLDOUT" in decision["reasons"]
    assert "MISSING_HISTORICAL_REPLAY" in decision["reasons"]
    assert "PAPER_MIN_SESSIONS" in decision["reasons"]
    assert "PAPER_MIN_TRADES" in decision["reasons"]
    assert store.current_champion(candidate["strategy_name"]) is None


def test_holdout_and_replay_windows_fail_closed(tmp_path):
    db_path = tmp_path / "trade.duckdb"
    candidate = _candidate(db_path)
    store = StrategyPromotionStore(db_path)
    metrics = _paper_metrics()
    values = {
        "candidate_id": candidate["candidate_id"],
        "evidence_kind": "HOLDOUT",
        "evidence_version": "invalid-window",
        "dataset_version": candidate["dataset_version"],
        "window_start": TRAINING_START,
        "window_end": TRAINING_END,
        "champion_version": candidate["base_strategy_version"],
        "candidate_metrics": metrics,
        "champion_metrics": metrics,
        "fee_bps": 5,
        "slippage_bps": 5,
        "created_at": EVALUATED,
    }
    with pytest.raises(ValueError, match="HOLDOUT_WINDOW_INVALID"):
        store.record_comparison(**values)

    values["evidence_kind"] = "HISTORICAL_REPLAY"
    values["window_start"] = HOLDOUT_START
    values["window_end"] = HOLDOUT_END
    with pytest.raises(ValueError, match="REPLAY_HOLDOUT_OVERLAP"):
        store.record_comparison(**values)


def test_candidate_with_stale_champion_base_cannot_promote(tmp_path):
    db_path = tmp_path / "trade.duckdb"
    candidate = _candidate(db_path)
    store = StrategyPromotionStore(db_path)
    _record_bar_evidence(store, candidate)
    store.record_comparison(
        candidate_id=candidate["candidate_id"],
        evidence_kind="PAPER",
        evidence_version="paper-run-10-sessions",
        dataset_version="paper-fills-v1",
        window_start=datetime(2026, 7, 2, tzinfo=timezone.utc),
        window_end=datetime(2026, 7, 15, tzinfo=timezone.utc),
        champion_version=candidate["base_strategy_version"],
        candidate_metrics=_paper_metrics(),
        champion_metrics=_paper_metrics(net_return=0.07),
        fee_bps=5,
        slippage_bps=5,
        created_at=EVALUATED,
    )
    store.decide(
        candidate["candidate_id"],
        policy=_policy(),
        created_at=EVALUATED + timedelta(hours=1),
    )

    review = EpisodeReviewStore(db_path).create(
        subject_id="episode-2",
        outcome="SUCCESS",
        error_code="NONE",
        facts={"snapshot_id": "snapshot-2"},
        decision={"strategy": "RSI震荡战法(60买40卖)"},
        execution={"fill_count": 6},
        result={"realized_pnl": 20},
        created_at=CREATED,
    )
    stale = StrategyCandidateStore(db_path).create_from_review(
        source_review_id=review["review_id"],
        strategy_name="RSI震荡战法(60买40卖)",
        base_strategy_version="production-ma-v3",
        dataset_version="holdout-dataset-v1",
        code_version="git-cafebabe",
        parameters={"rsi_n": 5},
        boundary=ExperimentBoundary(
            TRAINING_START,
            TRAINING_END,
            HOLDOUT_START,
            HOLDOUT_END,
        ),
        rationale="This candidate was based on the superseded champion.",
        created_at=CREATED,
    )
    for kind, start, end, dataset in (
        (
            "HOLDOUT",
            HOLDOUT_START,
            HOLDOUT_END,
            stale["dataset_version"],
        ),
        (
            "HISTORICAL_REPLAY",
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 6, 1, tzinfo=timezone.utc),
            "replay-v2",
        ),
        (
            "PAPER",
            datetime(2026, 7, 2, tzinfo=timezone.utc),
            datetime(2026, 7, 15, tzinfo=timezone.utc),
            "paper-v2",
        ),
    ):
        store.record_comparison(
            candidate_id=stale["candidate_id"],
            evidence_kind=kind,
            evidence_version=f"{kind.lower()}-v2",
            dataset_version=dataset,
            window_start=start,
            window_end=end,
            champion_version=stale["base_strategy_version"],
            candidate_metrics=_paper_metrics(),
            champion_metrics=_paper_metrics(net_return=0.07),
            fee_bps=5,
            slippage_bps=5,
            created_at=EVALUATED,
        )

    rejected = store.decide(
        stale["candidate_id"],
        policy=_policy(),
        created_at=EVALUATED + timedelta(hours=2),
    )
    assert rejected["event_type"] == "REJECTED"
    assert "STALE_BASE_VERSION" in rejected["reasons"]
