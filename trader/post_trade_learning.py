"""Runtime bridge from closed trade episodes to frozen reviews/candidates."""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import duckdb

from .episode_reviews import EpisodeReviewStore
from .strategy_candidates import (
    ExperimentBoundary,
    StrategyCandidateStore,
)


class PostTradeLearningCoordinator:
    """Create one immutable review and conservative candidate per closed episode."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self.reviews = EpisodeReviewStore(db_path)
        self.candidates = StrategyCandidateStore(db_path)

    def process(
        self,
        episode: dict[str, Any],
        *,
        created_at: datetime | None = None,
    ) -> dict[str, Any] | None:
        if str(episode.get("status", "")).upper() != "CLOSED":
            return None
        now = created_at or datetime.now(timezone.utc)
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("POST_TRADE_LEARNING_TIME_TZ_REQUIRED")
        episode_id = str(episode.get("episode_id", "")).strip()
        snapshot_id = str(episode.get("snapshot_id", "")).strip()
        if not episode_id or not snapshot_id:
            raise ValueError("POST_TRADE_LEARNING_EPISODE_REFERENCE_REQUIRED")
        existing = self._existing(episode_id)
        if existing is not None:
            return existing
        decision = self._decision_context(episode_id)
        review = self.reviews.create(
            subject_id=episode_id,
            outcome="SUCCESS",
            error_code="NONE",
            facts={
                "episode_snapshot_id": snapshot_id,
                "symbol": episode.get("symbol", ""),
                "first_fill_at": self._time_text(
                    episode.get("first_fill_at")
                ),
                "last_fill_at": self._time_text(
                    episode.get("last_fill_at")
                ),
            },
            decision={
                "decision_id": decision["decision_id"],
                "strategy": decision["strategy"],
                "strategy_version": decision["strategy_version"],
                "data_version": decision["data_version"],
            },
            execution={
                "fill_count": int(
                    (episode.get("attribution") or {}).get("fill_count", 0)
                ),
                "adverse_slippage": float(
                    episode.get("adverse_slippage", 0.0)
                ),
                "adjustment_count": int(
                    episode.get("adjustment_count", 0)
                ),
            },
            result={
                "realized_pnl": float(episode.get("realized_pnl", 0.0)),
                "open_quantity": float(episode.get("open_quantity", 0.0)),
                "strategy_invalidated": False,
            },
            created_at=now,
        )
        holdout_end = now - timedelta(seconds=1)
        holdout_start = holdout_end - timedelta(days=30)
        training_end = holdout_start - timedelta(seconds=1)
        training_start = training_end - timedelta(days=365)
        candidate = self.candidates.create_from_review(
            source_review_id=review["review_id"],
            strategy_name=decision["strategy"],
            base_strategy_version=decision["strategy_version"],
            dataset_version=decision["data_version"] or snapshot_id,
            code_version=os.getenv("APP_CODE_VERSION", "runtime-unversioned"),
            parameters={
                "proposal": "NO_PARAMETER_CHANGE_CONTROL",
                "source_episode_id": episode_id,
            },
            boundary=ExperimentBoundary(
                training_start=training_start,
                training_end=training_end,
                holdout_start=holdout_start,
                holdout_end=holdout_end,
            ),
            rationale=(
                "Control candidate generated from a frozen closed-episode review; "
                "independent holdout and Paper comparison remain mandatory."
            ),
            created_at=now,
        )
        return {"review": review, "candidate": candidate}

    def _existing(self, episode_id: str) -> dict[str, Any] | None:
        connection = duckdb.connect(self.db_path, read_only=True)
        try:
            row = connection.execute(
                """
                SELECT r.review_id, c.candidate_id
                FROM episode_reviews r
                LEFT JOIN strategy_candidate_versions c
                  ON c.source_review_id=r.review_id
                WHERE r.subject_id=?
                ORDER BY c.created_at, c.candidate_id
                LIMIT 1
                """,
                [episode_id],
            ).fetchone()
        finally:
            connection.close()
        if row is None or row[1] is None:
            return None
        return {
            "review": self.reviews.get(str(row[0])),
            "candidate": self.candidates.get(str(row[1])),
        }

    def _decision_context(self, episode_id: str) -> dict[str, str]:
        connection = duckdb.connect(self.db_path, read_only=True)
        try:
            row = connection.execute(
                """
                SELECT coalesce(d.decision_id, ''),
                       coalesce(d.strategy, 'runtime-production'),
                       coalesce(d.strategy_version, 'production-unknown'),
                       coalesce(d.data_version, '')
                FROM position_plan_versions p
                LEFT JOIN decision_plan_links l
                  ON l.plan_id=p.source_trade_plan_id
                LEFT JOIN strategy_decisions d
                  ON d.decision_id=l.decision_id
                WHERE p.position_plan_id=?
                ORDER BY p.version
                LIMIT 1
                """,
                [episode_id],
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return {
                "decision_id": "",
                "strategy": "runtime-production",
                "strategy_version": "production-unknown",
                "data_version": "",
            }
        return {
            "decision_id": str(row[0]),
            "strategy": str(row[1]),
            "strategy_version": str(row[2]),
            "data_version": str(row[3]),
        }

    @staticmethod
    def _time_text(value: Any) -> str:
        return value.isoformat() if isinstance(value, datetime) else str(value or "")
