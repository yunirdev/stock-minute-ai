"""Immutable strategy candidates produced from frozen episode reviews."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb


def _require_aware(value: datetime, code: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(code)


def _require_version(value: str, code: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(code)
    return normalized


@dataclass(frozen=True)
class ExperimentBoundary:
    """A non-overlapping train/holdout split frozen for one experiment."""

    training_start: datetime
    training_end: datetime
    holdout_start: datetime
    holdout_end: datetime

    def __post_init__(self) -> None:
        for value, code in (
            (self.training_start, "EXPERIMENT_TRAINING_START_TZ_REQUIRED"),
            (self.training_end, "EXPERIMENT_TRAINING_END_TZ_REQUIRED"),
            (self.holdout_start, "EXPERIMENT_HOLDOUT_START_TZ_REQUIRED"),
            (self.holdout_end, "EXPERIMENT_HOLDOUT_END_TZ_REQUIRED"),
        ):
            _require_aware(value, code)
        if self.training_start >= self.training_end:
            raise ValueError("EXPERIMENT_TRAINING_RANGE_INVALID")
        if self.training_end >= self.holdout_start:
            raise ValueError("EXPERIMENT_TRAINING_HOLDOUT_OVERLAP")
        if self.holdout_start >= self.holdout_end:
            raise ValueError("EXPERIMENT_HOLDOUT_RANGE_INVALID")


class StrategyCandidateStore:
    """Append-only candidate versions; this store cannot promote production."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._migrate()

    def _connect(self, *, read_only: bool = False):
        return duckdb.connect(self.db_path, read_only=read_only)

    def _migrate(self) -> None:
        connection = self._connect()
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS strategy_candidate_versions (
                    candidate_id TEXT PRIMARY KEY,
                    candidate_version TEXT,
                    experiment_id TEXT,
                    strategy_name TEXT,
                    base_strategy_version TEXT,
                    parent_candidate_id TEXT,
                    source_review_id TEXT,
                    dataset_version TEXT,
                    code_version TEXT,
                    parameter_version TEXT,
                    parameters_json TEXT,
                    training_start TIMESTAMPTZ,
                    training_end TIMESTAMPTZ,
                    holdout_start TIMESTAMPTZ,
                    holdout_end TIMESTAMPTZ,
                    rationale TEXT,
                    created_at TIMESTAMPTZ
                )
                """
            )
            connection.commit()
        finally:
            connection.close()

    def create_from_review(
        self,
        *,
        source_review_id: str,
        strategy_name: str,
        base_strategy_version: str,
        dataset_version: str,
        code_version: str,
        parameters: dict[str, Any],
        boundary: ExperimentBoundary,
        rationale: str,
        created_at: datetime,
        parent_candidate_id: str = "",
    ) -> dict[str, Any]:
        """Create a content-addressed candidate without changing production state."""
        review_id = _require_version(
            source_review_id,
            "STRATEGY_CANDIDATE_REVIEW_REQUIRED",
        )
        name = _require_version(
            strategy_name,
            "STRATEGY_CANDIDATE_NAME_REQUIRED",
        )
        base_version = _require_version(
            base_strategy_version,
            "STRATEGY_CANDIDATE_BASE_VERSION_REQUIRED",
        )
        data_version = _require_version(
            dataset_version,
            "STRATEGY_CANDIDATE_DATASET_VERSION_REQUIRED",
        )
        source_version = _require_version(
            code_version,
            "STRATEGY_CANDIDATE_CODE_VERSION_REQUIRED",
        )
        reason = rationale.strip()
        if not reason:
            raise ValueError("STRATEGY_CANDIDATE_RATIONALE_REQUIRED")
        _require_aware(created_at, "STRATEGY_CANDIDATE_TIME_TZ_REQUIRED")
        if created_at < boundary.holdout_end:
            raise ValueError("STRATEGY_CANDIDATE_HOLDOUT_NOT_COMPLETE")
        if not isinstance(parameters, dict):
            raise ValueError("STRATEGY_CANDIDATE_PARAMETERS_INVALID")
        try:
            parameters_json = json.dumps(
                parameters,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("STRATEGY_CANDIDATE_PARAMETERS_INVALID") from exc
        parameter_version = "params-" + hashlib.sha256(
            parameters_json.encode()
        ).hexdigest()[:20]
        experiment_payload = {
            "dataset_version": data_version,
            "code_version": source_version,
            "training_start": boundary.training_start.isoformat(),
            "training_end": boundary.training_end.isoformat(),
            "holdout_start": boundary.holdout_start.isoformat(),
            "holdout_end": boundary.holdout_end.isoformat(),
        }
        experiment_id = "experiment-" + self._digest(experiment_payload, 20)
        parent_id = parent_candidate_id.strip()

        connection = self._connect()
        try:
            review = connection.execute(
                """
                SELECT created_at FROM episode_reviews
                WHERE review_id=?
                """,
                [review_id],
            ).fetchone()
            if review is None:
                raise ValueError("STRATEGY_CANDIDATE_REVIEW_NOT_FOUND")
            if review[0] > created_at:
                raise ValueError("STRATEGY_CANDIDATE_REVIEW_FROM_FUTURE")
            if parent_id:
                parent = connection.execute(
                    """
                    SELECT strategy_name FROM strategy_candidate_versions
                    WHERE candidate_id=?
                    """,
                    [parent_id],
                ).fetchone()
                if parent is None:
                    raise ValueError("STRATEGY_CANDIDATE_PARENT_NOT_FOUND")
                if str(parent[0]) != name:
                    raise ValueError("STRATEGY_CANDIDATE_PARENT_STRATEGY_MISMATCH")

            payload = {
                "experiment_id": experiment_id,
                "strategy_name": name,
                "base_strategy_version": base_version,
                "parent_candidate_id": parent_id,
                "source_review_id": review_id,
                "dataset_version": data_version,
                "code_version": source_version,
                "parameter_version": parameter_version,
                "parameters": json.loads(parameters_json),
                "boundary": {
                    "training_start": boundary.training_start.isoformat(),
                    "training_end": boundary.training_end.isoformat(),
                    "holdout_start": boundary.holdout_start.isoformat(),
                    "holdout_end": boundary.holdout_end.isoformat(),
                },
                "rationale": reason,
            }
            candidate_hash = self._digest(payload, 24)
            candidate_id = "strategy-candidate-" + candidate_hash
            candidate_version = "candidate-version-" + candidate_hash[:16]
            connection.execute(
                """
                INSERT INTO strategy_candidate_versions (
                    candidate_id, candidate_version, experiment_id,
                    strategy_name, base_strategy_version, parent_candidate_id,
                    source_review_id, dataset_version, code_version,
                    parameter_version, parameters_json, training_start,
                    training_end, holdout_start, holdout_end, rationale,
                    created_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT (candidate_id) DO NOTHING
                """,
                [
                    candidate_id,
                    candidate_version,
                    experiment_id,
                    name,
                    base_version,
                    parent_id,
                    review_id,
                    data_version,
                    source_version,
                    parameter_version,
                    parameters_json,
                    boundary.training_start,
                    boundary.training_end,
                    boundary.holdout_start,
                    boundary.holdout_end,
                    reason,
                    created_at,
                ],
            )
            connection.commit()
        finally:
            connection.close()
        record = self.get(candidate_id)
        if record is None:  # pragma: no cover - defensive persistence guard
            raise RuntimeError("STRATEGY_CANDIDATE_PERSIST_FAILED")
        return record

    def get(self, candidate_id: str) -> dict[str, Any] | None:
        connection = self._connect(read_only=True)
        try:
            row = connection.execute(
                """
                SELECT * FROM strategy_candidate_versions
                WHERE candidate_id=?
                """,
                [candidate_id],
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        return {
            "candidate_id": str(row[0]),
            "candidate_version": str(row[1]),
            "experiment_id": str(row[2]),
            "strategy_name": str(row[3]),
            "base_strategy_version": str(row[4]),
            "parent_candidate_id": str(row[5] or ""),
            "source_review_id": str(row[6]),
            "dataset_version": str(row[7]),
            "code_version": str(row[8]),
            "parameter_version": str(row[9]),
            "parameters": json.loads(row[10]),
            "boundary": {
                "training_start": row[11],
                "training_end": row[12],
                "holdout_start": row[13],
                "holdout_end": row[14],
            },
            "rationale": str(row[15]),
            "created_at": row[16],
        }

    def list_versions(self, strategy_name: str) -> list[dict[str, Any]]:
        connection = self._connect(read_only=True)
        try:
            ids = connection.execute(
                """
                WITH RECURSIVE version_chain AS (
                    SELECT candidate_id, created_at, 1 AS depth
                    FROM strategy_candidate_versions
                    WHERE strategy_name=? AND parent_candidate_id=''
                    UNION ALL
                    SELECT child.candidate_id, child.created_at,
                           parent.depth + 1
                    FROM strategy_candidate_versions child
                    JOIN version_chain parent
                      ON child.parent_candidate_id=parent.candidate_id
                    WHERE child.strategy_name=?
                )
                SELECT candidate_id FROM version_chain
                ORDER BY depth, created_at, candidate_id
                """,
                [strategy_name, strategy_name],
            ).fetchall()
        finally:
            connection.close()
        return [
            record
            for candidate_id, in ids
            if (record := self.get(str(candidate_id))) is not None
        ]

    @staticmethod
    def _digest(payload: dict[str, Any], length: int) -> str:
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(canonical.encode()).hexdigest()[:length]
