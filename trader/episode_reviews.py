"""Frozen, layered review artifacts with a stable error taxonomy."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb

_OUTCOMES = {
    "SUCCESS",
    "RISK_REJECTED",
    "NO_FILL",
    "DATA_FAILURE",
    "BROKER_FAILURE",
}
_ERROR_PREFIX = {
    "SUCCESS": "NONE",
    "RISK_REJECTED": "RISK_",
    "NO_FILL": "EXECUTION_",
    "DATA_FAILURE": "DATA_",
    "BROKER_FAILURE": "BROKER_",
}


class EpisodeReviewStore:
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
                CREATE TABLE IF NOT EXISTS episode_reviews (
                    review_id TEXT PRIMARY KEY,
                    subject_id TEXT,
                    outcome TEXT,
                    error_code TEXT,
                    facts_json TEXT,
                    decision_json TEXT,
                    execution_json TEXT,
                    result_json TEXT,
                    created_at TIMESTAMPTZ
                )
                """
            )
            connection.commit()
        finally:
            connection.close()

    def create(
        self,
        *,
        subject_id: str,
        outcome: str,
        error_code: str,
        facts: dict[str, Any],
        decision: dict[str, Any],
        execution: dict[str, Any],
        result: dict[str, Any],
        created_at: datetime,
    ) -> dict[str, Any]:
        normalized_outcome = outcome.strip().upper()
        normalized_error = error_code.strip().upper()
        if normalized_outcome not in _OUTCOMES:
            raise ValueError("REVIEW_OUTCOME_INVALID")
        if not subject_id.strip():
            raise ValueError("REVIEW_SUBJECT_REQUIRED")
        if created_at.tzinfo is None or created_at.utcoffset() is None:
            raise ValueError("REVIEW_TIME_TZ_REQUIRED")
        prefix = _ERROR_PREFIX[normalized_outcome]
        if (
            normalized_outcome == "SUCCESS"
            and normalized_error != "NONE"
        ) or (
            normalized_outcome != "SUCCESS"
            and not normalized_error.startswith(prefix)
        ):
            raise ValueError("REVIEW_ERROR_TAXONOMY_INVALID")
        layers = {
            "facts": self._layer(facts, "facts"),
            "decision": self._layer(decision, "decision"),
            "execution": self._layer(execution, "execution"),
            "result": self._layer(result, "result"),
        }
        payload = {
            "subject_id": subject_id,
            "outcome": normalized_outcome,
            "error_code": normalized_error,
            **layers,
        }
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        review_id = "episode-review-" + hashlib.sha256(
            canonical.encode()
        ).hexdigest()[:24]
        connection = self._connect()
        try:
            connection.execute(
                """
                INSERT INTO episode_reviews VALUES
                (?,?,?,?,?,?,?,?,?)
                ON CONFLICT (review_id) DO NOTHING
                """,
                [
                    review_id,
                    subject_id,
                    normalized_outcome,
                    normalized_error,
                    json.dumps(layers["facts"], sort_keys=True),
                    json.dumps(layers["decision"], sort_keys=True),
                    json.dumps(layers["execution"], sort_keys=True),
                    json.dumps(layers["result"], sort_keys=True),
                    created_at,
                ],
            )
            connection.commit()
        finally:
            connection.close()
        return self.get(review_id)

    def get(self, review_id: str) -> dict[str, Any] | None:
        connection = self._connect(read_only=True)
        try:
            row = connection.execute(
                "SELECT * FROM episode_reviews WHERE review_id=?",
                [review_id],
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        return {
            "review_id": str(row[0]),
            "subject_id": str(row[1]),
            "outcome": str(row[2]),
            "error_code": str(row[3]),
            "facts": json.loads(row[4]),
            "decision": json.loads(row[5]),
            "execution": json.loads(row[6]),
            "result": json.loads(row[7]),
            "created_at": row[8],
        }

    @staticmethod
    def _layer(value: dict[str, Any], name: str) -> dict[str, Any]:
        if not isinstance(value, dict) or not value:
            raise ValueError(f"REVIEW_{name.upper()}_REQUIRED")
        json.dumps(value, allow_nan=False, default=str)
        return value
