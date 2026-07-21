"""Deterministic safety gate for AI scores used by paper auto-trading."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

@dataclass(frozen=True)
class AIScoreSnapshot:
    symbol: str
    score: float | None
    created_at: datetime | None
    run_id: str | None = None
    provider: str | None = None
    model: str | None = None
    provenance: str | None = None
    source: str | None = None
    generated_by: str | None = None
    is_stub: bool = False
    schema_version: int = 1
    contributors: list[dict[str, Any]] | None = None

@dataclass(frozen=True)
class AIScorePolicy:
    min_ai_score: float = 65.0
    max_age_minutes: float = 30.0

@dataclass(frozen=True)
class AIScoreValidationResult:
    valid: bool
    reason_code: str
    message: str
    age_seconds: float | None = None
    score: float | None = None
    run_id: str | None = None

class AIScoreValidator:
    def __init__(self, policy: AIScorePolicy, now: Callable[[], datetime] | None = None) -> None:
        self.policy = policy
        self._now = now or (lambda: datetime.now(timezone.utc))

    def validate(self, snapshot: AIScoreSnapshot | Mapping[str, Any] | None) -> AIScoreValidationResult:
        if snapshot is None:
            return AIScoreValidationResult(False, "AI_SCORE_MISSING", "AI score is missing")
        if not isinstance(snapshot, AIScoreSnapshot):
            snapshot = AIScoreSnapshot(**{k: snapshot.get(k) for k in AIScoreSnapshot.__dataclass_fields__ if k in snapshot})
        if snapshot.score is None:
            return AIScoreValidationResult(False, "AI_SCORE_MISSING", "AI score is missing", run_id=snapshot.run_id)
        try:
            score = float(snapshot.score)
        except (TypeError, ValueError):
            return AIScoreValidationResult(False, "AI_SCORE_OUT_OF_RANGE", "AI score is not numeric", run_id=snapshot.run_id)
        if not 0 <= score <= 100:
            return AIScoreValidationResult(False, "AI_SCORE_OUT_OF_RANGE", "AI score must be between 0 and 100", score=score, run_id=snapshot.run_id)
        created = snapshot.created_at
        if created is None:
            return AIScoreValidationResult(False, "AI_SCORE_TIMESTAMP_MISSING", "AI score timestamp is missing", score=score, run_id=snapshot.run_id)
        if created.tzinfo is None:
            return AIScoreValidationResult(False, "AI_SCORE_TIMESTAMP_INVALID", "AI score timestamp must be timezone-aware", score=score, run_id=snapshot.run_id)
        age = (self._now().astimezone(timezone.utc) - created.astimezone(timezone.utc)).total_seconds()
        if age < 0 or age > self.policy.max_age_minutes * 60:
            return AIScoreValidationResult(False, "AI_SCORE_STALE", "AI score is stale", age_seconds=age, score=score, run_id=snapshot.run_id)
        if snapshot.is_stub:
            return AIScoreValidationResult(False, "AI_SCORE_STUB", "Stub AI scores cannot qualify for automatic trading", age_seconds=age, score=score, run_id=snapshot.run_id)
        if not snapshot.provider:
            return AIScoreValidationResult(False, "AI_SCORE_PROVIDER_MISSING", "AI provider is missing", age_seconds=age, score=score, run_id=snapshot.run_id)
        if not snapshot.model:
            return AIScoreValidationResult(False, "AI_SCORE_MODEL_MISSING", "AI model is missing", age_seconds=age, score=score, run_id=snapshot.run_id)
        if not snapshot.run_id or not (snapshot.provenance or snapshot.source or snapshot.generated_by):
            return AIScoreValidationResult(False, "AI_SCORE_PROVENANCE_MISSING", "AI score provenance is missing", age_seconds=age, score=score, run_id=snapshot.run_id)
        if score < self.policy.min_ai_score:
            return AIScoreValidationResult(False, "AI_SCORE_BELOW_THRESHOLD", "AI score is below threshold", age_seconds=age, score=score, run_id=snapshot.run_id)
        return AIScoreValidationResult(True, "AI_SCORE_VALID", "AI score is valid", age_seconds=age, score=score, run_id=snapshot.run_id)
