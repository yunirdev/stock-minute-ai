"""Deterministic safety gate for AI scores used by paper auto-trading."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

@dataclass(frozen=True)
class AIScoreSnapshot:
    symbol: str
    score: float | None          # 双极量表 0-100，50=中性，越高越坚定看多、越低越坚定看空——
                                  # 只用于给人看（"SELL · 15.0 分"这类展示），不用于任何门槛比较。
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
    contributor_count: int = 0
    weight_coverage: float = 0.0
    has_llm: bool = False
    fallback_count: int = 0
    recommendation: str = "HOLD"        # BUY | SELL | HOLD — AI bull/bear verdict
    complexity: str = ""                # HIGH | MEDIUM | LIGHT — daily_research.complexity_for_rank()
    confidence: float | None = None     # 方向无关的信心强度，0-1，越高越坚定（不论 BUY 还是
                                         # SELL）——所有"够不够格"的门槛判断都应该比这个，不是
                                         # 比双极的 score。来自 daily_research 每条已完成研究项
                                         # 本来就有的 confidence 字段，不是从 score 反推的。

@dataclass(frozen=True)
class AIScorePolicy:
    min_ai_score: float = 65.0   # 名字沿用旧配置项（.env/CLI 向后兼容），语义是"最低信心强度"
                                  # （0-100，对应 confidence*100），不是"最低双极分数"。
    max_age_minutes: float = 30.0
    min_contributors: int = 1
    min_weight_coverage: float = 0.0
    require_llm: bool = False

@dataclass(frozen=True)
class AIScoreValidationResult:
    valid: bool
    reason_code: str
    message: str
    age_seconds: float | None = None
    score: float | None = None
    confidence: float | None = None   # 0-1，方向无关——调用方需要"这个信号有多大把握"时用这个，
                                       # 不要用 score/100（那样一个高信心 SELL 会被读成低信心）。
    run_id: str | None = None

def conviction_of(snapshot: AIScoreSnapshot) -> float | None:
    """方向无关的信心强度（0-1）——所有"这个信号够不够格"的判断都该比这个，
    不是比双极的 score（SELL 天然锁在 [0,50]，直接跟一个高门槛比会永远输，
    跟信号强弱无关，纯粹是双极表示法的副作用）。

    优先用 snapshot 自带的 confidence（daily_research 每条已完成研究项本来
    就有，未经过双极变换，是最准的）；没带这个字段的旧数据/外部调用，按
    daily_research._score() 的反函数从 score 退回去算一次。两处都缺，返回
    None（调用方应按"信号不可用"处理，不能当成 0 分）。
    """
    if snapshot.confidence is not None:
        try:
            value = float(snapshot.confidence)
        except (TypeError, ValueError):
            return None
        return value if 0 <= value <= 1 else None
    if snapshot.score is None:
        return None
    try:
        score = float(snapshot.score)
    except (TypeError, ValueError):
        return None
    return abs(score - 50.0) / 50.0


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
        if snapshot.contributors is not None:
            if snapshot.contributor_count < self.policy.min_contributors:
                return AIScoreValidationResult(False, "AI_SCORE_CONTRIBUTORS_INSUFFICIENT", "AI contributor count is insufficient", age_seconds=age, score=score, run_id=snapshot.run_id)
            if snapshot.weight_coverage < self.policy.min_weight_coverage:
                return AIScoreValidationResult(False, "AI_SCORE_COVERAGE_INSUFFICIENT", "AI weight coverage is insufficient", age_seconds=age, score=score, run_id=snapshot.run_id)
            if self.policy.require_llm and not snapshot.has_llm:
                return AIScoreValidationResult(False, "AI_SCORE_LLM_MISSING", "A real LLM contributor is required", age_seconds=age, score=score, run_id=snapshot.run_id)
        # 强度判断用方向无关的 conviction_of()，不直接比双极 score（原因见该函数
        # 的注释）。score/created_at 都已经在上面验证过是合法值了，这里理论上不会
        # 再拿到 None/越界——保留检查只是不信任 snapshot 可能是外部 Mapping 拼出来的。
        conviction = conviction_of(snapshot)
        if conviction is None:
            return AIScoreValidationResult(False, "AI_SCORE_CONFIDENCE_INVALID", "AI confidence is missing or invalid", age_seconds=age, score=score, run_id=snapshot.run_id)
        if conviction * 100 < self.policy.min_ai_score:
            return AIScoreValidationResult(False, "AI_SCORE_BELOW_THRESHOLD", "AI confidence is below threshold", age_seconds=age, score=score, confidence=conviction, run_id=snapshot.run_id)
        return AIScoreValidationResult(True, "AI_SCORE_VALID", "AI score is valid", age_seconds=age, score=score, confidence=conviction, run_id=snapshot.run_id)
