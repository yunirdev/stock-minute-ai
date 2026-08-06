"""
base.py
Agent 基类 + StubAgent（通用空实现）。

红线：agent 只产出 Advisory（status=DRAFT 的 TradePlan 包在 Advisory.payload 里）。
      绝不 import broker / order execution。
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from trader.models import AgentContext
from trader.models import Advisory, new_id, utc_now

if TYPE_CHECKING:
    from trader.ai.client import LLMClient

logger = logging.getLogger(__name__)


class AgentBase(ABC):
    """所有 agent 的抽象基类。"""

    role: str = "base"

    @abstractmethod
    def run(self, ctx: AgentContext) -> List[Advisory]: ...

    # ── 工厂方法 ─────────────────────────────────────────────────────────────

    def _advisory(
        self,
        kind: str,
        payload: Dict[str, Any],
        confidence: float = 0.0,
        model: str = "",
        is_fallback: bool | None = None,
    ) -> Advisory:
        """`is_fallback` 必须由调用方显式传入 `_llm_json()` 结果里的
        `_is_fallback`。

        原来这里只从 payload 里读 `_is_fallback`，但所有 agent 都是另起一个
        全新的 dict 当 payload（不会把 `_is_fallback` 带过去），导致这个标记
        对每个 agent 永远是 False —— 而 get_score_snapshots_from_db 正是靠它
        把 LLM 不可用时的硬编码降级分排除在加权综合分之外。结果就是 Ollama
        挂掉时，一堆硬编码的 50 分被当成真实 LLM 结论算进综合分。
        """
        if is_fallback is None:
            is_fallback = bool(payload.get("_is_fallback", False))
        return Advisory(
            advisory_id=new_id(),
            kind=kind,
            agent=self.role,
            payload=payload,
            confidence=confidence,
            model=model,
            created_at=utc_now(),
            is_fallback=bool(is_fallback),
        )

    @staticmethod
    def _is_fallback_result(*results: Dict[str, Any]) -> bool:
        """只要有一次 LLM 调用降级了，整条 advisory 就算降级产物。"""
        return any(bool(r.get("_is_fallback")) for r in results if isinstance(r, dict))

    # ── LLM 调用工具（子类使用）─────────────────────────────────────────────

    def _llm_json(
        self,
        client: LLMClient,
        system: str,
        user: str,
        model: str = "",
        temperature: float = 0.1,
        fallback: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """调用 LLM，返回 JSON dict；失败时返回 fallback。"""
        result = client.json_chat(system, user, model=model, temperature=temperature)
        if not result and fallback is not None:
            logger.warning("%s: LLM 返回空，使用 fallback", self.role)
            result = dict(fallback)
            result["_is_fallback"] = True
            return result
        return result or {}

    def _clamp_score(self, value: Any, default: float = 50.0) -> float:
        """把 LLM 返回的 score 夹到 [0, 100]。"""
        try:
            return max(0.0, min(100.0, float(value)))
        except (TypeError, ValueError):
            return default


class StubAgent(AgentBase):
    """通用占位 agent，始终返回空列表。"""

    def __init__(self, role: str = "stub") -> None:
        self.role = role

    def run(self, ctx: AgentContext) -> List[Advisory]:
        logger.debug("StubAgent[%s] called (no-op)", self.role)
        return []
