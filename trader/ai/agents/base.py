"""
base.py
Agent 基类 + StubAgent（通用空实现）。

红线：agent 只产出 Advisory（status=DRAFT 的 TradePlan 包在 Advisory.payload 里）。
      绝不 import broker / order_manager / scheduler。
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from trader.contracts import AgentContext
from trader.models import Advisory, new_id, utc_now

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
    ) -> Advisory:
        return Advisory(
            advisory_id=new_id(),
            kind=kind,
            agent=self.role,
            payload=payload,
            confidence=confidence,
            model=model,
            created_at=utc_now(),
        )

    # ── LLM 调用工具（子类使用）─────────────────────────────────────────────

    def _llm_json(
        self,
        client: "LLMClient",
        system: str,
        user: str,
        model: str = "",
        temperature: float = 0.1,
        fallback: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """调用 LLM，返回 JSON dict；失败时返回 fallback。"""
        from trader.ai.client import LLMClient  # local import to avoid circular
        result = client.json_chat(system, user, model=model, temperature=temperature)
        if not result and fallback is not None:
            logger.warning("%s: LLM 返回空，使用 fallback", self.role)
            return fallback
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
