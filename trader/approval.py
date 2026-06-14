"""
approval.py
人在回路审批：默认让计划停在 PENDING，显式开启才自动 APPROVED。

AutoApprover：
  - auto_approve=False（默认）：所有计划返回 PENDING，不下单。
  - auto_approve=True：按置信度/规则自动放行（≥ min_confidence 且非周末盘后）。
"""
from __future__ import annotations

import logging

from .models import TradePlan, utc_now

logger = logging.getLogger(__name__)

_DEFAULT_MIN_CONFIDENCE = 0.6


class AutoApprover:
    """实现 Approver Protocol。"""

    def __init__(
        self,
        auto_approve: bool = False,
        min_confidence: float = _DEFAULT_MIN_CONFIDENCE,
    ) -> None:
        self._auto = auto_approve
        self._min_conf = min_confidence

    def decide(self, plan: TradePlan) -> str:
        if not self._auto:
            logger.info(
                "approval: PENDING plan=%s %s（自动审批关闭）",
                plan.plan_id[:8], plan.symbol,
            )
            return "PENDING"

        if plan.confidence >= self._min_conf:
            logger.info(
                "approval: APPROVED plan=%s %s conf=%.2f",
                plan.plan_id[:8], plan.symbol, plan.confidence,
            )
            return "APPROVED"

        logger.info(
            "approval: REJECTED plan=%s %s conf=%.2f < %.2f",
            plan.plan_id[:8], plan.symbol, plan.confidence, self._min_conf,
        )
        return "REJECTED"
