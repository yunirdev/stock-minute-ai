"""
allocator.py
仓位分配：等权 + 单标的上限 + 现金约束，填 TradePlan.target_weight / .qty。
"""
from __future__ import annotations

import logging
from typing import Dict, List

from .models import Position, TradePlan

logger = logging.getLogger(__name__)

_DEFAULT_MAX_POSITION_PCT = 0.20   # 单标的最高 20% 组合权重
_DEFAULT_MAX_OPEN_PLANS = 10       # 最多同时处理计划数（保护）


class EqualWeightAllocator:
    """实现 Allocator Protocol —— 等权分配，满足总权重 ≤ 1、单标的 ≤ 上限。"""

    def __init__(
        self,
        max_position_pct: float = _DEFAULT_MAX_POSITION_PCT,
        max_open_plans: int = _DEFAULT_MAX_OPEN_PLANS,
    ) -> None:
        self._max_pct = max_position_pct
        self._max_plans = max_open_plans

    def allocate(
        self,
        plans: List[TradePlan],
        equity: float,
        positions: Dict[str, Position],
    ) -> List[TradePlan]:
        if not plans or equity <= 0:
            return plans

        # 按 confidence 降序截断
        sorted_plans = sorted(plans, key=lambda p: p.confidence, reverse=True)
        active = sorted_plans[: self._max_plans]
        n = len(active)

        equal_w = min(1.0 / n, self._max_pct)
        total_w = 0.0
        result: List[TradePlan] = []

        for plan in active:
            if total_w + equal_w > 1.0:
                logger.info("allocator: 现金不足，截断 %s", plan.symbol)
                break
            plan.target_weight = round(equal_w, 4)
            plan.qty = round((equity * equal_w) / max(plan.entry_price, 0.01), 4)
            total_w += equal_w
            result.append(plan)
            logger.debug(
                "allocate %s w=%.4f qty=%.4f entry=%.2f",
                plan.symbol, plan.target_weight, plan.qty, plan.entry_price,
            )

        return result
