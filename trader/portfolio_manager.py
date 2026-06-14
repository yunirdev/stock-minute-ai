"""
portfolio_manager.py
组合协调层：相关性/集中度检查（基础版 pass-through，留接口）。
"""
from __future__ import annotations

import logging
from typing import Dict, List

from .models import Position, TradePlan

logger = logging.getLogger(__name__)


class PassthroughPortfolioManager:
    """基础版：直接返回原计划，不做额外过滤（接口占位）。"""

    def reconcile(
        self,
        plans: List[TradePlan],
        positions: Dict[str, Position],
    ) -> List[TradePlan]:
        # 未来：相关性矩阵检查、集中度约束
        return plans
