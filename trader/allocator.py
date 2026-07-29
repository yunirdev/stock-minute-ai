"""
allocator.py
仓位分配：等权 + 单标的上限 + 现金约束，填 TradePlan.target_weight / .qty。
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Mapping

from .models import Position, Side, TradePlan

logger = logging.getLogger(__name__)

_DEFAULT_MAX_POSITION_PCT = 0.20   # 单标的最高 20% 组合权重
_DEFAULT_MAX_OPEN_PLANS = 10       # 最多同时处理计划数（保护）


class EqualWeightAllocator:
    """实现 Allocator —— 等权分配，满足总权重 ≤ 1、单标的 ≤ 上限。"""

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
        pending_buy_notional: Mapping[str, float] | None = None,
    ) -> List[TradePlan]:
        if not plans or equity <= 0:
            return plans

        pending_buy_notional = pending_buy_notional or {}

        # 按 confidence 降序截断
        sorted_plans = sorted(plans, key=lambda p: p.confidence, reverse=True)
        active = sorted_plans[: self._max_plans]
        n = len(active)

        equal_w = min(1.0 / n, self._max_pct)
        total_w = 0.0
        planned_buy_notional: Dict[str, float] = {}
        result: List[TradePlan] = []

        for plan in active:
            order_weight = equal_w
            increases_long = (
                plan.side == Side.BUY
                and plan.action not in {"CLOSE", "REDUCE"}
            )
            if increases_long:
                position = positions.get(plan.symbol)
                held_qty = max(float(position.qty), 0.0) if position else 0.0
                held_notional = held_qty * plan.entry_price
                pending_notional = max(
                    float(pending_buy_notional.get(plan.symbol, 0.0)), 0.0
                )
                reserved_notional = (
                    held_notional
                    + pending_notional
                    + planned_buy_notional.get(plan.symbol, 0.0)
                )
                remaining_notional = max(
                    equity * self._max_pct - reserved_notional,
                    0.0,
                )
                order_notional = min(equity * equal_w, remaining_notional)
                if order_notional <= 0:
                    logger.info(
                        "allocator: %s 累计仓位已达上限，跳过 BUY",
                        plan.symbol,
                    )
                    continue
                order_weight = order_notional / equity
                plan.target_weight = round(
                    (reserved_notional + order_notional) / equity,
                    4,
                )
                planned_buy_notional[plan.symbol] = (
                    planned_buy_notional.get(plan.symbol, 0.0) + order_notional
                )
            else:
                order_notional = equity * order_weight
                plan.target_weight = round(order_weight, 4)

            if total_w + order_weight > 1.0:
                logger.info("allocator: 现金不足，截断 %s", plan.symbol)
                break
            raw_qty = order_notional / max(plan.entry_price, 0.01)
            plan.qty = (
                math.floor(raw_qty * 10_000) / 10_000
                if increases_long
                else round(raw_qty, 4)
            )
            total_w += order_weight
            result.append(plan)
            logger.debug(
                "allocate %s w=%.4f qty=%.4f entry=%.2f",
                plan.symbol, plan.target_weight, plan.qty, plan.entry_price,
            )

        return result
