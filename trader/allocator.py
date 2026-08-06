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
        buying_power: float | None = None,
    ) -> List[TradePlan]:
        """`buying_power` (equity × configured leverage) is the capital base used
        for sizing and the per-symbol cap; defaults to `equity` (no leverage).
        `plan.target_weight` stays expressed as a fraction of real `equity` —
        it can exceed 1.0 under leverage, which correctly signals the account
        is levered rather than hiding it behind a buying-power-relative number.
        """
        if not plans or equity <= 0:
            return plans
        sizing_base = float(buying_power) if buying_power else equity
        if sizing_base <= 0:
            return plans

        pending_buy_notional = pending_buy_notional or {}
        result: List[TradePlan] = []

        # 平仓/减仓（CLOSE/REDUCE）是在释放敞口，不是占用它——必须始终放行，
        # 不受下面的等权分配数量、单标的上限、购买力截断影响；qty 精确等于
        # 当前实际持仓，不能用 equal_w × notional 反推（那样完全可能算出
        # 超过实际持仓的卖出量，导致把平仓单误变成裸空，或被 broker 直接拒单）。
        closing_plans = [p for p in plans if p.action in {"CLOSE", "REDUCE"}]
        opening_plans = [p for p in plans if p.action not in {"CLOSE", "REDUCE"}]
        for plan in closing_plans:
            position = positions.get(plan.symbol)
            held_qty = max(float(position.qty), 0.0) if position else 0.0
            if held_qty <= 0:
                logger.info(
                    "allocator: %s 无持仓可平，跳过 %s", plan.symbol, plan.action
                )
                continue
            whole_qty = math.floor(held_qty)
            if whole_qty < 1:
                logger.warning(
                    "allocator: %s 持仓仅 %g 股（不足 1 股），限价单无法平掉，跳过",
                    plan.symbol, held_qty,
                )
                continue
            if held_qty - whole_qty > 1e-9:
                logger.warning(
                    "allocator: %s 持仓 %g 股，本次只能平掉 %d 股，剩余 %g 股"
                    "（限价单只接受整股）",
                    plan.symbol, held_qty, whole_qty, held_qty - whole_qty,
                )
            plan.qty = float(whole_qty)
            plan.target_weight = 0.0
            result.append(plan)
            logger.debug(
                "allocate close %s qty=%.4f entry=%.2f",
                plan.symbol, plan.qty, plan.entry_price,
            )

        # 按 confidence 降序截断（只对新开仓计划生效）
        sorted_plans = sorted(opening_plans, key=lambda p: p.confidence, reverse=True)
        active = sorted_plans[: self._max_plans]
        n = len(active)
        if n == 0:
            return result

        equal_w = min(1.0 / n, self._max_pct)
        total_notional = 0.0
        planned_buy_notional: Dict[str, float] = {}

        for plan in active:
            increases_long = plan.side == Side.BUY
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
                    sizing_base * self._max_pct - reserved_notional,
                    0.0,
                )
                order_notional = min(sizing_base * equal_w, remaining_notional)
                if order_notional <= 0:
                    logger.info(
                        "allocator: %s 累计仓位已达上限，跳过 BUY",
                        plan.symbol,
                    )
                    continue
                plan.target_weight = round(
                    (reserved_notional + order_notional) / equity,
                    4,
                )
                planned_buy_notional[plan.symbol] = (
                    planned_buy_notional.get(plan.symbol, 0.0) + order_notional
                )
            else:
                # 新开空单（SELL + OPEN，当前无持仓）
                order_notional = sizing_base * equal_w
                plan.target_weight = round(order_notional / equity, 4)

            if total_notional + order_notional > sizing_base + 1e-6:
                logger.info("allocator: 购买力不足，截断 %s", plan.symbol)
                break
            # 整股：自动交易强制走限价单，而 Alpaca 限价单只接受整股。之前这里
            # 算到小数点后 4 位，风控按小数 qty 审批，broker 提交时再取整——
            # 审批的数量和实际下单的数量根本不是同一个数。在这里就取整，让
            # 风控看到的就是最终会提交的数量。
            raw_qty = order_notional / max(plan.entry_price, 0.01)
            whole_qty = math.floor(raw_qty)
            if whole_qty < 1:
                logger.info(
                    "allocator: %s 分配到的金额 $%.2f 不足 1 股（现价 $%.2f），跳过",
                    plan.symbol, order_notional, plan.entry_price,
                )
                continue
            plan.qty = float(whole_qty)
            order_notional = whole_qty * plan.entry_price   # 按实际股数回算占用
            total_notional += order_notional
            result.append(plan)
            logger.debug(
                "allocate %s w=%.4f qty=%.4f entry=%.2f",
                plan.symbol, plan.target_weight, plan.qty, plan.entry_price,
            )

        return result
