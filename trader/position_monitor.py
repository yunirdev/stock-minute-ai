"""
position_monitor.py
盯盘守护：实时检查持仓是否触发止损/止盈，生成平仓计划。
"""
from __future__ import annotations

import logging
from typing import Dict, List

from .models import Bar, Position, Side, TradePlan, new_id, utc_now

logger = logging.getLogger(__name__)


class StopTakeProfitMonitor:
    """实现 PositionMonitor Protocol —— 止损/止盈触发生成 CLOSE 计划。"""

    def check(
        self,
        positions: Dict[str, Position],
        live_plans: Dict[str, TradePlan],
        latest: Dict[str, Bar],
    ) -> List[TradePlan]:
        triggered: List[TradePlan] = []
        for symbol, pos in positions.items():
            bar = latest.get(symbol)
            if bar is None:
                continue
            plan = live_plans.get(symbol)
            if plan is None:
                continue
            price = bar.close

            stop_hit = (pos.qty > 0 and price <= plan.stop_loss) or \
                       (pos.qty < 0 and price >= plan.stop_loss)
            tp_hit = (pos.qty > 0 and price >= plan.take_profit) or \
                     (pos.qty < 0 and price <= plan.take_profit)

            if stop_hit or tp_hit:
                reason = "止损" if stop_hit else "止盈"
                close_side = Side.SELL if pos.qty > 0 else Side.BUY
                close_plan = TradePlan(
                    plan_id=new_id(),
                    symbol=symbol,
                    side=close_side,
                    action="CLOSE",
                    entry_price=price,
                    stop_loss=price,
                    take_profit=price,
                    qty=abs(pos.qty),
                    confidence=1.0,
                    rationale=f"{reason}触发 price={price:.2f} "
                              f"stop={plan.stop_loss:.2f} tp={plan.take_profit:.2f}",
                    source="position_monitor",
                    status="APPROVED",  # 止损/止盈直接自动 APPROVED
                    created_at=utc_now(),
                    metadata={"trigger": reason, "original_plan_id": plan.plan_id},
                )
                triggered.append(close_plan)
                logger.info(
                    "⚠️  %s %s 触发 price=%.2f stop=%.2f tp=%.2f",
                    reason, symbol, price, plan.stop_loss, plan.take_profit,
                )
        return triggered
