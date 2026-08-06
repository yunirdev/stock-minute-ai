"""
position_monitor.py
盯盘守护：实时检查持仓是否触发止损/止盈，生成平仓计划。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

from .models import Bar, Position, Side, TradePlan, new_id, utc_now
from .plan import atr as _atr

logger = logging.getLogger(__name__)


class StopTakeProfitMonitor:
    """实现 PositionMonitor —— 止损/止盈触发生成 CLOSE 计划。"""

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
                    status="READY",  # 止损/止盈由 Runtime 立即执行
                    created_at=utc_now(),
                    metadata={"trigger": reason, "original_plan_id": plan.plan_id},
                )
                triggered.append(close_plan)
                logger.info(
                    "⚠️  %s %s 触发 price=%.2f stop=%.2f tp=%.2f",
                    reason, symbol, price, plan.stop_loss, plan.take_profit,
                )
        return triggered


@dataclass(frozen=True)
class TrailingStopCandidate:
    """一次"止损可以收紧"的判定结果——只描述收紧到哪，不产生订单。"""

    symbol: str
    side: Side
    new_stop_loss: float


class TrailingStopEvaluator:
    """追踪止损：持仓浮盈时把止损线往有利方向收紧，锁定部分利润。

    跟 StopTakeProfitMonitor 判断"现在要不要平仓"不同，这里判断"要不要把
    止损线挪一挪"——只收紧、绝不放松（跟 position_adjustments.py 里
    PositionAdjustmentEvaluator._new_stop() 的校验规则完全对齐：多头新止损
    必须严格落在 (旧止损, 止盈) 区间内，空头反过来），所以这里算出来的候选
    值一定能通过下游那道验证，不会出现"追踪止损自己被风控拒绝"的情况。

    ATR 复用 plan.py 里 ATRPlanner 开仓时用的同一个 atr()，入场和追踪止损
    是同一把波动率尺子，不会出现"开仓按一种口径算风险、追踪止损按另一种"
    的不一致。
    """

    def __init__(self, *, atr_period: int = 14, atr_multiplier: float = 1.5) -> None:
        self._period = atr_period
        self._k = atr_multiplier

    def evaluate(
        self,
        positions: Dict[str, Position],
        live_plans: Dict[str, TradePlan],
        raw_bars_map: Dict[str, list],
    ) -> List[TrailingStopCandidate]:
        candidates: List[TrailingStopCandidate] = []
        for symbol, pos in positions.items():
            if not pos.qty:
                continue
            plan = live_plans.get(symbol)
            raw = raw_bars_map.get(symbol)
            if plan is None or not raw or len(raw) < 2:
                continue
            closes = [float(b.close) for b in raw]
            highs = [float(b.high) for b in raw]
            lows = [float(b.low) for b in raw]
            atr_value = _atr(closes, highs, lows, period=self._period)
            latest_close = closes[-1]

            if plan.side == Side.BUY:
                candidate_stop = latest_close - self._k * atr_value
                tightens = candidate_stop > plan.stop_loss
                within_take_profit = candidate_stop < plan.take_profit
            else:
                candidate_stop = latest_close + self._k * atr_value
                tightens = candidate_stop < plan.stop_loss
                within_take_profit = candidate_stop > plan.take_profit

            if tightens and within_take_profit:
                candidates.append(
                    TrailingStopCandidate(
                        symbol=symbol,
                        side=plan.side,
                        new_stop_loss=round(candidate_stop, 4),
                    )
                )
        return candidates
