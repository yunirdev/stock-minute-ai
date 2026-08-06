"""
plan.py
交易计划模块：给定 Candidate + 最新 Bar，生成 TradePlan(DRAFT)。

基础版：
  entry  = 最新收盘价（或回踩价）
  stop   = entry - k * ATR14
  tp     = entry + r * k * ATR14（r 默认 2，即 2:1 盈亏比）
  action 由是否已持仓决定（OPEN / ADD / REDUCE / CLOSE）
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np

from .models import Bar, Candidate, Side, TradePlan, new_id, utc_now

logger = logging.getLogger(__name__)

_DEFAULT_PARAMS: Dict[str, Any] = {
    "atr_period": 14,
    "atr_multiplier": 1.5,   # k
    "rr_ratio": 2.0,          # r（止盈 = k*r*ATR）
}


def atr(bars_close: "list[float]", bars_high: "list[float]",
        bars_low: "list[float]", period: int = 14) -> float:
    """简单 ATR 计算（True Range 均值）。

    公开导出（原来叫 `_atr`）——追踪止损（position_monitor.py 的
    TrailingStopEvaluator）复用同一套计算，跟入场时算 stop/tp 用的是
    同一把尺子，不会出现"入场用一种波动率口径、追踪止损用另一种"的不一致。
    """
    if len(bars_close) < period + 1:
        return float(np.std(bars_close[-period:]) or 1.0)
    trs = []
    for i in range(1, min(period + 1, len(bars_close))):
        tr = max(
            bars_high[-i] - bars_low[-i],
            abs(bars_high[-i] - bars_close[-i - 1]),
            abs(bars_low[-i] - bars_close[-i - 1]),
        )
        trs.append(tr)
    return float(np.mean(trs)) if trs else 1.0


class ATRPlanner:
    """实现 Planner —— 基于 ATR 的入场/止损/止盈规划。"""

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        self._params = {**_DEFAULT_PARAMS, **(params or {})}

    def make_plan(
        self,
        cand: Candidate,
        latest_bar: Bar,
        params: Dict[str, Any] | None = None,
        *,
        side: Side,
        current_qty: float = 0.0,
        bars_history: "list[Bar] | None" = None,
    ) -> TradePlan:
        p = {**self._params, **(params or {})}
        k = float(p.get("atr_multiplier", 1.5))
        rr = float(p.get("rr_ratio", 2.0))
        atr_period = int(p.get("atr_period", 14))

        # 有足够历史 bar 时算真正的 ATR(atr_period)；否则退化为单 bar range
        # （之前 atr_period 这个配置项读了却完全没用上，stop/tp 一直是用单
        #   bar 的 high-low 估的，比真实 ATR14 噪声大很多）。
        if bars_history and len(bars_history) >= 2:
            closes = [b.close for b in bars_history] + [latest_bar.close]
            highs  = [b.high  for b in bars_history] + [latest_bar.high]
            lows   = [b.low   for b in bars_history] + [latest_bar.low]
            atr_value = atr(closes, highs, lows, period=atr_period)
        else:
            # 用单 bar 的 range 做简化 ATR（无历史 bars 时 fallback）
            atr_value = max(latest_bar.high - latest_bar.low, latest_bar.close * 0.01)

        side = Side(side)
        entry = latest_bar.close

        # current_qty 是有符号的（正=多头，负=空头，见 Position.qty 的约定）。
        # 原来 SELL 那支只判断 current_qty>0（已有多头→REDUCE，否则一律 OPEN），
        # 把"空仓"和"已经持有空头，这次是加空"两种情况混成了同一个 OPEN——
        # 已经有空头时再来一次 OPEN 会被 PositionPlanFillProjector 当成开新仓
        # 处理，跟真实持仓对不上。BUY 那支同理需要照顾"已经持有空头，这次是
        # 回补减仓"的情况。
        if side == Side.BUY:
            stop = entry - k * atr_value
            tp = entry + rr * k * atr_value
            if current_qty > 0:
                action = "ADD"
            elif current_qty < 0:
                action = "REDUCE"   # 已持有空头，BUY 是回补减仓，不是开新仓
            else:
                action = "OPEN"
        else:
            stop = entry + k * atr_value
            tp = entry - rr * k * atr_value
            if current_qty < 0:
                action = "ADD"      # 已持有空头，SELL 是加空
            elif current_qty > 0:
                action = "REDUCE"   # 已持有多头，SELL 是减仓/平仓
            else:
                action = "OPEN"

        rationale = (
            f"side={side.value} score={cand.score:.1f} entry={entry:.2f} "
            f"stop={stop:.2f} tp={tp:.2f} ATR≈{atr_value:.4f}"
        )

        return TradePlan(
            plan_id=new_id(),
            symbol=cand.symbol,
            side=side,
            action=action,
            entry_price=round(entry, 4),
            stop_loss=round(stop, 4),
            take_profit=round(tp, 4),
            confidence=cand.score / 100.0,
            rationale=rationale,
            source="consensus",
            status="DRAFT",
            created_at=utc_now(),
            metadata={"candidate_reasons": cand.reasons},
        )
