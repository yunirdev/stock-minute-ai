"""
risk_engine.py
Pre-trade risk checks and real-time circuit breakers.

Flow:
    RiskEngine.evaluate_plan(plan, equity, positions) -> RiskVerdict
    RiskEngine.check_equity(current_equity)         # daily DD circuit breaker
    RiskEngine.record_failure() / record_success()  # consecutive failure guard
"""
from __future__ import annotations

import logging
import math
from typing import Dict, Mapping

from .config import TradingConfig
from .models import Position, RiskVerdict, Side, TradePlan

logger = logging.getLogger(__name__)


class RiskEngine:

    def __init__(self, config: TradingConfig) -> None:
        self._cfg = config
        self._consecutive_failures: int = 0
        self._daily_start_equity: float | None = None
        self._halted: bool = False
        self._halt_reason: str = ""

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def set_daily_start(self, equity: float) -> None:
        """Call once at session open to calibrate the daily DD limit."""
        self._daily_start_equity = equity

    def record_success(self) -> None:
        self._consecutive_failures = 0

    def record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._cfg.risk.max_consecutive_failures:
            self._halt(
                f"连续下单失败 {self._consecutive_failures} 次，系统暂停"
            )

    def _halt(self, reason: str) -> None:
        self._halted = True
        self._halt_reason = reason
        logger.critical("🛑 RISK HALT: %s", reason)

    def reset_halt(self) -> None:
        """Manual override — operator must confirm before calling."""
        logger.warning("风控熔断已被手动重置")
        self._halted = False
        self._halt_reason = ""
        self._consecutive_failures = 0

    # ------------------------------------------------------------------
    # Real-time checks (called every tick)
    # ------------------------------------------------------------------

    def check_equity(self, current_equity: float) -> None:
        """Trigger daily drawdown circuit breaker if threshold exceeded."""
        if self._daily_start_equity is None:
            return
        dd = (current_equity - self._daily_start_equity) / self._daily_start_equity
        if dd <= -self._cfg.risk.daily_drawdown_limit_pct:
            self._halt(
                f"日内回撤 {dd * 100:.2f}% 触发熔断线 "
                f"({self._cfg.risk.daily_drawdown_limit_pct * 100:.1f}%)"
            )

    # ------------------------------------------------------------------
    # Pre-trade evaluation
    # ------------------------------------------------------------------

    def evaluate_plan(
        self,
        plan: TradePlan,
        current_equity: float,
        positions: Dict[str, Position],
        pending_buy_notional: Mapping[str, float] | None = None,
        buying_power: float | None = None,
    ) -> RiskVerdict:
        """Plan-level pre-trade checks（用于 runtime.py 计划驱动管道）。

        `buying_power` (equity × configured leverage, computed by the caller)
        sizes the position-value cap; it defaults to `current_equity` when
        omitted (no leverage). The per-trade stop-loss risk check below stays
        anchored to real `current_equity` regardless of leverage — leverage
        buys bigger size, not a bigger real-dollar loss tolerance per trade.
        """
        if self._halted:
            return RiskVerdict(False, "系统熔断中: " + self._halt_reason)

        try:
            entry_price = float(plan.entry_price)
            stop_loss = float(plan.stop_loss)
            qty = float(plan.qty)
            equity = float(current_equity)
        except (TypeError, ValueError):
            return RiskVerdict(False, "计划价格、数量或权益无效")
        if not all(
            math.isfinite(value)
            for value in (entry_price, stop_loss, qty, equity)
        ):
            return RiskVerdict(False, "计划价格、数量或权益无效")
        if equity <= 0:
            return RiskVerdict(False, f"账户权益无效: {equity}")
        if stop_loss <= 0:
            return RiskVerdict(False, "计划缺少有效止损")
        if entry_price <= 0:
            return RiskVerdict(False, f"入场价无效: {entry_price}")
        if qty <= 0:
            return RiskVerdict(False, f"数量无效: {qty}")

        if plan.side == Side.BUY and entry_price <= stop_loss:
            return RiskVerdict(
                False,
                f"BUY: 入场价({entry_price:.2f}) ≤ 止损({stop_loss:.2f})",
            )
        if plan.side == Side.SELL and entry_price >= stop_loss:
            return RiskVerdict(
                False,
                f"SELL: 入场价({entry_price:.2f}) ≥ 止损({stop_loss:.2f})",
            )

        increases_exposure = plan.action not in {"CLOSE", "REDUCE"}
        if (
            plan.side == Side.SELL
            and increases_exposure
            and not self._cfg.risk.allow_short
        ):
            held = positions.get(plan.symbol)
            if not held or float(held.qty) <= 0:
                return RiskVerdict(False, "做空未启用 (risk.allow_short=False)")
        if increases_exposure:
            try:
                max_trade_risk_pct = float(
                    self._cfg.risk.max_trade_risk_pct
                )
            except (TypeError, ValueError):
                return RiskVerdict(False, "单笔风险配置无效")
            if (
                not math.isfinite(max_trade_risk_pct)
                or max_trade_risk_pct <= 0
            ):
                return RiskVerdict(False, "单笔风险配置无效")
            trade_risk = abs(entry_price - stop_loss) * qty
            max_trade_risk = equity * max_trade_risk_pct
            if trade_risk > max_trade_risk + 1e-9:
                return RiskVerdict(
                    False,
                    f"单笔止损风险 ${trade_risk:,.2f} 超上限 "
                    f"${max_trade_risk:,.2f} "
                    f"({max_trade_risk_pct * 100:.2f}% 资产)",
                )

        cost = entry_price * qty
        sizing_base = float(buying_power) if buying_power is not None else equity
        max_cost = sizing_base * self._cfg.risk.max_position_pct
        increases_long = (
            plan.side == Side.BUY
            and increases_exposure
        )
        if increases_long:
            position = positions.get(plan.symbol)
            held_qty = max(float(position.qty), 0.0) if position else 0.0
            held_notional = held_qty * entry_price
            try:
                pending_notional = max(
                    float((pending_buy_notional or {}).get(plan.symbol, 0.0)),
                    0.0,
                )
            except (TypeError, ValueError):
                return RiskVerdict(False, "未成交买单敞口无效")
            if not math.isfinite(pending_notional):
                return RiskVerdict(False, "未成交买单敞口无效")
            cumulative_cost = held_notional + pending_notional + cost
            if cumulative_cost > max_cost + 1e-9:
                return RiskVerdict(
                    False,
                    f"累计仓位 ${cumulative_cost:,.0f} 超上限 "
                    f"${max_cost:,.0f} "
                    f"({self._cfg.risk.max_position_pct * 100:.0f}% 资产)",
                )
        elif increases_exposure and cost > max_cost:
            return RiskVerdict(
                False,
                f"仓位成本 ${cost:,.0f} 超上限 ${max_cost:,.0f} "
                f"({self._cfg.risk.max_position_pct * 100:.0f}% 资产)",
            )

        if plan.action == "OPEN" and plan.symbol in positions:
            pos = positions[plan.symbol]
            # 不分多空——已经有仓位（不管是多头还是空头）时 OPEN 计划都该被拒，
            # 该用的是 ADD/REDUCE。原来只查 qty>0，空头持仓（qty<0）时这道防线
            # 形同虚设。
            if pos is not None and pos.qty != 0:
                return RiskVerdict(
                    False, f"{plan.symbol} 已有持仓 {pos.qty:.0f} 股，OPEN 计划被拒"
                )

        return RiskVerdict(True, "通过", suggested_qty=qty)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_halted(self) -> bool:
        return self._halted

    @property
    def halt_reason(self) -> str:
        return self._halt_reason

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures
