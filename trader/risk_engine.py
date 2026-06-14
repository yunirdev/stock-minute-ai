"""
risk_engine.py
Pre-trade risk checks and real-time circuit breakers.

Flow:
    RiskEngine.evaluate(signal, equity, positions) -> RiskVerdict
    RiskEngine.check_equity(current_equity)         # daily DD circuit breaker
    RiskEngine.record_failure() / record_success()  # consecutive failure guard
"""
from __future__ import annotations

import logging
from typing import Dict

from .config import TradingConfig
from .models import Position, RiskVerdict, Side, Signal, TradePlan

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

    def evaluate(
        self,
        signal: Signal,
        current_equity: float,
        positions: Dict[str, Position],
    ) -> RiskVerdict:
        """
        Evaluate a signal against all risk rules.
        Returns RiskVerdict(approved=False, reason=...) to block, or
        RiskVerdict(approved=True, suggested_qty=...) to allow.
        """
        if self._halted:
            return RiskVerdict(False, "系统熔断中: " + self._halt_reason)

        # Short-selling guard
        if not self._cfg.risk.allow_short and signal.side == Side.SELL:
            pos = positions.get(signal.symbol)
            if pos is None or pos.qty <= 0:
                return RiskVerdict(False, "不允许裸空仓")

        price = signal.exec_price
        if not (price > 0):
            return RiskVerdict(False, f"执行价无效: {price}")

        # Position size: max_position_pct * equity
        max_value = current_equity * self._cfg.risk.max_position_pct
        qty_by_size = max_value / price

        # Risk-per-trade: max_trade_risk_pct * equity / (price * 1% stop)
        risk_value = current_equity * self._cfg.risk.max_trade_risk_pct
        qty_by_risk = risk_value / (price * 0.01)

        qty = min(qty_by_size, qty_by_risk) * self._cfg.leverage

        # For SELL: cap to actual position size (avoid over-selling / going short accidentally)
        if signal.side == Side.SELL:
            pos = positions.get(signal.symbol)
            held = pos.qty if pos is not None else 0.0
            qty = min(qty, held)

        # Apply integer rounding for whole-share brokers
        qty = max(int(qty), 0)

        if qty < 1:
            return RiskVerdict(
                False,
                f"建议仓位不足1股 (equity={current_equity:.0f}, price={price:.2f})",
            )

        return RiskVerdict(True, "通过", suggested_qty=float(qty))

    def evaluate_plan(
        self,
        plan: TradePlan,
        current_equity: float,
        positions: Dict[str, Position],
    ) -> RiskVerdict:
        """Plan-level pre-trade checks（用于 runtime.py 计划驱动管道）。"""
        if self._halted:
            return RiskVerdict(False, "系统熔断中: " + self._halt_reason)

        if plan.stop_loss <= 0:
            return RiskVerdict(False, "计划缺少有效止损")
        if plan.entry_price <= 0:
            return RiskVerdict(False, f"入场价无效: {plan.entry_price}")
        if plan.qty <= 0:
            return RiskVerdict(False, f"数量无效: {plan.qty}")

        if plan.side == Side.BUY and plan.entry_price <= plan.stop_loss:
            return RiskVerdict(
                False,
                f"BUY: 入场价({plan.entry_price:.2f}) ≤ 止损({plan.stop_loss:.2f})",
            )
        if plan.side == Side.SELL and plan.entry_price >= plan.stop_loss:
            return RiskVerdict(
                False,
                f"SELL: 入场价({plan.entry_price:.2f}) ≥ 止损({plan.stop_loss:.2f})",
            )

        cost = plan.entry_price * plan.qty
        max_cost = current_equity * self._cfg.risk.max_position_pct
        if cost > max_cost:
            return RiskVerdict(
                False,
                f"仓位成本 ${cost:,.0f} 超上限 ${max_cost:,.0f} "
                f"({self._cfg.risk.max_position_pct * 100:.0f}% 资产)",
            )

        if plan.action == "OPEN" and plan.symbol in positions:
            pos = positions[plan.symbol]
            if pos is not None and pos.qty > 0:
                return RiskVerdict(
                    False, f"{plan.symbol} 已有持仓 {pos.qty:.0f} 股，OPEN 计划被拒"
                )

        return RiskVerdict(True, "通过", suggested_qty=float(plan.qty))

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
