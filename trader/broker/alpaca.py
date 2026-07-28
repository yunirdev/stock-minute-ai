"""
broker/alpaca.py
Alpaca 执行适配器（默认 paper 虚拟盘）。

封装 alpaca-py 的 TradingClient。`paper=True` 由 SDK 锁定到 paper-api.alpaca.markets
（虚拟盘、真实行情、不碰真钱）；实盘需显式 `paper=False`（且 broker_type=alpaca_live）。

Alpaca order fills are asynchronous.
place_order 只提交订单并返回 broker order id；是否成交由 Runtime 轮询
get_order_status / get_fill 获取（Alpaca 用真实行情撮合，可能排队、部分成交）。
"""
from __future__ import annotations

import logging
from typing import List, Optional

from ..models import Fill, OrderIntent, OrderStatus, Position, Side, utc_now

logger = logging.getLogger(__name__)

# Alpaca 订单状态 → 平台统一 OrderStatus
_STATUS_MAP = {
    "filled": OrderStatus.FILLED,
    "partially_filled": OrderStatus.PARTIAL,
    "new": OrderStatus.SUBMITTED,
    "accepted": OrderStatus.SUBMITTED,
    "pending_new": OrderStatus.PENDING,
    "accepted_for_bidding": OrderStatus.SUBMITTED,
    "done_for_day": OrderStatus.SUBMITTED,
    "calculated": OrderStatus.SUBMITTED,
    "pending_cancel": OrderStatus.SUBMITTED,
    "pending_replace": OrderStatus.SUBMITTED,
    "suspended": OrderStatus.PENDING,
    "canceled": OrderStatus.CANCELLED,
    "expired": OrderStatus.CANCELLED,
    "replaced": OrderStatus.CANCELLED,
    "rejected": OrderStatus.REJECTED,
}


def _map_status(raw) -> OrderStatus:
    s = str(getattr(raw, "value", raw)).lower()
    return _STATUS_MAP.get(s, OrderStatus.SUBMITTED)


def _side_of(raw) -> Side:
    return Side.BUY if str(getattr(raw, "value", raw)).lower() == "buy" else Side.SELL


class AlpacaBroker:
    """Alpaca 执行适配器。默认 paper（虚拟盘）。"""

    def __init__(self, api_key: str, secret_key: str, paper: bool = True) -> None:
        if not api_key or not secret_key:
            raise ValueError(
                "AlpacaBroker 需要 ALPACA_API_KEY / ALPACA_API_SECRET（请填入 .env）")
        from alpaca.trading.client import TradingClient  # 延迟导入，未装时不影响其它 broker
        self._client = TradingClient(api_key, secret_key, paper=paper)
        self._paper = paper
        logger.info("AlpacaBroker 已连接 (%s)", "PAPER 虚拟盘" if paper else "LIVE 实盘")

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------

    def place_order(self, intent: OrderIntent) -> str:
        if not self._paper:
            raise RuntimeError("LIVE_ORDER_SUBMISSION_DISABLED")
        if intent.order_type != "LMT":
            raise ValueError("AUTOMATIC_EXECUTION_REQUIRES_LMT")

        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import LimitOrderRequest

        side = OrderSide.BUY if intent.side == Side.BUY else OrderSide.SELL
        qty = max(int(intent.qty), 1)
        tif = TimeInForce.DAY

        px = intent.limit_price if intent.limit_price else intent.reference_price
        if not px or px <= 0:
            raise ValueError(f"LMT 单需要有效 limit_price: {intent.symbol}")
        req = LimitOrderRequest(
            symbol=intent.symbol,
            qty=qty,
            side=side,
            time_in_force=tif,
            limit_price=round(float(px), 2),
            client_order_id=intent.client_order_id or None,
        )

        order = self._client.submit_order(req)
        bid = str(order.id)
        logger.info("📨 ALPACA 提交 %s %s qty=%d type=%s id=%s",
                    intent.side.value, intent.symbol, qty, intent.order_type, bid)
        return bid

    def cancel_order(self, broker_order_id: str) -> bool:
        try:
            self._client.cancel_order_by_id(broker_order_id)
            return True
        except Exception as exc:
            logger.warning("ALPACA 撤单失败 %s: %s", broker_order_id, exc)
            return False

    def get_order_status(self, broker_order_id: str) -> OrderStatus:
        try:
            o = self._client.get_order_by_id(broker_order_id)
            return _map_status(o.status)
        except Exception as exc:
            logger.warning("ALPACA 查询订单失败 %s: %s", broker_order_id, exc)
            return OrderStatus.FAILED

    def get_fill(self, broker_order_id: str) -> Optional[Fill]:
        try:
            o = self._client.get_order_by_id(broker_order_id)
        except Exception as exc:
            logger.warning("ALPACA 查询成交失败 %s: %s", broker_order_id, exc)
            return None
        filled_qty = float(o.filled_qty or 0)
        if filled_qty <= 0:
            return None
        return Fill(
            order_id=str(o.id), intent_id="", symbol=o.symbol, side=_side_of(o.side),
            filled_qty=filled_qty, avg_price=float(o.filled_avg_price or 0),
            fill_time=getattr(o, "filled_at", None) or utc_now(), fee=0.0,
            broker_payload={"alpaca_status": str(getattr(o.status, "value", o.status))},
        )

    def get_open_orders(self) -> List[dict]:
        try:
            orders = self._client.get_orders()
            return [{"id": str(o.id), "client_order_id": getattr(o, "client_order_id", None), "status": str(getattr(o.status, "value", o.status))} for o in orders]
        except Exception as exc:
            logger.warning("ALPACA open order reconciliation failed: %s", type(exc).__name__)
            raise

    def get_recent_fills(self) -> List[Fill]:
        try:
            orders = self._client.get_orders()
            fills = []
            for o in orders:
                if float(getattr(o, "filled_qty", 0) or 0) > 0:
                    fill = self.get_fill(str(o.id))
                    if fill:
                        fills.append(fill)
            return fills
        except Exception as exc:
            logger.warning("ALPACA fill reconciliation failed: %s", type(exc).__name__)
            raise

    def get_positions(self) -> List[Position]:
        out: List[Position] = []
        try:
            for p in self._client.get_all_positions():
                out.append(Position(
                    symbol=p.symbol, qty=float(p.qty),
                    avg_entry_px=float(p.avg_entry_price),
                    unrealized_pnl=float(getattr(p, "unrealized_pl", 0) or 0),
                ))
        except Exception as exc:
            logger.warning(
                "ALPACA position reconciliation failed: %s",
                type(exc).__name__,
            )
            raise
        return out

    def get_account_cash(self) -> float:
        try:
            account = self._client.get_account()
            return float(account.cash)
        except Exception as exc:
            logger.warning(
                "ALPACA cash reconciliation failed: %s",
                type(exc).__name__,
            )
            raise

    def get_account_equity(self) -> float:
        try:
            acct = self._client.get_account()
            return float(acct.equity)
        except Exception as exc:
            logger.warning("ALPACA 查询权益失败: %s", exc)
            return 0.0
