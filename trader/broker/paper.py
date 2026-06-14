"""
broker/paper.py
Paper trading broker — fills instantly at the limit price with zero latency.
No network calls; safe to use without any API credentials.
"""
from __future__ import annotations

import logging
import uuid
from typing import Dict, List, Optional

from ..models import Fill, OrderIntent, OrderStatus, Position, Side, utc_now
from .base import BrokerAdapter

logger = logging.getLogger(__name__)


class PaperBroker(BrokerAdapter):
    """
    Simulated execution engine for paper trading.

    - All orders fill immediately at intent.limit_price (or close price).
    - Position tracking is managed by Portfolio; this adapter only
      generates Fill objects and records order state.
    """

    def __init__(self) -> None:
        self._orders: Dict[str, dict] = {}
        self._fills: Dict[str, Fill] = {}

    # ------------------------------------------------------------------
    # BrokerAdapter implementation
    # ------------------------------------------------------------------

    def place_order(self, intent: OrderIntent) -> str:
        broker_id = "PAPER-" + uuid.uuid4().hex[:8].upper()
        fill_price = intent.limit_price if intent.limit_price is not None else intent.reference_price
        if fill_price is None or fill_price <= 0:
            raise ValueError(f"PaperBroker requires a positive execution price for {intent.symbol}")

        fill = Fill(
            order_id=broker_id,
            intent_id=intent.intent_id,
            symbol=intent.symbol,
            side=intent.side,
            filled_qty=intent.qty,
            avg_price=fill_price,
            fill_time=utc_now(),
            fee=0.0,
        )
        self._fills[broker_id] = fill
        self._orders[broker_id] = {
            "status": OrderStatus.FILLED,
            "intent": intent,
        }
        logger.info(
            "📝 PAPER FILL  %s %s  qty=%.0f  @%.4f",
            intent.side.value, intent.symbol, intent.qty, fill_price,
        )
        return broker_id

    def cancel_order(self, broker_order_id: str) -> bool:
        order = self._orders.get(broker_order_id)
        if order and order["status"] not in (
            OrderStatus.FILLED, OrderStatus.CANCELLED
        ):
            order["status"] = OrderStatus.CANCELLED
            logger.info("PAPER CANCEL %s", broker_order_id)
            return True
        return False

    def get_order_status(self, broker_order_id: str) -> OrderStatus:
        order = self._orders.get(broker_order_id)
        return order["status"] if order else OrderStatus.FAILED

    def get_fill(self, broker_order_id: str) -> Optional[Fill]:
        return self._fills.get(broker_order_id)

    def get_positions(self) -> List[Position]:
        # Paper positions are owned by Portfolio; nothing extra here
        return []

    def get_account_equity(self) -> float:
        # Equity is tracked by Portfolio
        return 0.0
