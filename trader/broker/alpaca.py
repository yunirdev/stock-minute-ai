"""
broker/alpaca.py
Alpaca æ‰§è¡Œé€‚é…å™¨ï¼ˆé»˜è®¤ paper è™šæ‹Ÿç›˜ï¼‰ã€‚

å°è£… alpaca-py çš„ TradingClientã€‚`paper=True` ç”± SDK é”å®šåˆ° paper-api.alpaca.markets
ï¼ˆè™šæ‹Ÿç›˜ã€çœŸå®žè¡Œæƒ…ã€ä¸ç¢°çœŸé’±ï¼‰ï¼›å®žç›˜éœ€æ˜¾å¼ `paper=False`ï¼ˆä¸” broker_type=alpaca_liveï¼‰ã€‚

Alpaca order fills are asynchronous.
place_order åªæäº¤è®¢å•å¹¶è¿”å›ž broker order idï¼›æ˜¯å¦æˆäº¤ç”± scheduler è½®è¯¢
get_order_status / get_fill èŽ·å–ï¼ˆAlpaca ç”¨çœŸå®žè¡Œæƒ…æ’®åˆï¼Œå¯èƒ½æŽ’é˜Ÿã€éƒ¨åˆ†æˆäº¤ï¼‰ã€‚
"""
from __future__ import annotations

import logging
from typing import List, Optional

from ..models import Fill, OrderIntent, OrderStatus, Position, Side, utc_now
from .base import BrokerAdapter

logger = logging.getLogger(__name__)

# Alpaca è®¢å•çŠ¶æ€ â†’ å¹³å°ç»Ÿä¸€ OrderStatus
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


class AlpacaBroker(BrokerAdapter):
    """Alpaca æ‰§è¡Œé€‚é…å™¨ã€‚é»˜è®¤ paperï¼ˆè™šæ‹Ÿç›˜ï¼‰ã€‚å®žçŽ° BrokerAdapter å…¨éƒ¨æŽ¥å£ã€‚"""

    def __init__(self, api_key: str, secret_key: str, paper: bool = True) -> None:
        if not api_key or not secret_key:
            raise ValueError(
                "AlpacaBroker éœ€è¦ ALPACA_API_KEY / ALPACA_API_SECRETï¼ˆè¯·å¡«å…¥ .envï¼‰")
        from alpaca.trading.client import TradingClient  # å»¶è¿Ÿå¯¼å…¥ï¼Œæœªè£…æ—¶ä¸å½±å“å…¶å®ƒ broker
        self._client = TradingClient(api_key, secret_key, paper=paper)
        self._paper = paper
        logger.info("AlpacaBroker å·²è¿žæŽ¥ (%s)", "PAPER è™šæ‹Ÿç›˜" if paper else "LIVE å®žç›˜")

    # ------------------------------------------------------------------
    # BrokerAdapter implementation
    # ------------------------------------------------------------------

    def place_order(self, intent: OrderIntent) -> str:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

        side = OrderSide.BUY if intent.side == Side.BUY else OrderSide.SELL
        qty = max(int(intent.qty), 1)
        tif = TimeInForce.DAY

        # çº¢çº¿ï¼šé»˜è®¤åªä¸‹ LMTï¼›MKT ä»…åœ¨ intent æ˜Žç¡®è¦æ±‚ä¸”æ— é™ä»·æ—¶
        if intent.order_type == "MKT" and intent.limit_price is None:
            req = MarketOrderRequest(symbol=intent.symbol, qty=qty, side=side, time_in_force=tif)
        else:
            px = intent.limit_price if intent.limit_price else intent.reference_price
            if not px or px <= 0:
                raise ValueError(f"LMT å•éœ€è¦æœ‰æ•ˆ limit_price: {intent.symbol}")
            req = LimitOrderRequest(symbol=intent.symbol, qty=qty, side=side,
                                    time_in_force=tif, limit_price=round(float(px), 2))

        order = self._client.submit_order(req)
        bid = str(order.id)
        logger.info("ðŸ“¨ ALPACA æäº¤ %s %s qty=%d type=%s id=%s",
                    intent.side.value, intent.symbol, qty, intent.order_type, bid)
        return bid

    def cancel_order(self, broker_order_id: str) -> bool:
        try:
            self._client.cancel_order_by_id(broker_order_id)
            return True
        except Exception as exc:
            logger.warning("ALPACA æ’¤å•å¤±è´¥ %s: %s", broker_order_id, exc)
            return False

    def get_order_status(self, broker_order_id: str) -> OrderStatus:
        try:
            o = self._client.get_order_by_id(broker_order_id)
            return _map_status(o.status)
        except Exception as exc:
            logger.warning("ALPACA æŸ¥è¯¢è®¢å•å¤±è´¥ %s: %s", broker_order_id, exc)
            return OrderStatus.FAILED

    def get_fill(self, broker_order_id: str) -> Optional[Fill]:
        try:
            o = self._client.get_order_by_id(broker_order_id)
        except Exception as exc:
            logger.warning("ALPACA æŸ¥è¯¢æˆäº¤å¤±è´¥ %s: %s", broker_order_id, exc)
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
            logger.warning("ALPACA æŸ¥è¯¢æŒä»“å¤±è´¥: %s", exc)
        return out

    def get_account_equity(self) -> float:
        try:
            acct = self._client.get_account()
            return float(acct.equity)
        except Exception as exc:
            logger.warning("ALPACA æŸ¥è¯¢æƒç›Šå¤±è´¥: %s", exc)
            return 0.0

