"""
order_manager.py
Manages the pending-order queue for gap-fill / limit-entry logic.

Problem solved
--------------
When a strategy fires a signal, the suggested exec_price is often the
*previous* bar's high/low (e.g. "buy on breakout above last bar's high").
If the next bar opens with a gap that has already blown past that level,
filling at exec_price is impossible — we would be chasing the price.

Solution
--------
Instead of submitting the order immediately, we place a PendingOrder.
Each subsequent tick we check whether the current bar's [low, high] range
contains the limit_price.  If yes → fill.  Otherwise keep waiting.

Gap detection rules
-------------------
BUY  signal, limit_price = trigger:
    • Reachable on this bar  : bar.low  <= limit_price <= bar.high  → fill now
    • Gap UP   (bar.low  > limit_price): price already above trigger → PENDING
      (wait for pullback to limit_price)

SELL signal, limit_price = trigger:
    • Reachable on this bar  : bar.low  <= limit_price <= bar.high  → fill now
    • Gap DOWN (bar.high < limit_price): price already below trigger → PENDING
      (wait for bounce back to limit_price)

Cancellation conditions
-----------------------
1. bars_alive >= max_bars_alive  → EXPIRED (timeout)
2. Opposite-direction signal fires (i.e. strategy TP/SL triggered) → CANCELLED
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

from .models import PendingOrder, Side, Signal

logger = logging.getLogger(__name__)


class OrderManager:
    """
    Holds the pending-order queue and evaluates fill / expiry each tick.

    Usage in Scheduler
    ------------------
    1. After computing a signal, call ``maybe_enqueue(signal, current_bar)``
       instead of submitting directly.  The method returns:
         - ``True``  if the order was filled immediately (exec_price reachable)
         - ``False`` if it was queued as a PendingOrder (gap detected)

    2. At the start of each tick, call ``check_pending(current_bar, on_fill)``
       which iterates the queue, ages orders, checks fill conditions, and
       calls ``on_fill(pending)`` for any order that becomes fillable.
    """

    def __init__(self, max_bars_alive: int = 10) -> None:
        self._max_bars_alive = max_bars_alive
        # symbol → list of pending orders (usually 0 or 1 per symbol)
        self._queue: Dict[str, List[PendingOrder]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def maybe_enqueue(
        self,
        signal: Signal,
        bar_open: float,
        bar_high: float,
        bar_low: float,
        bar_close: float,
    ) -> bool:
        """
        Decide whether to fill immediately or queue a pending order.

        Returns
        -------
        True  — exec_price is reachable on this bar; caller should submit now.
        False — gap detected; order queued; caller should NOT submit now.
        """
        limit_px = signal.exec_price
        symbol   = signal.symbol

        # ── Is the limit price reachable on this bar? ──────────────────
        reachable = bar_low <= limit_px <= bar_high

        if reachable:
            logger.debug(
                "OrderManager: %s %s limit=%.4f reachable on bar [%.4f–%.4f] → fill now",
                signal.side.value, symbol, limit_px, bar_low, bar_high,
            )
            # Cancel any stale pending orders for this symbol in the same direction
            self._cancel_same_direction(symbol, signal.side, reason="superseded by immediate fill")
            return True  # caller submits immediately

        # ── Gap detected → queue ────────────────────────────────────────
        pending = PendingOrder(
            pending_id=str(uuid.uuid4())[:8],
            signal=signal,
            limit_price=limit_px,
            side=signal.side,
            symbol=symbol,
            created_at=datetime.now(timezone.utc),
            max_bars_alive=self._max_bars_alive,
            bars_alive=0,
            status="WAITING",
        )

        if symbol not in self._queue:
            self._queue[symbol] = []

        # Cancel any existing pending in the same direction (replace with fresher signal)
        self._cancel_same_direction(symbol, signal.side, reason="replaced by newer signal")

        self._queue[symbol].append(pending)

        gap_dir = "UP" if bar_low > limit_px else "DOWN"
        logger.info(
            "⏳ PENDING  %s %s  limit=%.4f  bar=[%.4f–%.4f]  gap=%s  "
            "max_bars=%d  id=%s",
            signal.side.value, symbol, limit_px,
            bar_low, bar_high, gap_dir,
            self._max_bars_alive, pending.pending_id,
        )
        return False  # caller should NOT submit now

    def check_pending(
        self,
        symbol: str,
        bar_high: float,
        bar_low: float,
        bar_close: float,
        on_fill: Callable[[PendingOrder], None],
    ) -> None:
        """
        Called every tick for each symbol.  Evaluates all WAITING orders:
        - Ages them (+1 bar)
        - Checks TP/SL guards → CANCELLED
        - Checks expiry       → EXPIRED
        - Checks fill range   → calls on_fill() → FILLED
        """
        orders = self._queue.get(symbol, [])
        if not orders:
            return

        still_waiting: List[PendingOrder] = []
        for order in orders:
            if order.status != "WAITING":
                continue

            order.bars_alive += 1

            # ── 1. Expiry check ─────────────────────────────────────────
            if order.bars_alive > order.max_bars_alive:
                order.status = "EXPIRED"
                logger.info(
                    "⌛ EXPIRED  %s %s  limit=%.4f  after %d bars  id=%s",
                    order.side.value, symbol, order.limit_price,
                    order.bars_alive, order.pending_id,
                )
                continue

            # ── 2. TP/SL guard ──────────────────────────────────────────
            cancelled = False
            if order.side == Side.BUY:
                if order.tp_price and bar_close > order.tp_price:
                    order.status = "CANCELLED"
                    logger.info(
                        "🚫 CANCELLED (price ran away) %s %s  limit=%.4f  "
                        "close=%.4f > tp_guard=%.4f  id=%s",
                        order.side.value, symbol, order.limit_price,
                        bar_close, order.tp_price, order.pending_id,
                    )
                    cancelled = True
                elif order.sl_price and bar_close < order.sl_price:
                    order.status = "CANCELLED"
                    logger.info(
                        "🚫 CANCELLED (falling knife) %s %s  limit=%.4f  "
                        "close=%.4f < sl_guard=%.4f  id=%s",
                        order.side.value, symbol, order.limit_price,
                        bar_close, order.sl_price, order.pending_id,
                    )
                    cancelled = True
            else:  # SELL / short
                if order.tp_price and bar_close < order.tp_price:
                    order.status = "CANCELLED"
                    logger.info(
                        "🚫 CANCELLED (price dropped away) %s %s  limit=%.4f  "
                        "close=%.4f < tp_guard=%.4f  id=%s",
                        order.side.value, symbol, order.limit_price,
                        bar_close, order.tp_price, order.pending_id,
                    )
                    cancelled = True
                elif order.sl_price and bar_close > order.sl_price:
                    order.status = "CANCELLED"
                    logger.info(
                        "🚫 CANCELLED (short squeeze) %s %s  limit=%.4f  "
                        "close=%.4f > sl_guard=%.4f  id=%s",
                        order.side.value, symbol, order.limit_price,
                        bar_close, order.sl_price, order.pending_id,
                    )
                    cancelled = True
            if cancelled:
                continue

            # ── 3. Fill check ────────────────────────────────────────────
            if bar_low <= order.limit_price <= bar_high:
                order.status = "FILLED"
                logger.info(
                    "✅ PENDING FILLED  %s %s  limit=%.4f  bar=[%.4f–%.4f]  "
                    "bars_waited=%d  id=%s",
                    order.side.value, symbol, order.limit_price,
                    bar_low, bar_high, order.bars_alive, order.pending_id,
                )
                on_fill(order)
                continue

            still_waiting.append(order)

        self._queue[symbol] = still_waiting

    def cancel_opposite(self, symbol: str, side: Side) -> None:
        """
        Cancel all WAITING orders in the *opposite* direction for a symbol.
        Called when a new signal fires in the opposite direction.
        """
        opposite = Side.SELL if side == Side.BUY else Side.BUY
        self._cancel_same_direction(symbol, opposite, reason="opposite signal fired")

    def pending_count(self, symbol: Optional[str] = None) -> int:
        """Return number of WAITING orders (for a symbol, or total)."""
        if symbol:
            return sum(1 for o in self._queue.get(symbol, []) if o.status == "WAITING")
        return sum(
            1 for orders in self._queue.values()
            for o in orders if o.status == "WAITING"
        )

    def get_pending(self, symbol: str) -> List[PendingOrder]:
        """Return all WAITING orders for a symbol."""
        return [o for o in self._queue.get(symbol, []) if o.status == "WAITING"]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cancel_same_direction(self, symbol: str, side: Side, reason: str) -> None:
        for order in self._queue.get(symbol, []):
            if order.status == "WAITING" and order.side == side:
                order.status = "CANCELLED"
                logger.info(
                    "🗑  CANCELLED (%s)  %s %s  limit=%.4f  id=%s",
                    reason, side.value, symbol, order.limit_price, order.pending_id,
                )
