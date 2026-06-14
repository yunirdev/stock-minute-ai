from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional


def new_id() -> str:
    return str(uuid.uuid4())


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"


# ---------------------------------------------------------------------------
# Core data models
# ---------------------------------------------------------------------------

@dataclass
class Bar:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Signal:
    """A trading signal produced by a strategy."""
    signal_id: str
    symbol: str
    strategy: str
    side: Side
    exec_price: float      # suggested execution price
    timeframe: str
    signal_time: datetime
    bar_close: float
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskVerdict:
    """Output from RiskEngine.evaluate()."""
    approved: bool
    reason: str = ""
    suggested_qty: float = 0.0


@dataclass
class OrderIntent:
    """A risk-approved, ready-to-submit order."""
    intent_id: str
    signal_id: str
    symbol: str
    side: Side
    qty: float
    order_type: str            # "LMT" | "MKT"
    limit_price: Optional[float]
    reference_price: Optional[float] = None
    tif: str = "DAY"
    risk_tag: str = ""
    created_at: datetime = field(default_factory=utc_now)


@dataclass
class Fill:
    """A confirmed execution report from the broker."""
    order_id: str
    intent_id: str
    symbol: str
    side: Side
    filled_qty: float
    avg_price: float
    fill_time: datetime
    fee: float = 0.0
    broker_payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PendingOrder:
    """
    A limit order waiting for price to pull back to the trigger level.

    Created when the strategy fires a signal but the bar has already
    gapped past exec_price (i.e. the price is unreachable on this bar).

    Lifecycle:
        WAITING  → price pulls back into [low, high] that contains limit_price
                   → FILLED (OrderIntent is submitted to broker)
        WAITING  → bars_alive >= max_bars_alive
                   → EXPIRED (order cancelled, no fill)
        WAITING  → strategy fires opposite-direction signal (TP/SL triggered)
                   → CANCELLED
    """
    pending_id: str
    signal: "Signal"                # original signal that triggered this order
    limit_price: float              # the price we want to buy/sell at
    side: Side
    symbol: str
    created_at: datetime
    max_bars_alive: int = 10        # cancel after this many ticks without fill
    bars_alive: int = 0             # incremented each tick
    status: str = "WAITING"         # WAITING | FILLED | EXPIRED | CANCELLED
    tp_price: Optional[float] = None
    sl_price: Optional[float] = None


@dataclass
class Position:
    """Current open position for one symbol."""
    symbol: str
    qty: float              # positive = long, negative = short
    avg_entry_px: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    last_updated: datetime = field(default_factory=utc_now)
