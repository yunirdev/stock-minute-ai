from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


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


# ---------------------------------------------------------------------------
# M0 新增数据模型（PLAN.md §4，M0 冻结）
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    """选股输出：一个候选标的及其可解释打分。"""
    symbol: str
    score: float                        # 0-100 综合/共识分
    rank: int
    reasons: Dict[str, Any]             # {"votes": {strategy: +1/-1}, "factors": {...}}
    as_of: datetime = field(default_factory=utc_now)


@dataclass
class TradePlan:
    """核心产物：纪律化交易计划（不下 market；entry/stop/tp 都是预设价位）。"""
    plan_id: str
    symbol: str
    side: Side
    action: str                         # OPEN | ADD | REDUCE | CLOSE | HOLD
    entry_price: float                  # 入手价（挂 LMT）
    stop_loss: float                    # 止损价
    take_profit: float                  # 止盈价
    target_weight: float = 0.0          # 目标组合权重（allocator 填）
    qty: float = 0.0                    # 数量（allocator/risk 填）
    confidence: float = 1.0
    rationale: str = ""                 # 为什么：哪些信号/agent/新闻
    source: str = "consensus"           # consensus | ai | manual
    status: str = "DRAFT"              # DRAFT | APPROVED | REJECTED | LIVE | CLOSED
    created_at: datetime = field(default_factory=utc_now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Advisory:
    """AI 旁路产出的建议工件（永不直接执行，必须过确定性风控）。"""
    advisory_id: str
    kind: str                           # selection | plan | review | news | risk_review
    agent: str                          # 产出它的 agent 角色名
    payload: Dict[str, Any]             # 结构化内容
    confidence: float = 0.0
    model: str = ""                     # 模型 id / 版本
    created_at: datetime = field(default_factory=utc_now)


@dataclass
class NewsEvent:
    """新闻/异动/日历/社区触发事件。"""
    event_id: str
    kind: str                           # news | price_move | calendar | community
    symbol: Optional[str]
    title: str
    summary: str = ""
    url: Optional[str] = None
    severity: float = 0.0               # 异动强度 0-1
    ts: datetime = field(default_factory=utc_now)
    source: str = ""


@dataclass
class ReviewReport:
    """盘后复盘归因。"""
    report_id: str
    period: str                         # daily | weekly
    market_summary: str
    portfolio_pnl: float
    attribution: Dict[str, Any]
    trades: List[Any] = field(default_factory=list)
    created_at: datetime = field(default_factory=utc_now)


@dataclass
class Alert:
    """看门狗/系统告警。"""
    level: str                          # info | warn | critical
    source: str
    message: str
    ts: datetime = field(default_factory=utc_now)


@dataclass
class Notification:
    """推送统一载体（notify 用）。"""
    title: str
    body: str
    kind: str = "info"                  # selection | plan | review | news | alert | info
    fields: Dict[str, Any] = field(default_factory=dict)
    plan_id: Optional[str] = None       # 若是计划推送，带 plan_id 支持一键审批
