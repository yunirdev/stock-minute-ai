from __future__ import annotations
import uuid
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
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
    timeframe: str = ""


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
    idempotency_key: str = ""
    client_order_id: str = ""
    decision_id: str = ""
    plan_id: str = ""
    candidate_plan_id: str = ""
    final_plan_id: str = ""
    final_plan_version: int = 0
    risk_check_id: str = ""
    evidence_refs: tuple[str, ...] = ()


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
class Position:
    """Current open position for one symbol."""
    symbol: str
    qty: float              # positive = long, negative = short
    avg_entry_px: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    last_updated: datetime = field(default_factory=utc_now)


# ---------------------------------------------------------------------------
# Decision and reporting models
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    """选股输出：一个候选标的及其可解释打分。"""
    symbol: str
    score: float                        # 0-100 综合/共识分
    rank: int
    reasons: Dict[str, Any]             # {"votes": {strategy: +1/-1}, "factors": {...}}
    as_of: datetime = field(default_factory=utc_now)


class CandidatePlanStatus(str, Enum):
    DRAFT = "DRAFT"
    VALIDATED = "VALIDATED"
    FINALIZED = "FINALIZED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class FinalTradePlanStatus(str, Enum):
    FINAL_EXECUTABLE = "FINAL_EXECUTABLE"
    ORDER_INTENT_CREATED = "ORDER_INTENT_CREATED"
    EXPIRED = "EXPIRED"
    REJECTED = "REJECTED"
    CLOSED = "CLOSED"


@dataclass(frozen=True)
class CandidatePlan:
    candidate_plan_id: str
    symbol: str
    side: Side
    action: str
    entry_price: float
    stop_loss: float
    take_profit: float
    proposed_qty: float
    decision_id: str
    strategy_version: str
    data_version: str
    evidence_refs: tuple[str, ...]
    created_at: datetime
    valid_until: datetime
    status: CandidatePlanStatus = CandidatePlanStatus.DRAFT

    def __post_init__(self) -> None:
        if not all(
            str(value).strip()
            for value in (
                self.candidate_plan_id,
                self.symbol,
                self.action,
                self.decision_id,
                self.strategy_version,
                self.data_version,
            )
        ):
            raise ValueError("CANDIDATE_PLAN_FIELD_REQUIRED")
        object.__setattr__(self, "symbol", self.symbol.strip().upper())
        _require_aware_timestamp(self.created_at, "candidate_created_at")
        _require_aware_timestamp(self.valid_until, "candidate_valid_until")
        if self.valid_until <= self.created_at:
            raise ValueError("CANDIDATE_PLAN_VALIDITY_INVALID")
        numeric = (
            self.entry_price,
            self.stop_loss,
            self.take_profit,
            self.proposed_qty,
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("CANDIDATE_PLAN_VALUE_NONFINITE")
        if min(numeric) <= 0:
            raise ValueError("CANDIDATE_PLAN_VALUE_INVALID")
        increases_exposure = self.action.upper() in {"OPEN", "ADD"}
        if increases_exposure and self.side == Side.BUY and not (
            self.stop_loss < self.entry_price < self.take_profit
        ):
            raise ValueError("CANDIDATE_PLAN_DIRECTION_INVALID")
        if increases_exposure and self.side == Side.SELL and not (
            self.take_profit < self.entry_price < self.stop_loss
        ):
            raise ValueError("CANDIDATE_PLAN_DIRECTION_INVALID")
        refs = tuple(
            dict.fromkeys(
                str(reference).strip()
                for reference in self.evidence_refs
                if str(reference).strip()
            )
        )
        if not refs:
            raise ValueError("CANDIDATE_PLAN_EVIDENCE_REQUIRED")
        object.__setattr__(self, "evidence_refs", refs)


@dataclass(frozen=True)
class FinalTradePlan:
    final_plan_id: str
    version: int
    candidate_plan_id: str
    symbol: str
    side: Side
    action: str
    entry_price: float
    stop_loss: float
    take_profit: float
    qty: float
    decision_id: str
    risk_check_id: str
    risk_config_version: str
    strategy_version: str
    data_version: str
    evidence_refs: tuple[str, ...]
    created_at: datetime
    valid_until: datetime
    status: FinalTradePlanStatus = (
        FinalTradePlanStatus.FINAL_EXECUTABLE
    )

    def __post_init__(self) -> None:
        if not all(
            str(value).strip()
            for value in (
                self.final_plan_id,
                self.candidate_plan_id,
                self.symbol,
                self.action,
                self.decision_id,
                self.risk_check_id,
                self.risk_config_version,
                self.strategy_version,
                self.data_version,
            )
        ):
            raise ValueError("FINAL_TRADE_PLAN_FIELD_REQUIRED")
        object.__setattr__(self, "symbol", self.symbol.strip().upper())
        _require_aware_timestamp(self.created_at, "final_plan_created_at")
        _require_aware_timestamp(self.valid_until, "final_plan_valid_until")
        if self.version < 1 or self.valid_until <= self.created_at:
            raise ValueError("FINAL_TRADE_PLAN_VERSION_OR_VALIDITY_INVALID")
        values = (
            self.entry_price,
            self.stop_loss,
            self.take_profit,
            self.qty,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("FINAL_TRADE_PLAN_VALUE_NONFINITE")
        if min(values) <= 0:
            raise ValueError("FINAL_TRADE_PLAN_VALUE_INVALID")
        increases_exposure = self.action.upper() in {"OPEN", "ADD"}
        if increases_exposure and self.side == Side.BUY and not (
            self.stop_loss < self.entry_price < self.take_profit
        ):
            raise ValueError("FINAL_TRADE_PLAN_DIRECTION_INVALID")
        if increases_exposure and self.side == Side.SELL and not (
            self.take_profit < self.entry_price < self.stop_loss
        ):
            raise ValueError("FINAL_TRADE_PLAN_DIRECTION_INVALID")
        if not self.evidence_refs:
            raise ValueError("FINAL_TRADE_PLAN_EVIDENCE_REQUIRED")


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
    status: str = "DRAFT"              # DRAFT | READY | DRY_RUN | REJECTED
    created_at: datetime = field(default_factory=utc_now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PositionPlanStatus(str, Enum):
    ACTIVE = "ACTIVE"
    REDUCING = "REDUCING"
    EXIT_PENDING = "EXIT_PENDING"
    CLOSED = "CLOSED"


class PositionAdjustmentAction(str, Enum):
    REDUCE = "REDUCE"
    EXIT = "EXIT"
    TIGHTEN_STOP = "TIGHTEN_STOP"


class PositionAdjustmentStatus(str, Enum):
    PLANNED = "PLANNED"
    ORDER_CREATED = "ORDER_CREATED"
    COMPLETED = "COMPLETED"


@dataclass(frozen=True)
class PositionAdjustment:
    """One deterministic response to a validated invalidation event."""

    adjustment_id: str
    event_id: str
    position_plan_id: str
    from_version_id: str
    to_version_id: str
    action: PositionAdjustmentAction
    status: PositionAdjustmentStatus
    quantity: float
    limit_price: float
    previous_stop_loss: float
    new_stop_loss: float
    order_plan_id: str
    order_intent_id: str = ""
    order_idempotency_key: str = ""
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        required = (
            self.adjustment_id,
            self.event_id,
            self.position_plan_id,
            self.from_version_id,
            self.to_version_id,
        )
        if not all(str(value).strip() for value in required):
            raise ValueError("POSITION_ADJUSTMENT_FIELD_REQUIRED")
        _require_aware_timestamp(
            self.created_at,
            "position_adjustment_created_at",
        )
        values = (
            self.quantity,
            self.limit_price,
            self.previous_stop_loss,
            self.new_stop_loss,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("POSITION_ADJUSTMENT_VALUE_NONFINITE")
        if self.previous_stop_loss <= 0 or self.new_stop_loss <= 0:
            raise ValueError("POSITION_ADJUSTMENT_STOP_INVALID")
        if self.action == PositionAdjustmentAction.TIGHTEN_STOP:
            if (
                self.quantity != 0
                or self.limit_price != 0
                or self.order_plan_id
            ):
                raise ValueError("STOP_ADJUSTMENT_ORDER_NOT_ALLOWED")
        elif (
            self.quantity <= 0
            or self.limit_price <= 0
            or not self.order_plan_id
        ):
            raise ValueError("POSITION_ADJUSTMENT_ORDER_REQUIRED")
        if (
            self.action != PositionAdjustmentAction.TIGHTEN_STOP
            and self.status
            in {
                PositionAdjustmentStatus.ORDER_CREATED,
                PositionAdjustmentStatus.COMPLETED,
            }
            and (
                not self.order_intent_id
                or not self.order_idempotency_key
            )
        ):
            raise ValueError("POSITION_ADJUSTMENT_INTENT_REQUIRED")


class InvalidationEventType(str, Enum):
    PRICE_STOP = "PRICE_STOP"
    BROKER_RESTRICTION = "BROKER_RESTRICTION"
    CORPORATE_ACTION = "CORPORATE_ACTION"
    TRADING_RESTRICTION = "TRADING_RESTRICTION"
    STRATEGY_INVALIDATED = "STRATEGY_INVALIDATED"


class InvalidationSource(str, Enum):
    MARKET_DATA = "MARKET_DATA"
    BROKER = "BROKER"
    CORPORATE_ACTION_DATA = "CORPORATE_ACTION_DATA"
    EXCHANGE = "EXCHANGE"
    STRATEGY_ENGINE = "STRATEGY_ENGINE"


@dataclass(frozen=True)
class InvalidationEvent:
    """One source-backed fact that may invalidate a PositionPlan."""

    event_id: str
    position_plan_id: str
    position_plan_version_id: str
    symbol: str
    event_type: InvalidationEventType
    source: InvalidationSource
    source_event_id: str
    rule_id: str
    as_of: datetime
    observed_at: datetime
    facts_json: str
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        required = (
            self.event_id,
            self.position_plan_id,
            self.position_plan_version_id,
            self.symbol,
            self.source_event_id,
            self.rule_id,
            self.facts_json,
        )
        if not all(str(value).strip() for value in required):
            raise ValueError("INVALIDATION_EVENT_FIELD_REQUIRED")
        object.__setattr__(self, "symbol", self.symbol.strip().upper())
        _require_aware_timestamp(self.as_of, "invalidation_event_as_of")
        _require_aware_timestamp(
            self.observed_at,
            "invalidation_event_observed_at",
        )
        refs = tuple(
            dict.fromkeys(
                str(reference).strip()
                for reference in self.evidence_refs
                if str(reference).strip()
            )
        )
        if not refs:
            raise ValueError("INVALIDATION_EVENT_EVIDENCE_REQUIRED")
        object.__setattr__(self, "evidence_refs", refs)


@dataclass(frozen=True)
class PositionPlan:
    """One immutable version in a filled position's durable plan chain."""

    position_plan_id: str
    version_id: str
    version: int
    parent_version_id: str
    symbol: str
    side: Side
    status: PositionPlanStatus
    source_trade_plan_id: str
    initial_fill_id: str
    initial_entry_price: float
    initial_quantity: float
    open_quantity: float
    average_entry_price: float
    stop_loss: float
    take_profit: float
    invalidation_rules: tuple[str, ...]
    change_reason: str
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        required = (
            self.position_plan_id,
            self.version_id,
            self.symbol,
            self.source_trade_plan_id,
            self.initial_fill_id,
            self.change_reason,
        )
        if not all(str(value).strip() for value in required):
            raise ValueError("POSITION_PLAN_FIELD_REQUIRED")
        object.__setattr__(self, "symbol", self.symbol.strip().upper())
        _require_aware_timestamp(self.created_at, "position_plan_created_at")
        if self.version < 1:
            raise ValueError("POSITION_PLAN_VERSION_INVALID")
        if self.version == 1 and self.parent_version_id:
            raise ValueError("POSITION_PLAN_INITIAL_PARENT_INVALID")
        if self.version > 1 and not self.parent_version_id:
            raise ValueError("POSITION_PLAN_PARENT_REQUIRED")
        numeric = (
            self.initial_entry_price,
            self.initial_quantity,
            self.open_quantity,
            self.average_entry_price,
            self.stop_loss,
            self.take_profit,
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("POSITION_PLAN_VALUE_NONFINITE")
        if (
            self.initial_entry_price <= 0
            or self.initial_quantity <= 0
            or self.average_entry_price <= 0
            or self.stop_loss <= 0
            or self.take_profit <= 0
            or self.open_quantity < 0
        ):
            raise ValueError("POSITION_PLAN_VALUE_INVALID")
        if self.status == PositionPlanStatus.CLOSED:
            if self.open_quantity != 0:
                raise ValueError("CLOSED_POSITION_PLAN_HAS_QUANTITY")
        elif self.open_quantity <= 0:
            raise ValueError("OPEN_POSITION_PLAN_QUANTITY_REQUIRED")
        if self.side == Side.BUY:
            valid_prices = (
                self.stop_loss
                < self.average_entry_price
                < self.take_profit
                if self.version == 1
                else self.stop_loss < self.take_profit
            )
            if not valid_prices:
                raise ValueError("LONG_POSITION_PLAN_PRICE_ORDER_INVALID")
        if self.side == Side.SELL:
            valid_prices = (
                self.take_profit
                < self.average_entry_price
                < self.stop_loss
                if self.version == 1
                else self.take_profit < self.stop_loss
            )
            if not valid_prices:
                raise ValueError("SHORT_POSITION_PLAN_PRICE_ORDER_INVALID")
        rules = tuple(
            dict.fromkeys(
                str(rule).strip()
                for rule in self.invalidation_rules
                if str(rule).strip()
            )
        )
        if not rules:
            raise ValueError("POSITION_PLAN_INVALIDATION_RULE_REQUIRED")
        object.__setattr__(self, "invalidation_rules", rules)


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
    is_fallback: bool = False


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
    plan_id: Optional[str] = None       # 若是计划推送，带 plan_id 支持审计追踪
    #: 业务去重身份，例如 "morning_brief:1:2026-08-03"、"daily_review:2026-08-03"。
    #:
    #: 不填时推送层回退到"内容哈希"去重，那种方式只能挡住逐字节相同的重复。
    #: 实测过一次教训：晨报在 31 秒内被推了两遍，四条里只有一条因为正文碰巧
    #: 一字不差被挡下，其余三条因为行情数字动了几位就被当成新消息放行了。而
    #: 且那次是两个进程各自持有内存里的"今天发过了"标记，落盘也救不了——只有
    #: 业务身份能跨进程去重。报告类 builder 都应该填这个字段。
    dedupe_key: Optional[str] = None


@dataclass
class AgentContext:
    """Read-only-by-convention input shared with analysis agents."""
    candidates: List[Candidate] = field(default_factory=list)
    plans: List[TradePlan] = field(default_factory=list)
    news: List[NewsEvent] = field(default_factory=list)
    positions: Dict[str, Position] = field(default_factory=dict)
    equity: float = 0.0
    as_of: Optional[datetime] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class ResearchSourceStatus(str, Enum):
    OK = "OK"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"
    MISSING = "MISSING"


class ResearchQuality(str, Enum):
    GOOD = "GOOD"
    DEGRADED = "DEGRADED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


def _require_aware_timestamp(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name.upper()}_TIMEZONE_REQUIRED")


def _require_quality_score(value: float, field_name: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name.upper()}_OUT_OF_RANGE")


@dataclass(frozen=True)
class ResearchSourceManifestEntry:
    """One source fact boundary used to build a research snapshot."""

    source: str
    status: ResearchSourceStatus
    as_of: datetime
    fetched_at: datetime
    quality_score: float
    coverage: tuple[str, ...]
    payload_version: str
    failure_code: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError("RESEARCH_SOURCE_REQUIRED")
        if not self.payload_version.strip():
            raise ValueError("SOURCE_PAYLOAD_VERSION_REQUIRED")
        _require_aware_timestamp(self.as_of, "source_as_of")
        _require_aware_timestamp(self.fetched_at, "source_fetched_at")
        if self.as_of > self.fetched_at:
            raise ValueError("SOURCE_AS_OF_AFTER_FETCH")
        _require_quality_score(self.quality_score, "source_quality_score")
        normalized_coverage = tuple(
            dict.fromkeys(
                item.strip()
                for item in self.coverage
                if item.strip()
            )
        )
        if not normalized_coverage:
            raise ValueError("SOURCE_COVERAGE_REQUIRED")
        object.__setattr__(self, "coverage", normalized_coverage)
        if self.status in {
            ResearchSourceStatus.FAILED,
            ResearchSourceStatus.MISSING,
        }:
            if self.quality_score != 0.0:
                raise ValueError("FAILED_SOURCE_QUALITY_MUST_BE_ZERO")
            if not self.failure_code.strip():
                raise ValueError("FAILED_SOURCE_CODE_REQUIRED")


@dataclass(frozen=True)
class ResearchSnapshot:
    """Versioned facts and provenance captured for one symbol research input."""

    snapshot_id: str
    symbol: str
    trading_date: str
    as_of: datetime
    data_cutoff: datetime
    captured_at: datetime
    source_manifest: tuple[ResearchSourceManifestEntry, ...]
    quality: ResearchQuality
    quality_score: float
    payload_version: str
    payload: Dict[str, Any]
    run_id: str = ""
    schema_version: int = 2
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.snapshot_id.strip():
            raise ValueError("SNAPSHOT_ID_REQUIRED")
        normalized_symbol = self.symbol.strip().upper()
        if not normalized_symbol:
            raise ValueError("SNAPSHOT_SYMBOL_REQUIRED")
        object.__setattr__(self, "symbol", normalized_symbol)
        try:
            date.fromisoformat(self.trading_date)
        except ValueError as exc:
            raise ValueError("SNAPSHOT_TRADING_DATE_INVALID") from exc
        for field_name in (
            "as_of",
            "data_cutoff",
            "captured_at",
            "created_at",
        ):
            _require_aware_timestamp(
                getattr(self, field_name),
                f"snapshot_{field_name}",
            )
        if self.as_of > self.data_cutoff:
            raise ValueError("SNAPSHOT_AS_OF_AFTER_DATA_CUTOFF")
        if self.data_cutoff > self.captured_at:
            raise ValueError("SNAPSHOT_DATA_CUTOFF_AFTER_CAPTURE")
        if self.captured_at > self.created_at:
            raise ValueError("SNAPSHOT_CAPTURE_AFTER_CREATED")
        _require_quality_score(self.quality_score, "snapshot_quality_score")
        if self.quality == ResearchQuality.FAILED and self.quality_score != 0.0:
            raise ValueError("FAILED_SNAPSHOT_QUALITY_MUST_BE_ZERO")
        if not self.payload_version.strip():
            raise ValueError("SNAPSHOT_PAYLOAD_VERSION_REQUIRED")
        if self.schema_version < 1:
            raise ValueError("SNAPSHOT_SCHEMA_VERSION_INVALID")
        if (
            self.schema_version >= 2
            and not self.source_manifest
        ):
            raise ValueError("SNAPSHOT_SOURCE_MANIFEST_REQUIRED")
