"""
trader/runtime.py
唯一生产交易循环：Candidate → TradePlan → allocation → AI safety → risk → DRY_RUN/LMT。

每轮负责 kill switch、watchdog、启动对账、行情、订单轮询、组合与审计持久化。
Agent/LLM 不直接调用 broker；自动执行仅允许 Alpaca Paper 限价单。
"""
from __future__ import annotations

import logging
import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd

from .allocator import EqualWeightAllocator
from .audit import AuditLog
from .broker.alpaca import AlpacaBroker
from .config import TradingConfig
from .data_cache import upsert_bars as _dc_upsert
from .data_feed import AlpacaDataFeed
from .market_calendar import SimpleMarketCalendar
from .models import (
    Bar,
    CandidatePlanStatus,
    FinalTradePlanStatus,
    InvalidationEvent,
    Notification,
    OrderIntent,
    OrderStatus,
    Position,
    PositionAdjustment,
    RiskVerdict,
    Side,
    TradePlan,
    utc_now,
)
from .news import (
    FinnhubSource,
    NewsEventStore,
    PriceMoveSource,
    SECEdgarSource,
    WallStreetCNSource,
    poll_all_sources,
)
from .notify import DiscordNotifier
from .plan import ATRPlanner
from .portfolio import Portfolio
from .invalidation_events import InvalidationEventStore
from .position_adjustments import PositionAdjustmentStore
from .position_plans import PositionPlanFillProjector, PositionPlanStore
from .position_quality import PositionQualityStore
from .post_trade_learning import PostTradeLearningCoordinator
from .production_evidence import ProductionEvidenceCoordinator
from .production_operations import ProductionOperationsCoordinator
from .position_monitor import StopTakeProfitMonitor
from .review import SimpleReviewer
from .risk_engine import RiskEngine
from .order_lifecycle import (
    OrderIntentStore,
    OrderLifecycle,
    ReconciliationReport,
    idempotency_key,
    reconcile_broker_facts,
)
from .paper_decision import PaperDecisionService, StrategyStatisticsRepository, UniverseProvider
from .bug_reporting import BugReporter
from .ai.safety import AIScorePolicy, AIScoreSnapshot, AIScoreValidator
from .daily_runtime_support import DailyRuntimeSupport
from .execution_pipeline import (
    ExecutionPipelineStore,
    candidate_from_trade_plan,
)
from .signal_reports import SignalStore, build_ready_signal_report
from .selection import AICandidateSelector
from .trade_episodes import TradeEpisodeStore
from .watchdog import FileKillSwitch, HeartbeatWatchdog

logger = logging.getLogger(__name__)

_MIN_CANDIDATE_SCORE = 55.0   # selection 阈值，低于此分的 Candidate 不进入计划

# 市场环境 → 选股分数阈值映射（HIGH_VOL 时跳过选股）
_REGIME_SCORE_MAP: dict[str, float | None] = {
    "bull_trend": 55.0,
    "neutral":    60.0,
    "bear_trend": 70.0,
    "high_vol":   None,   # None = 高波动暂停选股
}


def _timeframe_minutes(timeframe: str) -> Optional[int]:
    raw = str(timeframe or "").strip().lower()
    if raw.endswith("m") and raw[:-1].isdigit():
        return int(raw[:-1])
    if raw.endswith("h") and raw[:-1].isdigit():
        return int(raw[:-1]) * 60
    return None


def _latest_bar_is_fresh(latest: Bar, timeframe: str, *, now: Optional[datetime] = None) -> bool:
    """True when the newest intraday bar is recent enough for live decisions.

    Alpaca free/IEX-style feeds can be delayed, so this is not a zero-lag check.
    It prevents the dangerous case: after holidays/weekends or before the first
    delayed bar arrives, old bars from a previous session would otherwise pass
    the "len(raw) >= 30" test and generate orders.
    """
    minutes = _timeframe_minutes(timeframe)
    if minutes is None:
        return True
    ts = latest.timestamp
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    now_dt = now or datetime.now(timezone.utc)
    max_age = timedelta(minutes=max(45, minutes * 2 + 35))
    return now_dt - ts.astimezone(timezone.utc) <= max_age


def _get_current_regime_label() -> str:
    try:
        from .teams.market_env import read_regime_cache
        cached = read_regime_cache()
        return cached.regime.value if cached else "no_cache"
    except Exception:
        return "unknown"


def _get_score_threshold() -> float | None:
    """读取 T0 缓存，动态返回选股分数阈值。None 表示暂停选股。
    缓存超过 24h 视为过期，回退到默认值。"""
    try:
        from .teams.market_env import read_regime_cache
        cached = read_regime_cache()
        if cached is None:
            return _MIN_CANDIDATE_SCORE
        age = (datetime.now(timezone.utc) - cached.as_of).total_seconds()
        if age > 86400:
            logger.debug("T0 缓存已过期(%.0fh)，使用默认阈值", age / 3600)
            return _MIN_CANDIDATE_SCORE
        threshold = _REGIME_SCORE_MAP.get(cached.regime.value, _MIN_CANDIDATE_SCORE)
        logger.debug(
            "T0 市场环境=%s → 选股阈值=%s (age=%.0fh)",
            cached.regime.value, threshold, age / 3600,
        )
        return threshold
    except Exception as exc:
        logger.debug("_get_score_threshold 失败，使用默认值: %s", exc)
        return _MIN_CANDIDATE_SCORE


def _alpaca_bar_to_model(raw, symbol: str, timeframe: str) -> Bar:
    """把 alpaca-py bar 对象转成 trader.models.Bar（duck typing 兼容）。"""
    return Bar(
        symbol=symbol,
        timestamp=getattr(raw, "timestamp", utc_now()),
        open=float(raw.open),
        high=float(raw.high),
        low=float(raw.low),
        close=float(raw.close),
        volume=float(raw.volume),
        timeframe=timeframe,
    )


class Runtime:
    """
    计划驱动 Pipeline 主运行时。

    用法::

        rt = Runtime(config)
        rt.run()      # 阻塞，直到 stop() 或 Ctrl-C
        rt.stop()     # 从其他线程调用
    """

    def __init__(self, config: TradingConfig) -> None:
        self._cfg = config
        is_paper = config.broker_type == "alpaca_paper"

        self._kill = FileKillSwitch()
        self._calendar = SimpleMarketCalendar()
        self._watchdog = HeartbeatWatchdog(db_path=config.db_path)
        self._broker = AlpacaBroker(
            config.alpaca_api_key, config.alpaca_secret_key, paper=is_paper
        )
        # data_feed_type 之前是"设了但没人读"：main.py 有 --data-feed 参数、
        # .env 有 DATA_FEED_TYPE，帮助文本也写着能选数据源，但这里无论如何
        # 都硬编码 AlpacaDataFeed —— 设成别的值会被静默忽略。目前确实只实现
        # 了 Alpaca 一种，那就明确拒绝不支持的取值，而不是假装接受。
        feed_type = (config.data_feed_type or "alpaca").strip().lower()
        if feed_type != "alpaca":
            raise ValueError(
                f"UNSUPPORTED_DATA_FEED: {config.data_feed_type!r}（当前只实现了 alpaca）"
            )
        self._feed = AlpacaDataFeed(config)
        self._portfolio = Portfolio(config)
        self._audit = AuditLog(config)
        self._risk = RiskEngine(config)
        self._selector = AICandidateSelector()
        self._planner = ATRPlanner()
        self._allocator = EqualWeightAllocator(
            max_position_pct=config.risk.max_position_pct,
        )
        if config.auto_trade_paper and config.broker_type != "alpaca_paper":
            raise ValueError("AUTO_TRADE_REQUIRES_ALPACA_PAPER")
        self._pos_monitor = StopTakeProfitMonitor()
        # Fill notifications and the daily review push (below) are triggered
        # by the running engine itself, not a stray background job — same
        # authorization tier as the manual "发送" buttons, which already opt
        # in via external_send_enabled=True. Without it here, every fill/
        # review message was silently BLOCKED behind the
        # DISCORD_EXTERNAL_SEND_ENABLED gate (see morning_brief.py fix).
        self._notifier = DiscordNotifier(external_send_enabled=True)
        self._price_news = PriceMoveSource(
            universe=config.symbols, timeframe=config.timeframe
        )
        # "us" 频道已从 channels 移除：华尔街见闻的 us-channel 现在稳定返回
        # code=20000 + items=[]（频道已废弃），不报错、只是永远空手而归。
        # 保留它只会每轮多打一次无用请求。global 频道本身已覆盖美股要闻。
        self._wscn = WallStreetCNSource(
            universe=config.symbols, channels=["global"],
        )
        self._sec = SECEdgarSource(universe=config.symbols)
        self._finnhub = FinnhubSource(universe=config.symbols)
        self._news_store = NewsEventStore(config.db_path)
        self._reviewer = SimpleReviewer(db_path=config.db_path)

        self._running = False
        self._tick_count = 0
        self._open_orders: Dict[str, OrderIntent] = {}
        self._order_store = OrderIntentStore(config.db_path)
        self._position_plan_store = PositionPlanStore(config.db_path)
        self._position_plan_projector = PositionPlanFillProjector(
            self._position_plan_store
        )
        self._invalidation_event_store = InvalidationEventStore(
            config.db_path
        )
        self._position_adjustment_store = PositionAdjustmentStore(
            config.db_path
        )
        self._position_quality_store = PositionQualityStore(config.db_path)
        self._production_evidence = ProductionEvidenceCoordinator(
            config.db_path
        )
        self._production_operations = ProductionOperationsCoordinator(
            config.db_path,
            ai_db_path=(
                config.daily_research_db
                if config.daily_research_enabled
                else config.ai_score_db
            ),
        )
        self._execution_pipeline_store = ExecutionPipelineStore(
            config.db_path
        )
        self._trade_episode_store = TradeEpisodeStore(config.db_path)
        self._post_trade_learning = PostTradeLearningCoordinator(
            config.db_path
        )
        self._signal_store = SignalStore(config.db_path)
        self._bug_reporter = BugReporter(config.db_path, "runtime")
        self._reconciliation_blocked = False
        daily_research = bool(config.daily_research_enabled)
        self._decision_service = PaperDecisionService(
            allow_without_ai=config.allow_quant_without_ai,
            min_ai_score=config.min_ai_score,
            ai_max_age_minutes=(
                int(config.daily_research_max_age_hours * 60)
                if daily_research
                else int(config.ai_score_max_age_minutes)
            ),
            ai_min_contributors=1 if daily_research else config.ai_min_contributors,
            ai_min_weight_coverage=(
                1.0 if daily_research else config.ai_min_weight_coverage
            ),
            allow_short=config.risk.allow_short,
        )
        self._strategy_stats = StrategyStatisticsRepository.from_json(config.strategy_statistics_path)
        self._universe_provider = UniverseProvider(config.symbols, config.universe_max_symbols, config.universe_max_age_minutes)
        self._universe_snapshot = self._universe_provider.provide(cli_symbols=config.symbols, now=utc_now())
        self._daily_research = DailyRuntimeSupport(
            config,
            self._universe_snapshot.symbols,
            on_error=lambda code, summary: self._report_exception(
                error_code=code, summary=summary
            ),
        )
        self._live_plans: Dict[str, TradePlan] = {}  # symbol → 当前活跃计划
        self._monitor_plans: Dict[str, TradePlan] = {}
        self._daily_start_set = False
        self._last_review_date: Optional[str] = None
        self._last_brief_date: Optional[str] = None

        logger.info(
            "Runtime init symbols=%s tf=%s auto_trade_paper=%s paper=%s",
            config.symbols, config.timeframe, config.auto_trade_paper, is_paper,
        )

    # ── 公共接口 ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        """启动计划驱动循环，阻塞直到 stop() 或 Ctrl-C。"""
        logger.info("Runtime start interval=%ds", self._cfg.poll_interval_secs)
        self._running = True
        self._run_reconciliation()

        try:
            equity = self._broker.get_account_equity()
            if equity > 0:
                self._risk.set_daily_start(equity)
                self._daily_start_set = True
                logger.info("Daily start equity: %.2f", equity)
        except Exception as exc:
            logger.warning("获取初始权益失败: %s", exc)

        while self._running:
            try:
                self._tick()
                time.sleep(self._cfg.poll_interval_secs)
            except KeyboardInterrupt:
                logger.info("Runtime stopped by user")
                break
            except Exception as exc:
                self._bug_reporter.capture_exception(
                    exc,
                    operation="runtime.tick",
                    context={"tick_count": self._tick_count},
                )
                logger.error(
                    "Runtime tick failed: %s", type(exc).__name__, exc_info=True
                )

        self._daily_research.close()
        logger.info("Runtime stopped")

    def _recover_incomplete_adjustments(self) -> list[str]:
        store = getattr(self, "_position_adjustment_store", None)
        if store is None:
            return []
        errors: list[str] = []
        for adjustment in store.list_incomplete():
            if adjustment.status.value != "PLANNED":
                continue
            current = self._position_plan_store.current(
                adjustment.position_plan_id
            )
            if current is None:
                errors.append(
                    "POSITION_ADJUSTMENT_PLAN_MISSING:"
                    f"{adjustment.adjustment_id}"
                )
                continue
            try:
                order_plan = store.order_plan_for(
                    adjustment,
                    plan=current,
                )
                prepared_intent = self._execute_via_pipeline(
                    order_plan,
                    0.0,
                    self._portfolio.positions,
                )
                if prepared_intent is None:
                    raise RuntimeError(
                        "POSITION_ADJUSTMENT_INTENT_NOT_PREPARED"
                    )
                key = prepared_intent.idempotency_key
                row = self._order_store.get_by_key(key)
                if row is None:
                    errors.append(
                        "POSITION_ADJUSTMENT_ORDER_NOT_CREATED:"
                        f"{adjustment.adjustment_id}"
                    )
                    continue
                store.link_order(
                    adjustment.adjustment_id,
                    order_intent_id=str(row["intent_id"]),
                    order_idempotency_key=key,
                )
            except Exception as exc:
                errors.append(
                    "POSITION_ADJUSTMENT_RECOVERY_FAILED:"
                    f"{adjustment.adjustment_id}:{type(exc).__name__}"
                )
        return errors

    def _position_plan_reconciliation_errors(self) -> list[str]:
        store = getattr(self, "_position_plan_store", None)
        if store is None:
            return []
        errors: list[str] = []
        positions = self._portfolio.positions
        for plan in store.recover_open():
            position = positions.get(plan.symbol)
            local_qty = float(position.qty) if position is not None else 0.0
            if abs(local_qty - plan.open_quantity) > 1e-8:
                errors.append(
                    "POSITION_PLAN_QUANTITY_MISMATCH:"
                    f"{plan.symbol}:plan={plan.open_quantity:g},"
                    f"local={local_qty:g}"
                )
        adjustment_store = getattr(
            self,
            "_position_adjustment_store",
            None,
        )
        if adjustment_store is None:
            return errors
        order_rows = {
            str(row.get("idempotency_key") or ""): row
            for row in self._order_store.list_all()
        }
        terminal_failure_states = {
            OrderLifecycle.CANCELED.value,
            OrderLifecycle.REJECTED.value,
        }
        for adjustment in adjustment_store.list_incomplete():
            if adjustment.status.value == "PLANNED":
                errors.append(
                    "POSITION_ADJUSTMENT_UNLINKED:"
                    f"{adjustment.adjustment_id}"
                )
                continue
            row = order_rows.get(adjustment.order_idempotency_key)
            if row is None:
                errors.append(
                    "POSITION_ADJUSTMENT_ORDER_MISSING:"
                    f"{adjustment.adjustment_id}"
                )
                continue
            state = str(row.get("state") or "")
            if state == OrderLifecycle.FILLED.value:
                adjustment_store.mark_completed_by_order_plan(
                    adjustment.order_plan_id
                )
            elif state in terminal_failure_states:
                errors.append(
                    "POSITION_ADJUSTMENT_ORDER_TERMINAL:"
                    f"{adjustment.adjustment_id}:{state}"
                )
        return errors

    def _run_reconciliation(self) -> None:
        try:
            recovery_errors = self._recover_incomplete_adjustments()
            local_rows = self._order_store.list_all()
            broker_orders = list(self._broker.get_open_orders())
            broker_positions = list(self._broker.get_positions())
            broker_fills = list(self._broker.get_recent_fills())
            unexplained_fills: list[str] = []

            for fill in broker_fills:
                local = next((row for row in local_rows if row.get("broker_order_id") == fill.order_id), None)
                if local:
                    fill.intent_id = local["intent_id"]
                    self._portfolio.apply_fill(fill)
                    projector = getattr(
                        self,
                        "_position_plan_projector",
                        None,
                    )
                    if projector is not None:
                        current_plan = (
                            self._position_plan_store.current_for_symbol(
                                fill.symbol
                            )
                        )
                        trade_plan = None
                        if fill.side == Side.BUY and current_plan is None:
                            trade_plan = (
                                self._position_plan_store.recover_trade_plan(
                                    str(local.get("plan_id") or "")
                                )
                            )
                        if current_plan is not None or trade_plan is not None:
                            projected_plan = projector.apply(
                                fill=fill,
                                applied_delta=None,
                                trade_plan=trade_plan,
                            )
                            episode_store = getattr(
                                self,
                                "_trade_episode_store",
                                None,
                            )
                            if (
                                episode_store is not None
                                and projected_plan is not None
                            ):
                                episode = episode_store.sync(
                                    projected_plan.position_plan_id
                                )
                                self._post_trade_learning.process(episode)
                    self._signal_store.apply_fill(
                        str(local.get("plan_id", "")), fill
                    )
                    cumulative_filled_qty = float(fill.filled_qty)
                    total_qty = float(local.get("qty") or 0.0)
                    remaining_qty = max(
                        total_qty - cumulative_filled_qty,
                        0.0,
                    )
                    self._order_store.update(
                        local["idempotency_key"],
                        filled_qty=cumulative_filled_qty,
                        remaining_qty=remaining_qty,
                        state=(
                            OrderLifecycle.FILLED.value
                            if remaining_qty <= 0
                            else OrderLifecycle.PARTIALLY_FILLED.value
                        ),
                    )
                    if remaining_qty <= 0:
                        adjustment_store = getattr(
                            self,
                            "_position_adjustment_store",
                            None,
                        )
                        if adjustment_store is not None:
                            adjustment_store.mark_completed_by_order_plan(
                                str(local.get("plan_id") or "")
                            )
                else:
                    unexplained_fills.append(str(fill.order_id))

            local_rows = self._order_store.list_all()
            active_states = {
                OrderLifecycle.PERSISTED.value,
                OrderLifecycle.SENDING.value,
                OrderLifecycle.ACKNOWLEDGED.value,
                OrderLifecycle.UNKNOWN.value,
                OrderLifecycle.OPEN.value,
                OrderLifecycle.PARTIALLY_FILLED.value,
                OrderLifecycle.CANCEL_REQUESTED.value,
            }
            for order in broker_orders:
                broker_id = str(order.get("id", "")) if isinstance(order, dict) else str(getattr(order, "id", ""))
                broker_client_id = (
                    order.get("client_order_id")
                    if isinstance(order, dict)
                    else getattr(order, "client_order_id", None)
                )
                local = next(
                    (
                        row
                        for row in local_rows
                        if row.get("state") in active_states
                        and (
                            str(row.get("broker_order_id") or "") == broker_id
                            or (
                                broker_client_id
                                and row.get("client_order_id")
                                == broker_client_id
                            )
                        )
                    ),
                    None,
                )
                if local:
                    intent = OrderIntent(
                        intent_id=local["intent_id"], signal_id=local.get("decision_id", ""),
                        symbol=local["symbol"], side=Side(local["side"]), qty=local["qty"],
                        order_type=local["order_type"], limit_price=local["limit_price"],
                        tif=local["tif"], idempotency_key=local["idempotency_key"],
                        client_order_id=local.get("client_order_id", ""),
                        decision_id=local.get("decision_id", ""),
                        plan_id=local.get("plan_id", ""),
                        candidate_plan_id=local.get(
                            "candidate_plan_id",
                            "",
                        ),
                        final_plan_id=local.get("final_plan_id", ""),
                        final_plan_version=int(
                            local.get("final_plan_version") or 0
                        ),
                        risk_check_id=local.get("risk_check_id", ""),
                        evidence_refs=tuple(
                            json.loads(
                                local.get("evidence_refs_json") or "[]"
                            )
                        ),
                    )
                    self._open_orders[broker_id] = intent
                    self._order_store.update(
                        local["idempotency_key"],
                        broker_order_id=broker_id,
                        state=(
                            OrderLifecycle.PARTIALLY_FILLED.value
                            if float(local.get("filled_qty") or 0.0) > 0
                            else OrderLifecycle.OPEN.value
                        ),
                    )

            local_rows = self._order_store.list_all()
            report = reconcile_broker_facts(
                broker_orders,
                broker_positions,
                broker_fills,
                local_rows,
                self._portfolio.positions.values(),
            )
            report.errors.extend(
                f"UNEXPLAINED_FILL:{order_id}"
                for order_id in unexplained_fills
            )
            report.errors.extend(recovery_errors)
            report.errors.extend(
                self._position_plan_reconciliation_errors()
            )
            quality_store = getattr(
                self,
                "_position_quality_store",
                None,
            )
            if quality_store is not None:
                try:
                    observed_at = utc_now()
                    quality_store.capture(
                        broker_positions=broker_positions,
                        local_positions=self._portfolio.positions.values(),
                        observed_at=observed_at,
                        observation_kind="REAL",
                    )
                    quality_store.build_report(
                        as_of=observed_at.astimezone(
                            ZoneInfo("America/New_York")
                        ).date(),
                        required_days=30,
                        observation_kind="REAL",
                    )
                except Exception as exc:
                    report.errors.append(
                        "POSITION_QUALITY_CAPTURE_FAILED:"
                        f"{type(exc).__name__}"
                    )
            report.ok = (
                not report.unexplained_orders
                and not report.unexplained_positions
                and not report.errors
            )
            self._audit.log_reconciliation(report)
            self._reconciliation_blocked = not report.ok
            if self._reconciliation_blocked:
                logger.error("startup reconciliation blocked trading")
        except Exception as exc:
            self._reconciliation_blocked = True
            report = ReconciliationReport(
                ok=False,
                errors=[type(exc).__name__],
            )
            try:
                self._audit.log_reconciliation(report)
            except Exception:
                logger.exception("failed to persist reconciliation failure")
            self._bug_reporter.capture_exception(exc, operation="runtime.reconciliation")
            logger.error("startup reconciliation failed: %s", type(exc).__name__)
    def stop(self) -> None:
        self._running = False

    def _make_decision_plan(
        self,
        cand,
        latest_bar: Bar,
        decision,
        *,
        current_qty: float,
        bars_history: list[Bar],
    ) -> TradePlan:
        """Translate a validated StrategyDecision into an ATR plan.

        Direction belongs to PaperDecision. Candidate score is confidence
        evidence only and must never infer or override BUY/SELL here.
        """
        return self._planner.make_plan(
            cand,
            latest_bar,
            params=decision.params,
            side=decision.side,
            current_qty=current_qty,
            bars_history=bars_history,
        )

    # ── 主循环 ───────────────────────────────────────────────────────────────

    def _tick(self) -> None:
        self._tick_count += 1
        ts = utc_now()
        logger.info("── tick #%d  %s ──", self._tick_count, ts.strftime("%H:%M:%S UTC"))

        self._daily_research.tick(ts)
        try:
            position_report = self._position_quality_store.build_report(
                as_of=ts.astimezone(ZoneInfo("America/New_York")).date(),
                required_days=30,
                observation_kind="REAL",
            )
            self._production_evidence.tick(
                now=ts,
                research_run=self._daily_research.store.latest_run(),
                position_report=position_report,
                reconciliation_blocked=self._reconciliation_blocked,
            )
        except Exception as exc:
            self._bug_reporter.capture_exception(
                exc,
                operation="runtime.production_evidence",
            )
            logger.error(
                "production evidence capture failed: %s",
                type(exc).__name__,
            )
        try:
            backup = self._production_operations.tick(now=ts)
            if backup["status"] == "FAILED":
                logger.error(
                    "production backup failed: %s",
                    backup["error_code"],
                )
        except Exception as exc:
            self._bug_reporter.capture_exception(
                exc,
                operation="runtime.production_backup",
            )
            logger.error(
                "production backup failed: %s",
                type(exc).__name__,
            )
        for expired in self._signal_store.invalidate_expired(ts):
            self._monitor_plans.pop(expired.symbol, None)

        # 1. Kill switch 急停检查
        if self._reconciliation_blocked:
            logger.warning("[BLOCKED] startup reconciliation incomplete")
            return
        if self._kill.engaged():
            logger.warning("Kill switch ENGAGED — skip tick #%d", self._tick_count)
            self._audit.log_heartbeat(self._tick_count, 0.0)
            return

        # 2. Watchdog 告警
        for alert in self._watchdog.check():
            logger.warning("[WATCHDOG] %s: %s", alert.level.upper(), alert.message)
            # 只有 critical 值得打扰人；warn 级别留在日志里，否则每次数据源
            # 抖动都会响一次。
            if str(alert.level).lower() == "critical":
                self._report_exception(
                    error_code=f"WATCHDOG_{alert.source}".upper(),
                    summary=alert.message,
                    evidence=[f"tick={self._tick_count}"],
                    at=ts,
                )

        # 3. 权益 + 持仓（broker 是权威来源）
        try:
            equity = self._broker.get_account_equity()
            positions: Dict[str, Position] = {
                p.symbol: p for p in self._broker.get_positions()
            }
        except Exception as exc:
            self._bug_reporter.capture_exception(
                exc, operation="broker.snapshot"
            )
            logger.error("broker 数据获取失败，跳过本轮: %s", exc)
            self._report_exception(
                error_code="BROKER_SNAPSHOT_FAILED",
                summary=(
                    f"无法从券商读取权益/持仓（{type(exc).__name__}）。"
                    "引擎已跳过本轮，不会基于陈旧数据做任何决策。"
                ),
                evidence=[f"tick={self._tick_count}"],
                at=ts,
            )
            return
        logger.info("equity=%.2f  positions=%d", equity, len(positions))
        # 配置的杠杆倍数用于放大仓位规模（等权分配 + 风控上限的资金基数），
        # 单笔止损风险仍按真实 equity 计算，不随杠杆放大。
        buying_power = equity * max(float(self._cfg.leverage or 1.0), 1.0)

        # 4. 日内起点（只设一次）
        if not self._daily_start_set and equity > 0:
            self._risk.set_daily_start(equity)
            self._daily_start_set = True

        # 5. 权益熔断
        self._risk.check_equity(equity)
        if self._risk.is_halted:
            logger.warning("风控熔断: %s", self._risk.halt_reason)
            self._report_exception(
                error_code="RISK_HALTED",
                summary=(
                    f"风控已熔断，本轮不再开新仓：{self._risk.halt_reason}。"
                    f"当前权益 ${equity:,.2f}。"
                ),
                evidence=[f"tick={self._tick_count}"],
                at=ts,
            )
            self._audit.log_heartbeat(self._tick_count, equity)
            return

        # 6. 轮询已提交订单
        self._poll_orders()

        # 6b. 晨报（美东 9AM，每天一次，不受市场时段限制）
        self._maybe_morning_brief(ts)

        # 6c. 新闻事件：四路合并（WSCN + SEC 8-K + Finnhub + 价格异动），不受市场
        # 时段限制 —— SEC 8-K、WSCN 快讯盘前盘后都可能发布，PriceMoveSource 读的是
        # 本地 bars 缓存，都不需要等本轮实时 K 线拉取成功。
        self._poll_news(ts)

        # 7. 市场时段判断
        session = self._calendar.session_now()
        if session == "closed":
            logger.info("市场已休市 — 仅做快照+心跳")
            self._portfolio.snapshot_external_equity(equity)
            self._audit.log_heartbeat(self._tick_count, equity)
            self._maybe_daily_review(ts)
            self._publish_runtime_status(
                ts, session, equity, positions, message="market closed"
            )
            return

        # 8. 拉取 K 线，更新本地缓存 —— 盘前/盘中/盘后都刷新（feed=sip 账号下
        # Alpaca 本就包含盘前盘后成交），供 UI 图表/回测/新闻异动侦测使用；
        # 是否进入选股/下单只由下面单独的 session=="open" 判断控制，不受影响。
        raw_bars_map, model_bars = self._fetch_and_cache_bars(ts)

        if not raw_bars_map:
            logger.warning("无可用 K 线数据")
            # 只有常规交易时段拿不到任何 K 线才算故障。盘前/盘后流动性稀疏，
            # 所有标的的最新 bar 都过期是常态，在那两个时段告警等于每天早晚
            # 各刷一次屏。
            if session == "open":
                self._report_exception(
                    error_code="MARKET_DATA_UNAVAILABLE",
                    summary=(
                        f"常规交易时段内 {len(self._cfg.symbols)} 个标的全部拿不到"
                        "新鲜 K 线，本轮不做任何交易决策。"
                    ),
                    evidence=[f"tick={self._tick_count}", f"session={session}"],
                    at=ts,
                )
            self._portfolio.snapshot_external_equity(equity)
            self._audit.log_heartbeat(self._tick_count, equity)
            self._publish_runtime_status(
                ts, session, equity, positions, message="no fresh bars"
            )
            return

        if session != "open":
            logger.info("market session=%s — no new trade plans outside regular hours", session)
            self._portfolio.snapshot_external_equity(equity)
            self._audit.log_heartbeat(self._tick_count, equity)
            self._publish_runtime_status(ts, session, equity, positions)
            return

        # 10. 持仓监控：止损/止盈触发 → 生成 CLOSE 计划并立即执行
        if self._live_plans and model_bars:
            triggered = self._pos_monitor.check(positions, self._live_plans, model_bars)
            for close_plan in triggered:
                logger.info(
                    "[POS_MONITOR] %s %s @ %.2f",
                    close_plan.symbol, close_plan.rationale[:40], close_plan.entry_price,
                )
                self._execute_via_pipeline(
                    close_plan,
                    equity,
                    positions,
                )

        # 11. 选股（AI 直选：每日深度研究名单 + 轻量技术确认，T0 regime 动态阈值）
        ai_scores = self._read_ai_scores(ts)
        score_threshold = _get_score_threshold()
        if score_threshold is None:
            logger.info("T0 高波动(HIGH_VOL) → 暂停选股本轮")
            self._portfolio.snapshot_external_equity(equity)
            self._audit.log_heartbeat(self._tick_count, equity)
            return
        try:
            candidates = self._selector.select(
                ai_scores=ai_scores,
                bars=raw_bars_map,
                score_threshold=score_threshold,
            )
            logger.info(
                "selection: %d candidates (AI shortlist, score≥%.0f, regime=%s)",
                len(candidates), score_threshold,
                _get_current_regime_label(),
            )
        except Exception as exc:
            self._bug_reporter.capture_exception(
                exc, operation="selection.select"
            )
            logger.error("selection.select 失败: %s", exc, exc_info=True)
            candidates = []

        decisions_by_symbol = {}
        decisions = self._decision_service.decide(
            bars=raw_bars_map,
            positions=positions,
            candidates=candidates,
            strategy_statistics=self._strategy_stats,
            ai_advisories=ai_scores,
            market_regime=_get_current_regime_label(),
            now=ts,
            timeframe=self._cfg.timeframe,
            universe_version=self._universe_snapshot.universe_version,
            data_version=ts.isoformat(),
        )
        decisions_by_symbol = {decision.symbol: decision for decision in decisions}
        for decision in decisions:
            self._audit.log_strategy_decision(decision)
        candidates = [candidate for candidate in candidates if candidate.symbol in decisions_by_symbol]
        # 12. 计划生成（ATRPlanner）
        # 过滤已有未成交挂单的标的，防止同一 tick 重复提交 LMT
        symbols_with_open_orders = {intent.symbol for intent in self._open_orders.values()}
        if symbols_with_open_orders:
            before = len(candidates)
            candidates = [c for c in candidates if c.symbol not in symbols_with_open_orders]
            if len(candidates) < before:
                logger.info(
                    "跳过已有挂单的标的（防重复）: %s",
                    sorted(symbols_with_open_orders),
                )

        raw_plans: List[TradePlan] = []
        for cand in candidates:
            raw = raw_bars_map.get(cand.symbol)
            if not raw:
                continue
            try:
                qty_held = positions[cand.symbol].qty if cand.symbol in positions else 0.0
                bars_history = [
                    _alpaca_bar_to_model(b, cand.symbol, self._cfg.timeframe)
                    for b in raw[:-1]
                ]
                decision = decisions_by_symbol.get(cand.symbol)
                if decision is None:
                    logger.warning(
                        "PaperDecision missing for %s; skip plan generation",
                        cand.symbol,
                    )
                    continue
                plan = self._make_decision_plan(
                    cand,
                    _alpaca_bar_to_model(raw[-1], cand.symbol, self._cfg.timeframe),
                    decision,
                    current_qty=qty_held,
                    bars_history=bars_history,
                )
                plan.source = "paper_decision"
                plan.metadata.update(
                    {
                        "decision_id": decision.decision_id,
                        "strategy": decision.strategy,
                        "strategy_version": decision.strategy_version,
                        "strategy_statistics_id": (
                            decision.strategy_statistics_id
                        ),
                        "universe_version": decision.universe_version,
                        "data_version": decision.data_version,
                        "valid_until": decision.valid_until,
                    }
                )
                self._audit.link_decision_plan(decision.decision_id, plan.plan_id)
                raw_plans.append(plan)
                logger.info(
                    "Plan [%s] action=%s entry=%.2f stop=%.2f tp=%.2f",
                    plan.symbol, plan.action,
                    plan.entry_price, plan.stop_loss, plan.take_profit,
                )
            except Exception as exc:
                logger.warning("ATRPlanner %s 失败: %s", cand.symbol, exc)

        if not raw_plans:
            logger.info("本轮无计划生成")
            self._portfolio.snapshot_external_equity(equity)
            self._audit.log_heartbeat(self._tick_count, equity)
            self._publish_runtime_status(
                ts, session, equity, positions, bars=model_bars
            )
            return

        # 13. 仓位分配（EqualWeightAllocator 填 qty / target_weight）
        pending_buy_notional: Dict[str, float] = {}
        try:
            pending_buy_notional = (
                self._order_store.pending_buy_notional_by_symbol()
            )
            plans = self._allocator.allocate(
                raw_plans,
                equity,
                positions,
                pending_buy_notional=pending_buy_notional,
                buying_power=buying_power,
            )
        except Exception as exc:
            self._bug_reporter.capture_exception(
                exc, operation="allocator.allocate"
            )
            logger.error("allocator.allocate 失败: %s", exc, exc_info=True)
            plans = []

        # 14. AI 安全门 + 确定性风控
        # 14a. AI 自动交易模式：用 ai_states.duckdb 的综合评分更新 plan.confidence
        if self._cfg.auto_trade_paper:
            if ai_scores:
                for plan in plans:
                    snapshot = ai_scores.get(plan.symbol)
                    result = AIScoreValidator(
                        self._ai_score_policy(self._cfg.min_ai_score)
                    ).validate(snapshot)
                    if result.valid:
                        plan.confidence = result.score / 100.0
                        logger.info(
                            "AI score %s: %.1f → confidence=%.2f",
                            plan.symbol, result.score if result.score is not None else 0.0, plan.confidence,
                        )
                    else:
                        plan.confidence = (self._cfg.min_ai_score - 1) / 100.0
                        plan.status = "REJECTED"
                        logger.info("AI safety gate %s: %s", plan.symbol, result.reason_code)
                        self._audit.log_ai_safety_event(plan, result, self._cfg)
            else:
                logger.warning("AI safety gate: no readable scores from %s", self._cfg.ai_score_db)
                validator = AIScoreValidator(
                    self._ai_score_policy(self._cfg.min_ai_score)
                )
                for plan in plans:
                    plan.status = "REJECTED"
                    result = validator.validate(None)
                    self._audit.log_ai_safety_event(plan, result, self._cfg)

        for plan in plans:
            if plan.status == "REJECTED":
                continue
            verdict = self._risk.evaluate_plan(
                plan,
                equity,
                positions,
                pending_buy_notional=pending_buy_notional,
                buying_power=buying_power,
            )
            self._audit.log_plan_risk_event(
                plan,
                verdict,
                phase="PLAN",
            )
            if not verdict.approved:
                logger.info("Plan [%s] 风控拒绝: %s", plan.symbol, verdict.reason)
                plan.status = "REJECTED"
                self._audit.log_trade_plan(plan)
                continue

            decision = decisions_by_symbol.get(plan.symbol)
            latest_bar = model_bars.get(plan.symbol)
            if decision is not None and latest_bar is not None:
                try:
                    report = build_ready_signal_report(
                        plan,
                        decision,
                        ai_scores.get(plan.symbol),
                        latest_bar,
                        equity=equity,
                        now=ts,
                        timeframe=self._cfg.timeframe,
                    )
                    _stored_report, created = self._signal_store.register_ready(report)
                    if created or plan.symbol not in self._monitor_plans:
                        self._monitor_plans[plan.symbol] = plan
                except Exception as exc:
                    logger.warning(
                        "SignalReport %s failed: %s", plan.symbol, exc
                    )

            if not self._cfg.auto_trade_paper:
                plan.status = "DRY_RUN"
                self._audit.log_trade_plan(plan)
                logger.info(
                    "[DRY-RUN] plan=%s %s %s qty=%.0f @ %.2f (auto_trade_paper=False)",
                    plan.plan_id[:8], plan.symbol, plan.side.value, plan.qty, plan.entry_price,
                )
                continue

            plan.status = "READY"
            self._audit.log_trade_plan(plan)
            self._execute_via_pipeline(
                plan,
                equity,
                positions,
                risk_verdict=verdict,
            )

        # 16. 快照 + 心跳
        self._portfolio.snapshot_external_equity(equity)
        self._audit.log_heartbeat(self._tick_count, equity)
        self._publish_runtime_status(
            ts, session, equity, positions, bars=model_bars
        )

        # 17. 盘后复盘（每日一次，21:00 UTC 后触发）
        self._maybe_daily_review(ts)

    def _publish_runtime_status(
        self,
        ts: datetime,
        session: str,
        equity: float,
        positions: Dict[str, Position],
        *,
        bars: Optional[Dict[str, Bar]] = None,
        message: str = "",
    ) -> None:
        self._daily_research.publish_status(
            now=ts,
            tick_count=self._tick_count,
            session=session,
            equity=equity,
            reconciliation_blocked=self._reconciliation_blocked,
            kill_switch=self._kill.engaged(),
            bars=bars,
            positions=positions,
            plans=self._monitor_plans,
            open_orders=self._open_orders.values(),
            message=message,
        )

    # ── 执行单个计划 ─────────────────────────────────────────────────────────

    def _execute_via_pipeline(
        self,
        plan: TradePlan,
        equity: float,
        positions: Dict[str, Position],
        *,
        risk_verdict: RiskVerdict | None = None,
    ) -> OrderIntent | None:
        legacy_key = idempotency_key(
            plan.plan_id,
            plan.symbol,
            plan.side.value,
            plan.qty,
            plan.entry_price,
            plan.action,
        )
        legacy = self._order_store.get_by_key(legacy_key)
        if legacy and (
            legacy.get("broker_order_id")
            or legacy.get("state")
            in {
                OrderLifecycle.SENDING.value,
                OrderLifecycle.UNKNOWN.value,
                OrderLifecycle.OPEN.value,
                OrderLifecycle.PARTIALLY_FILLED.value,
            }
        ):
            return None
        store = getattr(self, "_execution_pipeline_store", None)
        if store is None:
            store = ExecutionPipelineStore(self._cfg.db_path)
            self._execution_pipeline_store = store
        now = max(utc_now(), plan.created_at)
        metadata = plan.metadata
        decision_id = str(
            metadata.get("decision_id") or plan.plan_id
        )
        evidence_refs = tuple(
            str(value)
            for value in (
                metadata.get("decision_id"),
                metadata.get("strategy_statistics_id"),
                metadata.get("data_version"),
                metadata.get("invalidation_event_id"),
                metadata.get("position_plan_version_id"),
            )
            if value
        ) or (f"runtime-plan:{plan.plan_id}",)
        valid_until = metadata.get("valid_until")
        if not isinstance(valid_until, datetime) or valid_until <= now:
            valid_until = now + timedelta(minutes=5)
        candidate = candidate_from_trade_plan(
            plan,
            decision_id=decision_id,
            strategy_version=str(
                metadata.get("strategy_version")
                or metadata.get("strategy")
                or "runtime-v1"
            ),
            data_version=str(
                metadata.get("data_version")
                or f"runtime-plan:{plan.plan_id}"
            ),
            evidence_refs=evidence_refs,
            valid_until=valid_until,
        )
        existing_candidate = store.get_candidate(
            candidate.candidate_plan_id
        )
        if (
            existing_candidate is not None
            and existing_candidate.status
            == CandidatePlanStatus.REJECTED
        ):
            plan.status = "REJECTED"
            return None
        if existing_candidate is None:
            store.register_candidate(candidate)
            store.validate_candidate(
                candidate.candidate_plan_id,
                now=now,
            )
        elif existing_candidate.status == CandidatePlanStatus.DRAFT:
            store.validate_candidate(
                candidate.candidate_plan_id,
                now=now,
            )
        verdict = risk_verdict
        if verdict is None:
            if plan.action in {"CLOSE", "REDUCE"}:
                verdict = RiskVerdict(
                    True,
                    "退出或减仓不增加敞口",
                    suggested_qty=plan.qty,
                )
            else:
                verdict = self._risk.evaluate_plan(
                    plan,
                    equity,
                    positions,
                    pending_buy_notional=(
                        self._order_store
                        .pending_buy_notional_by_symbol()
                    ),
                    buying_power=equity * max(float(self._cfg.leverage or 1.0), 1.0),
                )
        if not verdict.approved:
            store.reject_candidate(
                candidate.candidate_plan_id,
                now=now,
            )
            plan.status = "REJECTED"
            audit = getattr(self, "_audit", None)
            if audit is not None:
                audit.log_plan_risk_event(
                    plan,
                    verdict,
                    phase="PRE_SUBMIT",
                )
                audit.log_trade_plan(plan)
            return None
        final = store.get_final_by_candidate(
            candidate.candidate_plan_id
        )
        if final is None:
            final = store.finalize(
                candidate.candidate_plan_id,
                risk_verdict=verdict,
                risk_check_id=(
                    "risk-check-"
                    + hashlib.sha256(
                        "|".join(
                            (
                                candidate.candidate_plan_id,
                                verdict.reason,
                                f"{verdict.suggested_qty:.8f}",
                            )
                        ).encode()
                    ).hexdigest()[:24]
                ),
                risk_config_version="risk-config-v1",
                now=now,
            )
        if final.status == FinalTradePlanStatus.FINAL_EXECUTABLE:
            intent = store.create_order_intent(
                final.final_plan_id,
                now=now,
            )
        else:
            intent = store.get_intent_for_final(final.final_plan_id)
            if intent is None:
                raise RuntimeError("PIPELINE_INTENT_REFERENCE_MISSING")
        intent.plan_id = plan.plan_id
        self._submit_pipeline_intent(
            plan,
            equity,
            positions,
            intent=intent,
        )
        return intent

    def process_invalidation_event(
        self,
        event: InvalidationEvent,
        *,
        limit_price: float | None,
        received_at: datetime | None = None,
    ) -> PositionAdjustment:
        """Validate one fact and create at most one durable adjustment order."""
        existing = self._position_adjustment_store.get_by_event(
            event.event_id
        )
        if existing is not None:
            return existing
        current = self._position_plan_store.current(
            event.position_plan_id
        )
        if current is None:
            raise ValueError("INVALIDATION_POSITION_PLAN_NOT_FOUND")
        self._invalidation_event_store.record(
            event,
            plan=current,
            received_at=received_at,
        )
        adjustment, order_plan, _ = (
            self._position_adjustment_store.prepare(
                event,
                plan=current,
                limit_price=limit_price,
            )
        )
        if order_plan is not None:
            prepared_intent = self._execute_via_pipeline(
                order_plan,
                0.0,
                self._portfolio.positions,
            )
            if prepared_intent is not None:
                key = prepared_intent.idempotency_key
                row = self._order_store.get_by_key(key)
                if row is not None:
                    adjustment = self._position_adjustment_store.link_order(
                        adjustment.adjustment_id,
                        order_intent_id=str(row["intent_id"]),
                        order_idempotency_key=key,
                    )
        return adjustment

    def _submit_pipeline_intent(
        self,
        plan: TradePlan,
        equity: float,
        positions: Dict[str, Position],
        *,
        intent: OrderIntent,
    ) -> None:
        """将通过 AI 安全门和确定性风控的计划转成 LMT 限价单。

        仅 AI 自动模拟盘模式可提交；kill switch 始终可紧急拦截。
        """
        if not self._cfg.auto_trade_paper:
            logger.warning("[BLOCKED] plan=%s AI automatic paper trading is disabled", plan.plan_id[:8])
            return
        if self._cfg.broker_type != "alpaca_paper":
            logger.error(
                "[BLOCKED] plan=%s automatic submission requires alpaca_paper",
                plan.plan_id[:8],
            )
            return
        if self._reconciliation_blocked:
            logger.warning("[BLOCKED] startup reconciliation incomplete")
            return
        if self._kill.engaged():
            logger.warning(
                "[BLOCKED] plan=%s kill switch engaged", plan.plan_id[:8]
            )
            return

        if plan.action in {"CLOSE", "REDUCE"}:
            self._signal_store.mark_exit(
                plan.symbol, plan_id=plan.plan_id, at=utc_now()
            )

        key = intent.idempotency_key
        existing = self._order_store.get_by_key(key)
        if existing and (existing.get("broker_order_id") or existing.get("state") in {OrderLifecycle.SENDING.value, OrderLifecycle.UNKNOWN.value, OrderLifecycle.OPEN.value, OrderLifecycle.PARTIALLY_FILLED.value}):
            logger.info("idempotent order already exists plan=%s", plan.plan_id[:8])
            return
        increases_exposure = plan.action not in {"CLOSE", "REDUCE"}
        if increases_exposure:
            try:
                pending_buy_notional = (
                    self._order_store.pending_buy_notional_by_symbol()
                    if plan.side == Side.BUY
                    else {}
                )
                verdict = self._risk.evaluate_plan(
                    plan,
                    equity,
                    positions,
                    pending_buy_notional=pending_buy_notional,
                    # Must match the buying_power used when this plan was
                    # first approved in _tick() — omitting it here would
                    # silently re-check against bare equity and could reject
                    # an already-approved leveraged plan right before
                    # submission.
                    buying_power=equity * max(float(self._cfg.leverage or 1.0), 1.0),
                )
                audit = getattr(self, "_audit", None)
                if audit is not None:
                    audit.log_plan_risk_event(
                        plan,
                        verdict,
                        phase="PRE_SUBMIT",
                    )
            except Exception as exc:
                plan.status = "REJECTED"
                self._bug_reporter.capture_exception(
                    exc,
                    operation="risk.pre_submit",
                    symbol=plan.symbol,
                    plan_id=plan.plan_id,
                )
                logger.error(
                    "[BLOCKED] pre-submit risk unavailable for %s",
                    plan.symbol,
                )
                return
            if not verdict.approved:
                plan.status = "REJECTED"
                audit = getattr(self, "_audit", None)
                if audit is not None:
                    audit.log_trade_plan(plan)
                logger.warning(
                    "[BLOCKED] pre-submit risk rejected %s: %s",
                    plan.symbol,
                    verdict.reason,
                )
                return
        else:
            audit = getattr(self, "_audit", None)
            if audit is not None:
                audit.log_plan_risk_event(
                    plan,
                    RiskVerdict(
                        True,
                        "退出或减仓不增加敞口",
                        suggested_qty=plan.qty,
                    ),
                    phase="PRE_SUBMIT",
                )
        self._order_store.persist(intent, key, plan.plan_id)
        self._order_store.update(key, state=OrderLifecycle.SENDING.value)
        try:
            broker_id = self._broker.place_order(intent)
            self._open_orders[broker_id] = intent
            self._order_store.update(key, state=OrderLifecycle.OPEN.value, broker_order_id=broker_id, submitted_at=utc_now())
            self._live_plans[plan.symbol] = plan
            self._risk.record_success()
            logger.info(
                "ORDER submitted %s %s %s qty=%.0f @ %.2f  broker_id=%s",
                plan.symbol, plan.side.value, plan.action,
                intent.qty, intent.limit_price, broker_id,
            )
        except Exception as exc:
            self._bug_reporter.capture_exception(
                exc,
                operation="broker.place_order",
                symbol=plan.symbol,
                plan_id=plan.plan_id,
                intent_id=intent.intent_id,
                context={"client_order_id": intent.client_order_id},
            )
            logger.error(
                "place_order [%s] failed; state UNKNOWN: %s",
                plan.symbol,
                type(exc).__name__,
            )
            self._order_store.update(
                key,
                state=OrderLifecycle.UNKNOWN.value,
                last_error=type(exc).__name__,
            )
            self._risk.record_failure()

    # ── 轮询订单状态 ─────────────────────────────────────────────────────────

    def _poll_orders(self) -> None:
        if not self._open_orders:
            return
        done: List[str] = []
        for broker_id, intent in list(self._open_orders.items()):
            try:
                status = self._broker.get_order_status(broker_id)
            except Exception as exc:
                logger.warning("get_order_status %s 失败: %s", broker_id, exc)
                continue
            if status in (OrderStatus.FILLED, OrderStatus.PARTIAL):
                try:
                    fill = self._broker.get_fill(broker_id)
                except Exception as exc:
                    # Do not mark this order done on a failed fill lookup —
                    # leave it in _open_orders so the next tick retries.
                    # Applying a fill or removing the order without one would
                    # silently lose the fill (see get_fill's docstring note).
                    logger.warning("get_fill %s 失败: %s", broker_id, exc)
                    continue
                if fill is not None:
                    fill.intent_id = intent.intent_id
                    cumulative_filled_qty = float(fill.filled_qty)
                    self._portfolio.apply_fill(fill)
                    projected_plan = self._position_plan_projector.apply(
                        fill=fill,
                        applied_delta=None,
                        trade_plan=self._live_plans.get(fill.symbol),
                    )
                    episode_store = getattr(
                        self,
                        "_trade_episode_store",
                        None,
                    )
                    if (
                        episode_store is not None
                        and projected_plan is not None
                    ):
                        episode = episode_store.sync(
                            projected_plan.position_plan_id
                        )
                        self._post_trade_learning.process(episode)
                    self._signal_store.apply_fill(intent.plan_id, fill)
                    if intent.idempotency_key:
                        self._order_store.update(intent.idempotency_key, filled_qty=cumulative_filled_qty, remaining_qty=max(0.0, intent.qty - cumulative_filled_qty), state=(OrderLifecycle.FILLED.value if status == OrderStatus.FILLED else OrderLifecycle.PARTIALLY_FILLED.value))
                    self._risk.record_success()
                    logger.info(
                        "FILLED %s %s qty=%.0f @ %.4f",
                        fill.symbol, fill.side.value, fill.filled_qty, fill.avg_price,
                    )
                    self._notifier.send(Notification(
                        title=f"成交: {fill.symbol}",
                        body=(
                            f"{fill.side.value} qty={fill.filled_qty:.0f} "
                            f"@ {fill.avg_price:.2f}"
                        ),
                        kind="plan",
                    ))
                # BUY 成交后从 _live_plans 清理（允许后续重新开仓）
                if fill is not None and fill.side == Side.SELL:
                    self._live_plans.pop(fill.symbol, None)
                    self._monitor_plans.pop(fill.symbol, None)
                if status == OrderStatus.FILLED:
                    if fill is None:
                        # Broker reports FILLED but has no fill details yet
                        # (a benign race between status and fill endpoints) —
                        # retry next tick instead of marking done without
                        # ever having applied the fill.
                        logger.warning(
                            "%s status=FILLED but get_fill returned None — retry next tick",
                            broker_id,
                        )
                        continue
                    adjustment_store = getattr(
                        self,
                        "_position_adjustment_store",
                        None,
                    )
                    if adjustment_store is not None:
                        adjustment_store.mark_completed_by_order_plan(
                            intent.plan_id
                        )
                    done.append(broker_id)
            elif status in (OrderStatus.CANCELLED, OrderStatus.REJECTED):
                logger.info("Order closed %s status=%s", broker_id, status.value)
                # 订单取消/拒绝：也从 _live_plans 清理，允许重新进入选股
                self._live_plans.pop(intent.symbol, None)
                self._monitor_plans.pop(intent.symbol, None)
                if intent.idempotency_key:
                    self._order_store.update(intent.idempotency_key, state=(OrderLifecycle.CANCELED.value if status == OrderStatus.CANCELLED else OrderLifecycle.REJECTED.value))
                done.append(broker_id)
        for broker_id in done:
            self._open_orders.pop(broker_id, None)

    # ── 每日复盘 ─────────────────────────────────────────────────────────────

    def _read_ai_scores(
        self, now: Optional[datetime] = None
    ) -> Dict[str, AIScoreSnapshot]:
        """Read today's frozen research; legacy snapshots are opt-in fallback."""
        if self._cfg.daily_research_enabled:
            return self._daily_research.snapshots(now or utc_now())
        try:
            from .ai.manager import get_score_snapshots_from_db
            return get_score_snapshots_from_db(self._cfg.ai_score_db)
        except Exception as exc:
            logger.warning("_read_ai_scores 失败: %s", exc)
            return {}

    def _ai_score_policy(self, minimum: float) -> AIScorePolicy:
        if self._cfg.daily_research_enabled:
            return AIScorePolicy(
                minimum,
                self._cfg.daily_research_max_age_hours * 60,
                min_contributors=1,
                min_weight_coverage=1.0,
                require_llm=True,
            )
        return AIScorePolicy(
            minimum,
            self._cfg.ai_score_max_age_minutes,
            min_contributors=(
                1
                if self._cfg.allow_quant_without_ai
                else self._cfg.ai_min_contributors
            ),
            min_weight_coverage=(
                0.0
                if self._cfg.allow_quant_without_ai
                else self._cfg.ai_min_weight_coverage
            ),
            require_llm=not self._cfg.allow_quant_without_ai,
        )

    def _report_exception(
        self,
        *,
        error_code: str,
        summary: str,
        evidence: Optional[List[str]] = None,
        at: Optional[datetime] = None,
    ) -> None:
        """把运行期故障推给读者。

        在这之前，引擎的故障只写日志：watchdog 已经在喊"心跳超时 194222s"，
        风控熔断、broker 断连也都记录得很完整——但没有任何一条会送到人眼前，
        而这些恰恰是最需要立刻知道的事。

        告警链路本身绝不能拖垮主循环：这里吞掉所有异常，最多留一行 debug。
        推送失败不该让一个本来只是"数据源抖了一下"的 tick 变成崩溃。
        """
        try:
            from .daily_discord import build_exception_message

            self._notifier.send(
                build_exception_message(
                    error_code=error_code,
                    summary=summary,
                    evidence_refs=list(evidence or []),
                    occurred_at=at or datetime.now(timezone.utc),
                )
            )
        except Exception as exc:  # noqa: BLE001 - 告警失败不能影响交易主流程
            logger.debug("异常告警推送失败 %s: %s", error_code, exc)

    def _maybe_morning_brief(self, ts: datetime) -> None:
        """美东 9AM（UTC 13h/14h）自动发送晨报，每天只发一次。"""
        try:
            from .morning_brief import should_send_brief, send_morning_brief
            if not should_send_brief(ts, self._last_brief_date):
                return
            logger.info("发送晨报...")
            send_morning_brief(
                symbols=list(self._cfg.symbols),
                db_path=self._cfg.db_path,
            )
            self._last_brief_date = ts.strftime("%Y-%m-%d")
        except Exception as exc:
            logger.warning("晨报发送失败: %s", exc)

    def _fetch_and_cache_bars(
        self, ts: datetime
    ) -> tuple[Dict[str, list], Dict[str, Bar]]:
        """Fetch latest bars per symbol and upsert them into the local cache.

        Called every tick regardless of market session — this is data
        acquisition for monitoring/display/research, not a trading decision.
        Symbols with too few bars or a stale latest bar are simply omitted
        from the returned maps (extended-hours liquidity is naturally
        sparser, so gaps are expected, not an error).
        """
        raw_bars_map: Dict[str, list] = {}
        model_bars: Dict[str, Bar] = {}
        for symbol in self._cfg.symbols:
            try:
                raw = self._feed.fetch_bars(symbol, n_bars=self._cfg.bars_lookback)
                if len(raw) < 30:
                    logger.warning("%s: 仅 %d 根 K 线，跳过", symbol, len(raw))
                    continue
                latest_bar = _alpaca_bar_to_model(raw[-1], symbol, self._cfg.timeframe)
                if not _latest_bar_is_fresh(latest_bar, self._cfg.timeframe, now=ts):
                    logger.warning(
                        "%s: stale latest bar %s — skip live decision",
                        symbol, latest_bar.timestamp,
                    )
                    continue
                raw_bars_map[symbol] = raw
                rows = [
                    {"timestamp_utc": b.timestamp,
                     "open": b.open, "high": b.high, "low": b.low,
                     "close": b.close, "volume": b.volume}
                    for b in raw
                ]
                _dc_upsert(symbol, self._cfg.timeframe, pd.DataFrame(rows))
                model_bars[symbol] = latest_bar
            except Exception as exc:
                logger.warning("fetch_bars %s 失败: %s", symbol, exc)
        return raw_bars_map, model_bars

    def _poll_news(self, ts: datetime) -> None:
        """Poll and persist news every tick, independent of market session."""
        news_sources = [
            ("wscn", self._wscn, ts),
            ("sec", self._sec, ts - timedelta(hours=20)),  # 8-K 可能盘前发
            ("finnhub", self._finnhub, ts - timedelta(hours=4)),
            ("price", self._price_news, ts),
        ]
        poll_all_sources(news_sources, self._news_store, now=ts)

    def _maybe_daily_review(self, ts: datetime) -> None:
        """每日美东 daily_review_hour_et 点（.env 可配置，默认 16）后触发一次复盘，当天只运行一次。"""
        today = ts.strftime("%Y-%m-%d")
        if today == self._last_review_date:
            return
        review_hour_et = ts.astimezone(ZoneInfo("America/New_York")).replace(
            hour=self._cfg.daily_review_hour_et, minute=0, second=0, microsecond=0
        )
        if ts < review_hour_et.astimezone(timezone.utc):
            return
        try:
            report = self._reviewer.review(period="daily", as_of=ts)
            trade_count = report.attribution.get("trade_count", len(report.trades))
            logger.info(
                "Daily review %s: pnl=%.2f trades=%d",
                today, report.portfolio_pnl, trade_count,
            )
            from .discord_report import build_daily_review_message
            self._notifier.send(build_daily_review_message(
                today=today,
                pnl=report.portfolio_pnl,
                trade_count=trade_count,
                trades=report.trades,
                market_summary=getattr(report, "market_summary", ""),
                symbols=list(self._cfg.symbols) if getattr(self._cfg, "symbols", None) else None,
            ))
            # Only mark today "done" once the review actually succeeded —
            # marking it before the try meant a transient DB-lock failure
            # (the live engine holds trade.duckdb open concurrently) would
            # permanently skip the day's review instead of being retried on
            # the next tick.
            self._last_review_date = today
        except Exception as exc:
            logger.warning("每日复盘失败: %s", exc)
