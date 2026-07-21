"""
trader/runtime.py
M1 计划驱动管道 Runtime。

与 scheduler.py 的区别
  scheduler.py  = 信号驱动（TA Signal → risk → order）用于实时策略回路。
  runtime.py    = 计划驱动（Candidate → TradePlan → allocate → AI safety → risk → LMT）
                  用于 AI 辅助决策流程。

每轮（tick）流程
  kill_switch → watchdog → equity/positions → risk.check_equity
  → poll_orders → market_session → fetch_bars → news
  → pos_monitor → selection → plan → allocate → evaluate_plan
  → AI safety → risk → execute/DRY_RUN → portfolio.snapshot → heartbeat → daily_review

安全红线（必须，不得绕过）
  - AI agent 不直连 broker：只产出 Advisory/TradePlan，由 Runtime 统一执行。
  - 仅 auto_trade_paper=True 且 broker_type=alpaca_paper 时自动下单。
  - 只挂 LMT：绝不下 market order（AlpacaBroker 也有防护，双重保险）。
  - AI 评分、确定性风控、kill switch、幂等与启动对账均不得绕过。
  - 密钥不入库：日志中不打印 API Key / Secret。
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import pandas as pd

from .allocator import EqualWeightAllocator
from .audit import AuditLog
from .broker.alpaca import AlpacaBroker
from .config import TradingConfig
from .data_cache import upsert_bars as _dc_upsert
from .data_feed import AlpacaDataFeed
from .market_calendar import SimpleMarketCalendar
from .models import (
    Bar, Notification, OrderIntent, OrderStatus,
    Position, Side, TradePlan, new_id, utc_now,
)
from .news import FinnhubSource, PriceMoveSource, SECEdgarSource, WallStreetCNSource
from .notify import DiscordNotifier
from .plan import ATRPlanner
from .portfolio import Portfolio
from .position_monitor import StopTakeProfitMonitor
from .review import SimpleReviewer
from .risk_engine import RiskEngine
from .order_lifecycle import OrderIntentStore, OrderLifecycle, idempotency_key, client_order_id, reconcile_broker
from .paper_decision import PaperDecisionService, StrategyStatisticsRepository, UniverseProvider
from .bug_reporting import BugReporter
from .ai.safety import AIScorePolicy, AIScoreSnapshot, AIScoreValidator
from .selection import ConsensusSelector
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
        is_paper = config.broker_type != "alpaca_live"

        self._kill = FileKillSwitch()
        self._calendar = SimpleMarketCalendar()
        self._watchdog = HeartbeatWatchdog(db_path=config.db_path)
        self._broker = AlpacaBroker(
            config.alpaca_api_key, config.alpaca_secret_key, paper=is_paper
        )
        self._feed = AlpacaDataFeed(config)
        self._portfolio = Portfolio(config)
        self._audit = AuditLog(config)
        self._risk = RiskEngine(config)
        self._selector = ConsensusSelector(strategies=config.strategies)
        self._planner = ATRPlanner()
        self._allocator = EqualWeightAllocator(
            max_position_pct=config.risk.max_position_pct,
        )
        if config.auto_trade_paper and config.broker_type != "alpaca_paper":
            raise ValueError("AUTO_TRADE_REQUIRES_ALPACA_PAPER")
        self._pos_monitor = StopTakeProfitMonitor()
        self._notifier = DiscordNotifier()
        self._price_news = PriceMoveSource(
            universe=config.symbols, timeframe=config.timeframe
        )
        self._wscn = WallStreetCNSource(
            universe=config.symbols, channels=["global", "us"],
        )
        self._sec = SECEdgarSource(universe=config.symbols)
        self._finnhub = FinnhubSource(universe=config.symbols)
        self._reviewer = SimpleReviewer(db_path=config.db_path)

        self._running = False
        self._tick_count = 0
        self._open_orders: Dict[str, OrderIntent] = {}
        self._order_store = OrderIntentStore(config.db_path)
        self._bug_reporter = BugReporter(config.db_path, "runtime")
        self._reconciliation_blocked = False
        self._decision_service = PaperDecisionService(allow_without_ai=config.allow_quant_without_ai, ai_max_age_minutes=int(config.ai_score_max_age_minutes))
        self._strategy_stats = StrategyStatisticsRepository.from_json(config.strategy_statistics_path)
        self._universe_provider = UniverseProvider(config.symbols, config.universe_max_symbols, config.universe_max_age_minutes)
        self._universe_snapshot = self._universe_provider.provide(cli_symbols=config.symbols, now=utc_now())
        self._live_plans: Dict[str, TradePlan] = {}  # symbol → 当前活跃计划
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
            time.sleep(self._cfg.poll_interval_secs)

        logger.info("Runtime stopped")

    def _run_reconciliation(self) -> None:
        try:
            report = reconcile_broker(self._broker, self._order_store.list_all(), self._portfolio.positions.values())
            self._audit.log_reconciliation(report)
            self._reconciliation_blocked = not report.ok
            if self._reconciliation_blocked:
                logger.error("startup reconciliation blocked trading")
            local_rows = self._order_store.list_all()
            for fill in self._broker.get_recent_fills():
                local = next((row for row in local_rows if row.get("broker_order_id") == fill.order_id), None)
                if local:
                    fill.intent_id = local["intent_id"]
                    self._portfolio.apply_fill(fill)
            for order in self._broker.get_open_orders():
                broker_id = str(order.get("id", "")) if isinstance(order, dict) else str(getattr(order, "id", ""))
                local = next((row for row in self._order_store.list_all() if row.get("broker_order_id") == broker_id), None)
                if local:
                    self._open_orders[broker_id] = OrderIntent(
                        intent_id=local["intent_id"], signal_id=local.get("decision_id", ""),
                        symbol=local["symbol"], side=Side(local["side"]), qty=local["qty"],
                        order_type=local["order_type"], limit_price=local["limit_price"],
                        tif=local["tif"], idempotency_key=local["idempotency_key"],
                        client_order_id=local.get("client_order_id", ""))
        except Exception as exc:
            self._reconciliation_blocked = True
            self._bug_reporter.capture_exception(exc, operation="runtime.reconciliation")
            logger.error("startup reconciliation failed: %s", type(exc).__name__)
    def stop(self) -> None:
        self._running = False

    # ── 主循环 ───────────────────────────────────────────────────────────────

    def _tick(self) -> None:
        self._tick_count += 1
        ts = utc_now()
        logger.info("── tick #%d  %s ──", self._tick_count, ts.strftime("%H:%M:%S UTC"))

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
            return
        logger.info("equity=%.2f  positions=%d", equity, len(positions))

        # 4. 日内起点（只设一次）
        if not self._daily_start_set and equity > 0:
            self._risk.set_daily_start(equity)
            self._daily_start_set = True

        # 5. 权益熔断
        self._risk.check_equity(equity)
        if self._risk.is_halted:
            logger.warning("风控熔断: %s", self._risk.halt_reason)
            self._audit.log_heartbeat(self._tick_count, equity)
            return

        # 6. 轮询已提交订单
        self._poll_orders()

        # 6b. 晨报（美东 9AM，每天一次，不受市场时段限制）
        self._maybe_morning_brief(ts)

        # 7. 市场时段判断
        session = self._calendar.session_now()
        if session == "closed":
            logger.info("市场已休市 — 仅做快照+心跳")
            self._portfolio.snapshot_external_equity(equity)
            self._audit.log_heartbeat(self._tick_count, equity)
            self._maybe_daily_review(ts)
            return
        if session != "open":
            logger.info("market session=%s — no new trade plans outside regular hours", session)
            self._portfolio.snapshot_external_equity(equity)
            self._audit.log_heartbeat(self._tick_count, equity)
            return

        # 8. 拉取 K 线，更新数据缓存
        raw_bars_map: Dict[str, list] = {}   # symbol → alpaca bar list
        model_bars: Dict[str, Bar] = {}       # symbol → 最新 trader.models.Bar（给 pos_monitor 用）

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

        if not raw_bars_map:
            logger.warning("无可用 K 线数据，跳过 selection")
            self._portfolio.snapshot_external_equity(equity)
            self._audit.log_heartbeat(self._tick_count, equity)
            return

        # 9. 新闻事件：四路合并（WSCN + SEC 8-K + Finnhub + 价格异动）
        from datetime import timedelta
        news = []
        _news_sources = [
            ("wscn",    self._wscn,       ts),
            ("sec",     self._sec,        ts - timedelta(hours=20)),  # 8-K 可能盘前发
            ("finnhub", self._finnhub,    ts - timedelta(hours=4)),
            ("price",   self._price_news, ts),
        ]
        for src_name, src, since_dt in _news_sources:
            try:
                batch = src.poll(since=since_dt)
                news.extend(batch)
                if batch:
                    logger.info("新闻 [%s]: %d 条", src_name, len(batch))
            except Exception as exc:
                logger.warning("news.poll [%s] 失败: %s", src_name, exc)

        # 10. 持仓监控：止损/止盈触发 → 生成 CLOSE 计划并立即执行
        if self._live_plans and model_bars:
            triggered = self._pos_monitor.check(positions, self._live_plans, model_bars)
            for close_plan in triggered:
                logger.info(
                    "[POS_MONITOR] %s %s @ %.2f",
                    close_plan.symbol, close_plan.rationale[:40], close_plan.entry_price,
                )
                self._execute_plan(close_plan, equity, positions)

        # 11. 选股（ConsensusSelector → T0 regime 动态 score 阈值过滤）
        score_threshold = _get_score_threshold()
        if score_threshold is None:
            logger.info("T0 高波动(HIGH_VOL) → 暂停选股本轮")
            self._portfolio.snapshot_external_equity(equity)
            self._audit.log_heartbeat(self._tick_count, equity)
            return
        try:
            candidates = self._selector.select(
                universe=list(self._universe_snapshot.symbols),
                timeframe=self._cfg.timeframe,
                as_of=ts,
            )
            candidates = [c for c in candidates if c.score >= score_threshold]
            logger.info(
                "selection: %d candidates (score≥%.0f, regime=%s)",
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
        if self._cfg.paper_decision_enabled:
            decisions = self._decision_service.decide(
                bars=raw_bars_map,
                positions=positions,
                candidates=candidates,
                strategy_statistics=self._strategy_stats,
                ai_advisories=self._read_ai_scores(),
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
                plan = self._planner.make_plan(
                    cand,
                    _alpaca_bar_to_model(raw[-1], cand.symbol, self._cfg.timeframe),
                    params=decision.params if decision else None,
                    current_qty=qty_held,
                    bars_history=bars_history,
                )
                if decision:
                    plan.source = "paper_decision"
                    plan.metadata.update({"decision_id": decision.decision_id, "strategy": decision.strategy, "strategy_statistics_id": decision.strategy_statistics_id, "universe_version": decision.universe_version, "data_version": decision.data_version})
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
            return

        # 13. 仓位分配（EqualWeightAllocator 填 qty / target_weight）
        try:
            plans = self._allocator.allocate(raw_plans, equity, positions)
        except Exception as exc:
            self._bug_reporter.capture_exception(
                exc, operation="allocator.allocate"
            )
            logger.error("allocator.allocate 失败: %s", exc, exc_info=True)
            plans = raw_plans

        # 14. AI 安全门 + 确定性风控
        # 14a. AI 自动交易模式：用 ai_states.duckdb 的综合评分更新 plan.confidence
        if self._cfg.auto_trade_paper:
            ai_scores = self._read_ai_scores()
            if ai_scores:
                for plan in plans:
                    snapshot = ai_scores.get(plan.symbol)
                    result = AIScoreValidator(AIScorePolicy(self._cfg.min_ai_score, self._cfg.ai_score_max_age_minutes)).validate(snapshot)
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
                validator = AIScoreValidator(AIScorePolicy(self._cfg.min_ai_score, self._cfg.ai_score_max_age_minutes))
                for plan in plans:
                    plan.status = "REJECTED"
                    result = validator.validate(None)
                    self._audit.log_ai_safety_event(plan, result, self._cfg)

        for plan in plans:
            if plan.status == "REJECTED":
                continue
            verdict = self._risk.evaluate_plan(plan, equity, positions)
            if not verdict.approved:
                logger.info("Plan [%s] 风控拒绝: %s", plan.symbol, verdict.reason)
                plan.status = "REJECTED"
                continue

            if self._cfg.paper_decision_enabled and self._cfg.paper_decision_shadow_mode:
                plan.status = "SHADOW"
                self._audit.log_trade_plan(plan)
                continue
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
            self._execute_plan(plan, equity, positions)

        # 16. 快照 + 心跳
        self._portfolio.snapshot_external_equity(equity)
        self._audit.log_heartbeat(self._tick_count, equity)

        # 17. 盘后复盘（每日一次，21:00 UTC 后触发）
        self._maybe_daily_review(ts)

    # ── 执行单个计划 ─────────────────────────────────────────────────────────

    def _execute_plan(
        self,
        plan: TradePlan,
        equity: float,
        positions: Dict[str, Position],
    ) -> None:
        """将通过 AI 安全门和确定性风控的计划转成 LMT 限价单。

        仅 AI 自动模拟盘模式可提交；kill switch 始终可紧急拦截。
        """
        if self._cfg.paper_decision_enabled and self._cfg.paper_decision_shadow_mode:
            logger.info("[SHADOW] plan=%s broker submission disabled", plan.plan_id[:8])
            return
        if not self._cfg.auto_trade_paper:
            logger.warning("[BLOCKED] plan=%s AI automatic paper trading is disabled", plan.plan_id[:8])
            return
        if self._reconciliation_blocked:
            logger.warning("[BLOCKED] startup reconciliation incomplete")
            return
        if self._kill.engaged():
            logger.warning(
                "[BLOCKED] plan=%s kill switch engaged", plan.plan_id[:8]
            )
            return

        key = idempotency_key(plan.plan_id, plan.symbol, plan.side.value, plan.qty, plan.entry_price, plan.action)
        existing = self._order_store.get_by_key(key)
        if existing and (existing.get("broker_order_id") or existing.get("state") in {OrderLifecycle.SENDING.value, OrderLifecycle.UNKNOWN.value, OrderLifecycle.OPEN.value, OrderLifecycle.PARTIALLY_FILLED.value}):
            logger.info("idempotent order already exists plan=%s", plan.plan_id[:8])
            return
        intent = OrderIntent(
            intent_id=existing.get("intent_id") if existing else new_id(),
            signal_id=plan.plan_id,
            symbol=plan.symbol,
            side=plan.side,
            qty=plan.qty,
            order_type="LMT",
            limit_price=plan.entry_price,
            reference_price=plan.entry_price,
            tif="DAY",
            risk_tag=f"runtime/{plan.action}",
            created_at=utc_now(),
            idempotency_key=key,
            client_order_id=client_order_id(key),
            decision_id=plan.metadata.get("decision_id", ""),
            plan_id=plan.plan_id,
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
                fill = self._broker.get_fill(broker_id)
                if fill is not None:
                    fill.intent_id = intent.intent_id
                    self._portfolio.apply_fill(fill)
                    if intent.idempotency_key:
                        self._order_store.update(intent.idempotency_key, filled_qty=fill.filled_qty, remaining_qty=max(0.0, intent.qty - fill.filled_qty), state=(OrderLifecycle.FILLED.value if status == OrderStatus.FILLED else OrderLifecycle.PARTIALLY_FILLED.value))
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
                if status == OrderStatus.FILLED:
                    done.append(broker_id)
            elif status in (OrderStatus.CANCELLED, OrderStatus.REJECTED):
                logger.info("Order closed %s status=%s", broker_id, status.value)
                # 订单取消/拒绝：也从 _live_plans 清理，允许重新进入选股
                self._live_plans.pop(intent.symbol, None)
                if intent.idempotency_key:
                    self._order_store.update(intent.idempotency_key, state=(OrderLifecycle.CANCELED.value if status == OrderStatus.CANCELLED else OrderLifecycle.REJECTED.value))
                done.append(broker_id)
        for broker_id in done:
            self._open_orders.pop(broker_id, None)

    # ── 每日复盘 ─────────────────────────────────────────────────────────────

    def _read_ai_scores(self) -> Dict[str, AIScoreSnapshot]:
        """从 ai_states.duckdb 读取最新 AI 加权综合分（0-100）。失败返回空字典。"""
        try:
            from .ai.manager import get_score_snapshots_from_db
            return get_score_snapshots_from_db(self._cfg.ai_score_db)
        except Exception as exc:
            logger.warning("_read_ai_scores 失败: %s", exc)
            return {}

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

    def _maybe_daily_review(self, ts: datetime) -> None:
        """每日 21:00 UTC（≈ 美东 16:00）后触发一次复盘，当天只运行一次。"""
        today = ts.strftime("%Y-%m-%d")
        if today == self._last_review_date:
            return
        if ts.hour < 21:
            return
        self._last_review_date = today
        try:
            report = self._reviewer.review(period="1d", as_of=ts)
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
                market_summary=getattr(report, "market_summary", ""),
                symbols=list(self._cfg.symbols) if getattr(self._cfg, "symbols", None) else None,
            ))
        except Exception as exc:
            logger.warning("每日复盘失败: %s", exc)
