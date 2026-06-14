"""
trader/runtime.py
M1 计划驱动管道 Runtime。

与 scheduler.py 的区别
  scheduler.py  = 信号驱动（TA Signal → risk → order）用于实时策略回路。
  runtime.py    = 计划驱动（Candidate → TradePlan → allocate → approve → LMT）
                  用于 AI 辅助决策流程。

每轮（tick）流程
  kill_switch → watchdog → equity/positions → risk.check_equity
  → poll_orders → market_session → fetch_bars → news
  → pos_monitor → selection → plan → allocate → evaluate_plan
  → approval → execute → portfolio.snapshot → heartbeat → daily_review

安全红线（必须，不得绕过）
  - AI 不下单：agent/LLM 只产出 Advisory/TradePlan(DRAFT)，绝不直接调 broker。
  - 执行开关：仅 config.execution_enabled=True AND !kill_switch.engaged() 时下单。
  - 只挂 LMT：绝不下 market order（AlpacaBroker 也有防护，双重保险）。
  - 人在回路：AutoApprover.auto_approve=False → 默认 PENDING，不会自动执行。
  - 密钥不入库：日志中不打印 API Key / Secret。
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd

from .approval import AutoApprover
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
from .news import PriceMoveSource
from .notify import DiscordNotifier
from .plan import ATRPlanner
from .portfolio import Portfolio
from .position_monitor import StopTakeProfitMonitor
from .review import SimpleReviewer
from .risk_engine import RiskEngine
from .selection import ConsensusSelector
from .watchdog import FileKillSwitch, HeartbeatWatchdog

logger = logging.getLogger(__name__)

_MIN_CANDIDATE_SCORE = 55.0   # selection 阈值，低于此分的 Candidate 不进入计划


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
        self._approver = AutoApprover(auto_approve=False)
        self._pos_monitor = StopTakeProfitMonitor()
        self._notifier = DiscordNotifier()
        self._news_source = PriceMoveSource(
            universe=config.symbols, timeframe=config.timeframe
        )
        self._reviewer = SimpleReviewer(db_path=config.db_path)

        self._running = False
        self._tick_count = 0
        self._open_orders: Dict[str, OrderIntent] = {}
        self._live_plans: Dict[str, TradePlan] = {}  # symbol → 当前活跃计划
        self._daily_start_set = False
        self._last_review_date: Optional[str] = None

        logger.info(
            "Runtime init symbols=%s tf=%s execution_enabled=%s paper=%s",
            config.symbols, config.timeframe, config.execution_enabled, is_paper,
        )

    # ── 公共接口 ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        """启动计划驱动循环，阻塞直到 stop() 或 Ctrl-C。"""
        logger.info("Runtime start interval=%ds", self._cfg.poll_interval_secs)
        self._running = True

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
                logger.error("Runtime tick 异常: %s", exc, exc_info=True)
            time.sleep(self._cfg.poll_interval_secs)

        logger.info("Runtime stopped")

    def stop(self) -> None:
        self._running = False

    # ── 主循环 ───────────────────────────────────────────────────────────────

    def _tick(self) -> None:
        self._tick_count += 1
        ts = utc_now()
        logger.info("── tick #%d  %s ──", self._tick_count, ts.strftime("%H:%M:%S UTC"))

        # 1. Kill switch 急停检查
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

        # 7. 市场时段判断
        session = self._calendar.session_now()
        if session == "closed":
            logger.info("市场已休市 — 仅做快照+心跳")
            self._portfolio.snapshot_external_equity(equity)
            self._audit.log_heartbeat(self._tick_count, equity)
            self._maybe_daily_review(ts)
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
                raw_bars_map[symbol] = raw
                rows = [
                    {"timestamp_utc": b.timestamp,
                     "open": b.open, "high": b.high, "low": b.low,
                     "close": b.close, "volume": b.volume}
                    for b in raw
                ]
                _dc_upsert(symbol, self._cfg.timeframe, pd.DataFrame(rows))
                model_bars[symbol] = _alpaca_bar_to_model(raw[-1], symbol, self._cfg.timeframe)
            except Exception as exc:
                logger.warning("fetch_bars %s 失败: %s", symbol, exc)

        if not raw_bars_map:
            logger.warning("无可用 K 线数据，跳过 selection")
            self._portfolio.snapshot_external_equity(equity)
            self._audit.log_heartbeat(self._tick_count, equity)
            return

        # 9. 新闻事件（仅记录，不阻断流程）
        try:
            news = self._news_source.poll(since=ts)
            if news:
                logger.info("新闻事件: %d 条", len(news))
        except Exception as exc:
            logger.warning("news.poll 失败: %s", exc)
            news = []

        # 10. 持仓监控：止损/止盈触发 → 生成 CLOSE 计划并立即执行
        if self._live_plans and model_bars:
            triggered = self._pos_monitor.check(positions, self._live_plans, model_bars)
            for close_plan in triggered:
                logger.info(
                    "[POS_MONITOR] %s %s @ %.2f",
                    close_plan.symbol, close_plan.rationale[:40], close_plan.entry_price,
                )
                self._execute_plan(close_plan, equity, positions)

        # 11. 选股（ConsensusSelector → score 阈值过滤）
        try:
            candidates = self._selector.select(
                universe=self._cfg.symbols,
                timeframe=self._cfg.timeframe,
                as_of=ts,
            )
            candidates = [c for c in candidates if c.score >= _MIN_CANDIDATE_SCORE]
            logger.info("selection: %d candidates (score≥%.0f)", len(candidates), _MIN_CANDIDATE_SCORE)
        except Exception as exc:
            logger.error("selection.select 失败: %s", exc, exc_info=True)
            candidates = []

        # 12. 计划生成（ATRPlanner）
        raw_plans: List[TradePlan] = []
        for cand in candidates:
            raw = raw_bars_map.get(cand.symbol)
            if not raw:
                continue
            try:
                qty_held = positions[cand.symbol].qty if cand.symbol in positions else 0.0
                plan = self._planner.make_plan(
                    cand,
                    _alpaca_bar_to_model(raw[-1], cand.symbol, self._cfg.timeframe),
                    current_qty=qty_held,
                )
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
            logger.error("allocator.allocate 失败: %s", exc, exc_info=True)
            plans = raw_plans

        # 14. 风控 + 审批
        for plan in plans:
            verdict = self._risk.evaluate_plan(plan, equity, positions)
            if not verdict.approved:
                logger.info("Plan [%s] 风控拒绝: %s", plan.symbol, verdict.reason)
                plan.status = "REJECTED"
                continue

            decision = self._approver.decide(plan)
            plan.status = decision
            logger.info("Plan [%s] 审批=%s", plan.symbol, decision)

            if decision == "APPROVED":
                self._notifier.send(Notification(
                    title=f"计划已批准: {plan.symbol}",
                    body=(
                        f"{plan.action} {plan.side.value} qty={plan.qty:.0f} "
                        f"entry={plan.entry_price:.2f} stop={plan.stop_loss:.2f} "
                        f"tp={plan.take_profit:.2f}"
                    ),
                    kind="plan",
                    plan_id=plan.plan_id,
                ))

        # 15. 执行 APPROVED 计划（受 execution_enabled + kill_switch 双重保护）
        for plan in plans:
            if plan.status == "APPROVED":
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
        """将 APPROVED 计划转成 LMT 限价单提交给 broker。

        红线双重保护：
          1. execution_enabled=False → DRY-RUN，只记日志不下单。
          2. kill_switch.engaged()   → 紧急拦截。
        """
        if not self._cfg.execution_enabled:
            logger.info(
                "[DRY-RUN] plan=%s %s %s qty=%.0f @ %.2f (execution_enabled=False)",
                plan.plan_id[:8], plan.symbol, plan.side.value, plan.qty, plan.entry_price,
            )
            return
        if self._kill.engaged():
            logger.warning(
                "[BLOCKED] plan=%s kill switch engaged", plan.plan_id[:8]
            )
            return

        intent = OrderIntent(
            intent_id=new_id(),
            signal_id=plan.plan_id,
            symbol=plan.symbol,
            side=plan.side,
            qty=plan.qty,
            order_type="LMT",            # 红线：只挂限价单
            limit_price=plan.entry_price,
            reference_price=plan.entry_price,
            tif="DAY",
            risk_tag=f"runtime/{plan.action}",
            created_at=utc_now(),
        )

        try:
            broker_id = self._broker.place_order(intent)
            self._open_orders[broker_id] = intent
            self._live_plans[plan.symbol] = plan
            self._risk.record_success()
            logger.info(
                "ORDER submitted %s %s %s qty=%.0f @ %.2f  broker_id=%s",
                plan.symbol, plan.side.value, plan.action,
                intent.qty, intent.limit_price, broker_id,
            )
        except Exception as exc:
            logger.error("place_order [%s] 失败: %s", plan.symbol, exc)
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
            if status == OrderStatus.FILLED:
                fill = self._broker.get_fill(broker_id)
                if fill is not None:
                    fill.intent_id = intent.intent_id
                    self._portfolio.apply_fill(fill)
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
                done.append(broker_id)
            elif status in (OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.FAILED):
                logger.info("Order closed %s status=%s", broker_id, status.value)
                done.append(broker_id)
        for broker_id in done:
            self._open_orders.pop(broker_id, None)

    # ── 每日复盘 ─────────────────────────────────────────────────────────────

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
            self._notifier.send(Notification(
                title=f"每日复盘 {today}",
                body=(
                    f"交易数={trade_count}  "
                    f"盈亏={report.portfolio_pnl:+,.2f}  "
                    f"{report.market_summary}"
                ),
                kind="review",
                fields={
                    "pnl": report.portfolio_pnl,
                    "trade_count": trade_count,
                },
            ))
        except Exception as exc:
            logger.warning("每日复盘失败: %s", exc)
