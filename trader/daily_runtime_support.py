"""Runtime-facing coordinator for daily research and live status publication."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Iterable, Mapping

from .daily_research import (
    DailyResearchService,
    DailyResearchStore,
    DailyResearchWorker,
    TradingAgentsAdapter,
)
from .models import Bar, Position, TradePlan
from .runtime_status import build_runtime_status, write_runtime_status

logger = logging.getLogger(__name__)


class DailyRuntimeSupport:
    """Own the background daily batch without exposing broker access to agents."""

    def __init__(
        self,
        config: Any,
        universe: Iterable[str],
        *,
        on_error: Any = None,
    ) -> None:
        # on_error(error_code, summary) —— 由 runtime 注入，用来把研究批次的
        # 故障送到读者眼前。这里刻意不直接依赖 notifier：这个类的职责是"在后
        # 台跑研究批次且不把券商访问权暴露给 agent"，塞一个推送客户端进来会
        # 让它多担一份不属于自己的责任。
        self._on_error = on_error
        self.enabled = bool(getattr(config, "daily_research_enabled", True))
        self.max_age_hours = float(
            getattr(config, "daily_research_max_age_hours", 36.0)
        )
        db_path = str(
            getattr(config, "daily_research_db", "")
            or getattr(config, "ai_score_db", "ai_states.duckdb")
        )
        self.store = DailyResearchStore(db_path)
        service = DailyResearchService(
            self.store,
            TradingAgentsAdapter(),
            # notifier=None 是刻意的，不是漏接：研究结果由收盘报告统一播报
            # （runtime._maybe_flush_close_report 会主动读取本批次结果并合进
            # "明天做什么"一节）。这里再接一个 notifier 会让同一批研究结论在
            # 频道里出现两次。
            notifier=None,
        )
        self.worker = DailyResearchWorker(
            service,
            universe,
            timeframe=str(getattr(config, "timeframe", "5m")),
            screen_limit=int(getattr(config, "daily_research_screen_limit", 10)),
            deep_limit=int(getattr(config, "daily_research_deep_limit", 5)),
            strategy_statistics_path=str(
                getattr(config, 'strategy_statistics_path', '')
            ),
            close_hour_et=int(
                getattr(config, "daily_research_close_hour_et", 16)
            ),
            close_minute_et=int(
                getattr(config, "daily_research_close_minute_et", 15)
            ),
        )
        self.latest_run = None

    def tick(self, now: datetime) -> Any | None:
        if not self.enabled:
            return None
        try:
            completed = self.worker.poll()
            if completed is not None:
                self.latest_run = completed
                logger.info(
                    "Daily research completed run=%s status=%s completed=%d failed=%d",
                    completed.run_id,
                    completed.status,
                    completed.completed_symbols,
                    completed.failed_symbols,
                )
                # 批次跑完但一个标的都没成功 == 明天没有可用的研究结论，
                # 这值得读者知道，而不是等到第二天发现收盘报告里空空如也。
                if completed.completed_symbols == 0 and completed.failed_symbols:
                    self._notify_error(
                        "DAILY_RESEARCH_ALL_FAILED",
                        f"研究批次 {completed.run_id} 的 "
                        f"{completed.failed_symbols} 个标的全部失败，"
                        "明日没有可用的深度研究结论。",
                    )
            if self.worker.start_if_due(now):
                logger.info("Daily TradingAgents research batch started")
        except Exception as exc:
            logger.error("Daily research scheduler failed: %s", exc)
            self._notify_error(
                "DAILY_RESEARCH_SCHEDULER_FAILED",
                f"研究批次调度失败（{type(exc).__name__}）：{exc}",
            )
        return self.latest_run

    def _notify_error(self, code: str, summary: str) -> None:
        if self._on_error is None:
            return
        try:
            self._on_error(code, summary)
        except Exception as exc:  # noqa: BLE001 - 告警失败不能反过来打断调度
            logger.debug("研究故障告警失败: %s", exc)

    def snapshots(self, now: datetime):
        if not self.enabled:
            return {}
        return self.store.score_snapshots(now, max_age_hours=self.max_age_hours)

    def publish_status(
        self,
        *,
        now: datetime,
        tick_count: int,
        session: str,
        equity: float,
        reconciliation_blocked: bool,
        kill_switch: bool,
        bars: Mapping[str, Bar] | None = None,
        positions: Mapping[str, Position] | None = None,
        plans: Mapping[str, TradePlan] | None = None,
        open_orders: Iterable[Any] = (),
        message: str = "",
        auto_trade_paper: bool = False,
    ) -> None:
        snapshots = self.snapshots(now)
        run = self.store.latest_run()
        payload = build_runtime_status(
            now=now,
            tick_count=tick_count,
            session=session,
            equity=equity,
            reconciliation_blocked=reconciliation_blocked,
            kill_switch=kill_switch,
            bars=bars,
            positions=positions,
            plans=plans,
            research_snapshots=snapshots,
            research_run=run,
            open_orders=open_orders,
            message=message,
            auto_trade_paper=auto_trade_paper,
        )
        try:
            write_runtime_status(payload)
        except Exception as exc:
            logger.debug("Runtime status write failed: %s", exc)

    def close(self) -> None:
        self.worker.close()
