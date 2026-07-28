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

    def __init__(self, config: Any, universe: Iterable[str]) -> None:
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
            if self.worker.start_if_due(now):
                logger.info("Daily TradingAgents research batch started")
        except Exception as exc:
            logger.error("Daily research scheduler failed: %s", exc)
        return self.latest_run

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
        )
        try:
            write_runtime_status(payload)
        except Exception as exc:
            logger.debug("Runtime status write failed: %s", exc)

    def close(self) -> None:
        self.worker.close()
