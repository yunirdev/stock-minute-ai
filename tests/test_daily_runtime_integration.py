from datetime import datetime, timezone
from types import SimpleNamespace

from trader.config import TradingConfig
from trader.daily_research import DailyResearchWorker
from trader.runtime import Runtime


def test_trading_config_enables_daily_research_by_default():
    config = TradingConfig()
    assert config.daily_research_enabled is True
    assert config.daily_research_max_age_hours == 36
    assert config.daily_research_screen_limit == 10
    assert config.daily_research_deep_limit == 5


def test_runtime_daily_ai_policy_uses_separate_ttl_and_one_graph():
    runtime = Runtime.__new__(Runtime)
    runtime._cfg = TradingConfig(
        daily_research_enabled=True,
        daily_research_max_age_hours=40,
    )
    policy = runtime._ai_score_policy(70)
    assert policy.max_age_minutes == 40 * 60
    assert policy.min_contributors == 1
    assert policy.min_weight_coverage == 1.0
    assert policy.require_llm is True


def test_daily_worker_does_not_repeat_failed_target_date():
    failed = SimpleNamespace(
        run_id="failed-run",
        status="FAILED",
        trading_date="2026-07-27",
    )

    class _Store:
        recovered = []

        def recover_stale_runs(self, *, now, stale_after_seconds):
            self.recovered.append((now, stale_after_seconds))
            return []

        def latest_run(self, trading_date):
            assert trading_date == "2026-07-27"
            return failed

    service = SimpleNamespace(store=_Store())
    worker = DailyResearchWorker(
        service,
        ["AAPL"],
        timeframe="5m",
        screen_limit=10,
        deep_limit=5,
    )
    try:
        premarket = datetime(2026, 7, 27, 12, 30, tzinfo=timezone.utc)
        assert worker.start_if_due(premarket) is False
        assert service.store.recovered == [(premarket, 7200)]
    finally:
        worker.close()
