from datetime import datetime, timedelta, timezone

from trader.focus_pool import FocusPoolInput, FocusPoolPolicy, FocusPoolStore
from trader.research_budget import (
    ResearchBudgetPolicy,
    ResearchBudgetStore,
    ResearchEstimate,
)
from trader.universe_registry import UniverseAsset, UniverseRegistryStore

NOW = datetime(2026, 7, 27, 20, 0, tzinfo=timezone.utc)


def _pool(db_path):
    universe = UniverseRegistryStore(db_path).create_version(
        universe_name="US listed",
        source_version="source-v1",
        assets=[
            UniverseAsset(
                symbol=symbol,
                asset_type="STOCK",
                exchange="NASDAQ",
                status="ACTIVE",
                tradable=True,
                source="alpaca-assets",
                as_of=NOW,
            )
            for symbol in ("AAPL", "MSFT", "NVDA", "AMD")
        ],
        as_of=NOW,
        created_at=NOW,
    )
    return FocusPoolStore(db_path).attempt_build(
        pool_name="daily-focus",
        universe_version=universe["version_id"],
        inputs=[
            FocusPoolInput(
                symbol=symbol,
                holdout_reliable=True,
                holdout_score=score,
                average_dollar_volume=50_000_000,
                data_quality=1,
            )
            for symbol, score in (
                ("AAPL", 0.95),
                ("MSFT", 0.9),
                ("NVDA", 0.85),
                ("AMD", 0.8),
            )
        ],
        policy=FocusPoolPolicy(max_size=4),
        as_of=NOW,
        created_at=NOW,
    )


def _estimates():
    return [
        ResearchEstimate(symbol, estimated_cost=1, estimated_seconds=10)
        for symbol in ("AAPL", "MSFT", "NVDA", "AMD")
    ]


def _policy(**overrides):
    values = {
        "max_symbols": 3,
        "max_estimated_cost": 3,
        "max_runtime_seconds": 100,
        "batch_size": 1,
        "max_retries": 1,
        "attempt_timeout_seconds": 30,
    }
    values.update(overrides)
    return ResearchBudgetPolicy(**values)


def test_plan_enforces_priority_symbol_and_cost_quota(tmp_path):
    db_path = tmp_path / "research.duckdb"
    pool = _pool(db_path)
    store = ResearchBudgetStore(db_path)
    run = store.plan(
        pool_id=pool["pool_id"],
        trading_date="2026-07-27",
        estimates=_estimates(),
        policy=_policy(),
        started_at=NOW,
    )

    assert run["planned_count"] == 3
    assert run["deferred_count"] == 1
    assert [item["symbol"] for item in run["items"]] == [
        "AAPL",
        "MSFT",
        "NVDA",
        "AMD",
    ]
    assert run["items"][-1]["status"] == "DEFERRED"
    assert run["items"][-1]["deferral_reason"] == "SYMBOL_QUOTA"
    assert store.claim_batch(run["budget_run_id"], now=NOW)[0]["symbol"] == "AAPL"


def test_failure_retries_timeout_and_terminal_status_are_durable(tmp_path):
    db_path = tmp_path / "research.duckdb"
    pool = _pool(db_path)
    store = ResearchBudgetStore(db_path)
    run = store.plan(
        pool_id=pool["pool_id"],
        trading_date="2026-07-27",
        estimates=_estimates(),
        policy=_policy(max_symbols=2, max_estimated_cost=2),
        started_at=NOW,
    )
    first = store.claim_batch(run["budget_run_id"], now=NOW)[0]
    retried = store.finish(
        first["work_id"],
        success=False,
        error_code="API_TIMEOUT",
        actual_cost=0.2,
        actual_seconds=5,
        now=NOW + timedelta(seconds=5),
    )
    assert retried["status"] == "RETRY"
    assert retried["attempts"] == 1

    second_attempt = store.claim_batch(
        run["budget_run_id"],
        now=NOW + timedelta(seconds=6),
    )[0]
    assert second_attempt["symbol"] == "AAPL"
    store.finish(
        second_attempt["work_id"],
        success=True,
        actual_cost=0.3,
        actual_seconds=4,
        now=NOW + timedelta(seconds=10),
    )
    msft = store.claim_batch(
        run["budget_run_id"],
        now=NOW + timedelta(seconds=11),
    )[0]
    assert msft["symbol"] == "MSFT"

    recovered = ResearchBudgetStore(db_path)
    timed_retry = recovered.claim_batch(
        run["budget_run_id"],
        now=NOW + timedelta(seconds=42),
    )[0]
    assert timed_retry["symbol"] == "MSFT"
    assert timed_retry["attempts"] == 2
    failed = recovered.finish(
        timed_retry["work_id"],
        success=False,
        error_code="MODEL_FAILED",
        actual_cost=0.4,
        actual_seconds=3,
        now=NOW + timedelta(seconds=45),
    )

    assert failed["status"] == "FAILED"
    assert failed["actual_seconds"] == 33
    assert recovered.get_run(run["budget_run_id"])["status"] == (
        "COMPLETED_WITH_ERRORS"
    )


def test_restart_resumes_pending_without_reclaiming_completed(tmp_path):
    db_path = tmp_path / "research.duckdb"
    pool = _pool(db_path)
    store = ResearchBudgetStore(db_path)
    run = store.plan(
        pool_id=pool["pool_id"],
        trading_date="2026-07-27",
        estimates=_estimates(),
        policy=_policy(max_symbols=2, max_estimated_cost=2),
        started_at=NOW,
    )
    first = store.claim_batch(run["budget_run_id"], now=NOW)[0]
    store.finish(
        first["work_id"],
        success=True,
        actual_cost=0.5,
        actual_seconds=5,
        now=NOW + timedelta(seconds=5),
    )

    recovered = ResearchBudgetStore(db_path)
    next_batch = recovered.claim_batch(
        run["budget_run_id"],
        now=NOW + timedelta(seconds=6),
    )

    assert [item["symbol"] for item in next_batch] == ["MSFT"]
    assert recovered.get_item(first["work_id"])["status"] == "COMPLETED"


def test_actual_cost_or_runtime_exhaustion_defers_remaining_work(tmp_path):
    db_path = tmp_path / "research.duckdb"
    pool = _pool(db_path)
    store = ResearchBudgetStore(db_path)
    run = store.plan(
        pool_id=pool["pool_id"],
        trading_date="2026-07-27",
        estimates=[
            ResearchEstimate(symbol, 0.1, 1)
            for symbol in ("AAPL", "MSFT", "NVDA", "AMD")
        ],
        policy=_policy(
            max_symbols=2,
            max_estimated_cost=1,
            max_runtime_seconds=20,
        ),
        started_at=NOW,
    )
    first = store.claim_batch(run["budget_run_id"], now=NOW)[0]
    store.finish(
        first["work_id"],
        success=True,
        actual_cost=1.0,
        actual_seconds=2,
        now=NOW + timedelta(seconds=2),
    )

    assert store.claim_batch(
        run["budget_run_id"],
        now=NOW + timedelta(seconds=3),
    ) == []
    final = store.get_run(run["budget_run_id"])
    msft = next(item for item in final["items"] if item["symbol"] == "MSFT")
    assert msft["status"] == "DEFERRED"
    assert msft["deferral_reason"] == "ACTUAL_COST_QUOTA"
    assert final["status"] == "COMPLETED_WITH_DEFERRED"
