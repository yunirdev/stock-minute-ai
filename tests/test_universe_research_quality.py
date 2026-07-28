from datetime import datetime, timedelta, timezone

import pandas as pd

from trader.focus_pool import FocusPoolInput, FocusPoolPolicy, FocusPoolStore
from trader.research_budget import (
    ResearchBudgetPolicy,
    ResearchBudgetStore,
    ResearchEstimate,
)
from trader.universe_registry import UniverseAsset, UniverseRegistryStore
from trader.universe_research_quality import (
    UniverseResearchGate,
    UniverseResearchQualityStore,
)

BASE = datetime(2026, 1, 2, 20, 0, tzinfo=timezone.utc)


def _foundation(db_path):
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
                as_of=BASE,
            )
            for symbol in ("AAPL", "MSFT")
        ],
        as_of=BASE,
        created_at=BASE,
    )
    pool = FocusPoolStore(db_path).attempt_build(
        pool_name="daily-focus",
        universe_version=universe["version_id"],
        inputs=[
            FocusPoolInput(symbol, True, 0.9, 50_000_000, 1)
            for symbol in ("AAPL", "MSFT")
        ],
        policy=FocusPoolPolicy(max_size=2),
        as_of=BASE,
        created_at=BASE,
    )
    return universe, pool


def _capture_days(db_path, count, *, evidence_type, fail_last=False):
    universe, pool = _foundation(db_path)
    budgets = ResearchBudgetStore(db_path)
    quality = UniverseResearchQualityStore(db_path)
    sessions = pd.bdate_range("2026-01-05", periods=count, tz="UTC")
    observations = []
    for index, stamp in enumerate(sessions):
        started = stamp.to_pydatetime().replace(hour=14)
        trading_date = stamp.date().isoformat()
        run = budgets.plan(
            pool_id=pool["pool_id"],
            trading_date=trading_date,
            estimates=[
                ResearchEstimate(symbol, 1, 10)
                for symbol in ("AAPL", "MSFT")
            ],
            policy=ResearchBudgetPolicy(
                max_symbols=2,
                max_estimated_cost=10,
                max_runtime_seconds=100,
                batch_size=2,
                max_retries=0,
                attempt_timeout_seconds=30,
            ),
            started_at=started,
        )
        claimed = budgets.claim_batch(run["budget_run_id"], now=started)
        for item_index, item in enumerate(claimed):
            failed = fail_last and index == count - 1 and item_index == 0
            budgets.finish(
                item["work_id"],
                success=not failed,
                error_code="MODEL_FAILED" if failed else "",
                actual_cost=1,
                actual_seconds=10,
                now=started + timedelta(seconds=10 + item_index),
            )
        observations.append(
            quality.capture(
                evidence_type=evidence_type,
                trading_date=trading_date,
                universe_version=universe["version_id"],
                pool_id=pool["pool_id"],
                budget_run_id=run["budget_run_id"],
                created_at=started + timedelta(seconds=20),
            )
        )
    return quality, observations


def test_twenty_synthetic_sessions_pass_coverage_cost_and_window_gate(tmp_path):
    quality, observations = _capture_days(
        tmp_path / "research.duckdb",
        20,
        evidence_type="SYNTHETIC",
    )
    report = quality.report(
        evidence_type="SYNTHETIC",
        through_date=observations[-1]["trading_date"],
    )

    assert report["passed"]
    assert report["observed_sessions"] == 20
    assert all(day["screening_coverage"] == 1 for day in report["daily"])
    assert all(day["research_completion"] == 1 for day in report["daily"])
    assert all(day["cost_utilization"] == 0.2 for day in report["daily"])
    assert all(day["duration_utilization"] <= 0.11 for day in report["daily"])


def test_failure_day_is_reported_and_real_evidence_stays_separate(tmp_path):
    db_path = tmp_path / "research.duckdb"
    quality, observations = _capture_days(
        db_path,
        20,
        evidence_type="SYNTHETIC",
        fail_last=True,
    )
    report = quality.report(
        evidence_type="SYNTHETIC",
        through_date=observations[-1]["trading_date"],
        gate=UniverseResearchGate(max_research_failure_rate=0),
    )
    real = quality.report(
        evidence_type="REAL",
        through_date=observations[-1]["trading_date"],
    )

    assert not report["passed"]
    assert {
        failure["reason"] for failure in report["failures"]
    } >= {"RESEARCH_COMPLETION", "RESEARCH_FAILURE_RATE", "BUDGET_STATUS"}
    assert real["observed_sessions"] == 0
    assert not real["passed"]
    assert real["failures"] == [
        {"trading_date": "", "reason": "INSUFFICIENT_TRADING_SESSIONS"}
    ]


def test_observation_is_idempotent_and_date_cannot_be_rewritten(tmp_path):
    db_path = tmp_path / "research.duckdb"
    quality, observations = _capture_days(
        db_path,
        1,
        evidence_type="REAL",
    )
    first = observations[0]
    duplicate = quality.capture(
        evidence_type="REAL",
        trading_date=first["trading_date"],
        universe_version=first["universe_version"],
        pool_id=first["pool_id"],
        budget_run_id=first["budget_run_id"],
        created_at=first["created_at"] + timedelta(minutes=1),
    )

    assert duplicate == first
    assert quality.report(
        evidence_type="SYNTHETIC",
        through_date=first["trading_date"],
    )["observed_sessions"] == 0
