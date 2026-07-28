from datetime import datetime, timedelta, timezone

import duckdb

from trader.daily_research_quality import (
    generate_daily_research_quality_report,
    save_daily_research_quality_report,
)

NOW = datetime(2026, 7, 26, 20, tzinfo=timezone.utc)


def _quality_db(path, *, missing_contract=False):
    conn = duckdb.connect(str(path))
    try:
        conn.execute(
            "CREATE TABLE daily_research_runs "
            "(run_id TEXT, trading_date TEXT, status TEXT, "
            "started_at TIMESTAMPTZ, completed_at TIMESTAMPTZ, "
            "error_code TEXT)"
        )
        conn.execute(
            "CREATE TABLE daily_research_items "
            "(run_id TEXT, status TEXT, error_code TEXT, snapshot_id TEXT, "
            "data_version TEXT, model_version TEXT, invocation_id TEXT, "
            "ta_snapshot_id TEXT)"
        )
        conn.execute(
            "CREATE TABLE daily_research_ta_snapshot_links "
            "(run_id TEXT, status TEXT)"
        )
        current = datetime(2026, 6, 29, 12, tzinfo=timezone.utc)
        for index in range(20):
            while current.weekday() >= 5:
                current += timedelta(days=1)
            run_id = f"run-{index}"
            conn.execute(
                "INSERT INTO daily_research_runs VALUES (?,?,?,?,?,?)",
                [
                    run_id,
                    current.date().isoformat(),
                    "COMPLETED",
                    current,
                    current + timedelta(minutes=5),
                    "",
                ],
            )
            values = ["s", "d", "m", "i", "ta"]
            if missing_contract and index == 19:
                values[-1] = ""
            conn.execute(
                "INSERT INTO daily_research_items VALUES (?,?,?,?,?,?,?,?)",
                [run_id, "COMPLETED", "", *values],
            )
            conn.execute(
                "INSERT INTO daily_research_ta_snapshot_links VALUES (?,?)",
                [run_id, "WRITTEN"],
            )
            current += timedelta(days=1)
        conn.commit()
    finally:
        conn.close()


def test_twenty_day_batch_quality_report_and_store(tmp_path):
    path = tmp_path / "quality.duckdb"
    _quality_db(path)

    report = generate_daily_research_quality_report(
        path,
        generated_at=NOW,
    )

    assert report["passed"]
    assert report["observed_days"] == 20
    assert report["run_success_rate"] == 1.0
    assert report["item_success_rate"] == 1.0
    assert report["contract_coverage"] == 1.0
    assert report["source_snapshot_coverage"] == 1.0
    assert report["p95_duration_seconds"] == 300.0
    assert report["execution_input_switched"] is False
    assert save_daily_research_quality_report(path, report)
    assert not save_daily_research_quality_report(path, report)


def test_batch_quality_rejects_missing_contract(tmp_path):
    path = tmp_path / "quality.duckdb"
    _quality_db(path, missing_contract=True)

    report = generate_daily_research_quality_report(path)

    assert not report["passed"]
    assert report["contract_coverage"] == 0.95
    assert not report["gates"]["contract_coverage"]
