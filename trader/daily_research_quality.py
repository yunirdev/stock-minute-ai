"""Quantitative TradingAgents batch quality report."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import duckdb


def _p95(values):
    rows = sorted(float(value) for value in values)
    return rows[max(0, math.ceil(len(rows) * 0.95) - 1)] if rows else 0.0


def generate_daily_research_quality_report(
    db_path: str | Path,
    *,
    required_days: int = 20,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    if required_days < 1:
        raise ValueError("DAILY_RESEARCH_QUALITY_DAYS_INVALID")
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        runs = conn.execute(
            """
            SELECT run_id, trading_date, status, started_at, completed_at,
                   error_code
            FROM daily_research_runs
            ORDER BY trading_date, started_at
            """
        ).fetchall()
        items = conn.execute(
            """
            SELECT run_id, status, error_code, snapshot_id, data_version,
                   model_version, invocation_id, ta_snapshot_id
            FROM daily_research_items
            """
        ).fetchall()
        links = conn.execute(
            """
            SELECT run_id, status FROM daily_research_ta_snapshot_links
            """
        ).fetchall()
    finally:
        conn.close()
    dates = sorted({str(row[1]) for row in runs})[-required_days:]
    selected_runs = [row for row in runs if str(row[1]) in dates]
    run_ids = {str(row[0]) for row in selected_runs}
    selected_items = [row for row in items if str(row[0]) in run_ids]
    selected_links = [row for row in links if str(row[0]) in run_ids]
    successful_runs = [
        row for row in selected_runs
        if row[2] in {"COMPLETED", "COMPLETED_WITH_ERRORS"}
    ]
    completed_items = [row for row in selected_items if row[1] == "COMPLETED"]
    contract_items = [
        row for row in completed_items if all(str(value or "") for value in row[3:8])
    ]
    durations = [
        (row[4] - row[3]).total_seconds()
        for row in selected_runs if row[3] is not None and row[4] is not None
    ]
    by_date = {
        date: sum(str(row[1]) == date for row in selected_runs)
        for date in dates
    }
    run_success_rate = (
        len(successful_runs) / len(selected_runs) if selected_runs else 0.0
    )
    item_success_rate = (
        len(completed_items) / len(selected_items) if selected_items else 0.0
    )
    contract_coverage = (
        len(contract_items) / len(completed_items) if completed_items else 0.0
    )
    source_snapshot_coverage = (
        sum(row[1] in {"WRITTEN", "EXISTS"} for row in selected_links)
        / len(completed_items)
        if completed_items else 0.0
    )
    gates = {
        "observation_window": len(dates) >= required_days,
        "one_run_per_day": bool(dates) and all(value == 1 for value in by_date.values()),
        "run_success_rate": run_success_rate >= 0.95,
        "item_success_rate": item_success_rate >= 0.90,
        "contract_coverage": contract_coverage == 1.0,
        "source_snapshot_coverage": source_snapshot_coverage == 1.0,
        "batch_latency": _p95(durations) <= 7_200.0,
    }
    report = {
        "generated_at": (generated_at or datetime.now(timezone.utc)).isoformat(),
        "required_days": required_days,
        "observed_days": len(dates),
        "window_start": dates[0] if dates else None,
        "window_end": dates[-1] if dates else None,
        "runs": len(selected_runs),
        "successful_runs": len(successful_runs),
        "run_success_rate": run_success_rate,
        "items": len(selected_items),
        "completed_items": len(completed_items),
        "item_success_rate": item_success_rate,
        "contract_coverage": contract_coverage,
        "source_snapshot_coverage": source_snapshot_coverage,
        "p95_duration_seconds": _p95(durations),
        "run_failures": sorted(
            str(row[5]) for row in selected_runs if str(row[5] or "")
        ),
        "item_failures": sorted(
            str(row[2]) for row in selected_items if str(row[2] or "")
        ),
        "gates": gates,
        "passed": all(gates.values()),
        "execution_input_switched": False,
    }
    encoded = json.dumps(report, sort_keys=True, separators=(",", ":"))
    report["report_id"] = (
        "daily-research-quality-"
        + hashlib.sha256(encoded.encode()).hexdigest()[:20]
    )
    return report


def save_daily_research_quality_report(
    db_path: str | Path,
    report: Mapping[str, Any],
) -> bool:
    conn = duckdb.connect(str(db_path))
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_research_quality_reports (
                report_id TEXT PRIMARY KEY,
                generated_at TIMESTAMPTZ,
                passed BOOLEAN,
                payload TEXT
            )
            """
        )
        if conn.execute(
            "SELECT 1 FROM daily_research_quality_reports WHERE report_id=?",
            [report["report_id"]],
        ).fetchone():
            return False
        conn.execute(
            "INSERT INTO daily_research_quality_reports VALUES (?,?,?,?)",
            [
                report["report_id"],
                datetime.fromisoformat(str(report["generated_at"])),
                bool(report["passed"]),
                json.dumps(dict(report), sort_keys=True, separators=(",", ":")),
            ],
        )
        conn.commit()
        return True
    finally:
        conn.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="ai_states.duckdb")
    parser.add_argument("--days", type=int, default=20)
    args = parser.parse_args(argv)
    report = generate_daily_research_quality_report(
        args.db, required_days=args.days
    )
    save_daily_research_quality_report(args.db, report)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
