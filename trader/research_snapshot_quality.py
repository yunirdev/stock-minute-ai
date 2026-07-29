"""Auditable quality reporting for frozen shadow ResearchSnapshots."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

from .models import ResearchQuality, ResearchSourceStatus
from .research_snapshot import ResearchSnapshotStore

_CANDIDATE_FIELDS = {
    "symbol",
    "rank",
    "score",
    "status",
    "source_quality_score",
    "ai_score",
    "tactical_score",
    "data_confidence",
    "reasons",
    "risk_flags",
    "as_of",
}
_BAR_FIELDS = {
    "timestamp_utc",
    "open",
    "high",
    "low",
    "close",
    "volume",
}
_MANIFEST_SOURCES = {
    "local_bar_cache",
    "strategy_statistics",
    "deterministic_screening",
}
_CRITICAL_SLOT_COUNT = (
    len(_CANDIDATE_FIELDS)
    + len(_BAR_FIELDS)
    + len(_MANIFEST_SOURCES)
)


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _tables(conn: duckdb.DuckDBPyConnection) -> set[str]:
    return {
        row[0]
        for row in conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema='main'"
        ).fetchall()
    }


def generate_snapshot_quality_report(
    db_path: str | Path,
    *,
    required_days: int = 10,
    min_coverage: float = 0.95,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    if required_days < 1:
        raise ValueError("QUALITY_REPORT_REQUIRED_DAYS_INVALID")
    if not 0.0 <= min_coverage <= 1.0:
        raise ValueError("QUALITY_REPORT_COVERAGE_INVALID")
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(path)
    snapshot_store = ResearchSnapshotStore(path)
    conn = duckdb.connect(str(path), read_only=True)
    try:
        required_tables = {
            "daily_research_runs",
            "daily_research_snapshot_links",
            "daily_research_snapshot_comparisons",
            "research_snapshots",
        }
        missing = sorted(required_tables - _tables(conn))
        if missing:
            raise ValueError(
                "QUALITY_REPORT_SCHEMA_MISSING:" + ",".join(missing)
            )
        runs = conn.execute(
            """
            SELECT run_id, trading_date, status, started_at
            FROM daily_research_runs
            ORDER BY trading_date, started_at
            """
        ).fetchall()
        links = conn.execute(
            """
            SELECT run_id, symbol, snapshot_id, status, error_code
            FROM daily_research_snapshot_links
            """
        ).fetchall()
        comparisons = conn.execute(
            """
            SELECT run_id, symbol, snapshot_id, status,
                   differences_json, classification
            FROM daily_research_snapshot_comparisons
            """
        ).fetchall()
    finally:
        conn.close()

    latest_run_by_date: dict[str, tuple] = {}
    for row in runs:
        latest_run_by_date[str(row[1])] = row
    selected_dates = sorted(latest_run_by_date)[-required_days:]
    selected_runs = {
        str(latest_run_by_date[trading_date][0]): trading_date
        for trading_date in selected_dates
    }
    links_by_run: dict[str, list[tuple]] = {
        run_id: [] for run_id in selected_runs
    }
    for row in links:
        if str(row[0]) in selected_runs:
            links_by_run[str(row[0])].append(row)
    comparisons_by_key = {
        (str(row[0]), str(row[1])): row
        for row in comparisons
        if str(row[0]) in selected_runs
    }

    expected_snapshots = 0
    available_snapshots = 0
    acceptable_snapshots = 0
    comparison_count = 0
    critical_covered = 0
    write_failures = 0
    replay_errors = 0
    unclassified_differences = 0
    total_differences = 0
    daily: list[dict[str, Any]] = []

    for run_id, trading_date in sorted(
        selected_runs.items(),
        key=lambda item: item[1],
    ):
        run_links = links_by_run.get(run_id, [])
        day = {
            "trading_date": trading_date,
            "run_id": run_id,
            "expected_snapshots": len(run_links),
            "available_snapshots": 0,
            "acceptable_snapshots": 0,
            "write_failures": 0,
            "replay_errors": 0,
            "unclassified_differences": 0,
        }
        expected_snapshots += len(run_links)
        for link in run_links:
            _link_run, symbol, snapshot_id, link_status, _error = link
            comparison = comparisons_by_key.get((run_id, str(symbol)))
            if comparison is not None:
                comparison_count += 1
                differences = json.loads(str(comparison[4] or "[]"))
                total_differences += len(differences)
                unclassified = sum(
                    item.get("classification")
                    in {"", "UNCLASSIFIED", None}
                    for item in differences
                )
                if str(comparison[5]) == "UNCLASSIFIED" and not differences:
                    unclassified += 1
                unclassified_differences += unclassified
                day["unclassified_differences"] += unclassified
            if str(link_status) not in {"WRITTEN", "EXISTS"}:
                write_failures += 1
                day["write_failures"] += 1
                continue
            try:
                snapshot = snapshot_store.replay(str(snapshot_id))
            except Exception:
                replay_errors += 1
                day["replay_errors"] += 1
                continue
            available_snapshots += 1
            day["available_snapshots"] += 1
            candidate = dict(snapshot.payload.get("candidate") or {})
            bars = dict(snapshot.payload.get("bars") or {})
            critical_covered += len(
                _CANDIDATE_FIELDS & set(candidate)
            )
            critical_covered += len(
                _BAR_FIELDS & set(bars.get("columns") or [])
            )
            manifest_by_source = {
                entry.source: entry for entry in snapshot.source_manifest
            }
            critical_covered += len(
                _MANIFEST_SOURCES & set(manifest_by_source)
            )
            sources_acceptable = all(
                manifest_by_source.get(source) is not None
                and manifest_by_source[source].status
                not in {
                    ResearchSourceStatus.FAILED,
                    ResearchSourceStatus.MISSING,
                }
                for source in _MANIFEST_SOURCES
            )
            if (
                snapshot.quality
                not in {ResearchQuality.PARTIAL, ResearchQuality.FAILED}
                and sources_acceptable
            ):
                acceptable_snapshots += 1
                day["acceptable_snapshots"] += 1
        daily.append(day)

    snapshot_coverage = _ratio(
        available_snapshots,
        expected_snapshots,
    )
    acceptable_quality_coverage = _ratio(
        acceptable_snapshots,
        expected_snapshots,
    )
    comparison_coverage = _ratio(
        comparison_count,
        expected_snapshots,
    )
    critical_field_coverage = _ratio(
        critical_covered,
        expected_snapshots * _CRITICAL_SLOT_COUNT,
    )
    observed_days = len(selected_dates)
    passed = (
        observed_days >= required_days
        and expected_snapshots > 0
        and snapshot_coverage >= min_coverage
        and acceptable_quality_coverage >= min_coverage
        and comparison_coverage >= min_coverage
        and critical_field_coverage >= min_coverage
        and write_failures == 0
        and replay_errors == 0
        and unclassified_differences == 0
    )
    report = {
        "generated_at": (
            generated_at or datetime.now(timezone.utc)
        ).astimezone(timezone.utc).isoformat(),
        "required_days": required_days,
        "observed_trading_days": observed_days,
        "window_start": selected_dates[0] if selected_dates else None,
        "window_end": selected_dates[-1] if selected_dates else None,
        "min_coverage": min_coverage,
        "expected_snapshots": expected_snapshots,
        "available_snapshots": available_snapshots,
        "acceptable_snapshots": acceptable_snapshots,
        "snapshot_coverage": snapshot_coverage,
        "acceptable_quality_coverage": acceptable_quality_coverage,
        "comparison_coverage": comparison_coverage,
        "critical_field_coverage": critical_field_coverage,
        "write_failures": write_failures,
        "replay_errors": replay_errors,
        "total_differences": total_differences,
        "unclassified_differences": unclassified_differences,
        "passed": passed,
        "daily": daily,
    }
    fingerprint = json.dumps(
        report,
        sort_keys=True,
        separators=(",", ":"),
    )
    report["report_id"] = (
        "snapshot-quality-"
        + hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:20]
    )
    return report


def save_snapshot_quality_report(
    db_path: str | Path,
    report: dict[str, Any],
) -> bool:
    conn = duckdb.connect(str(db_path))
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS research_snapshot_quality_reports (
                report_id TEXT PRIMARY KEY,
                generated_at TIMESTAMPTZ,
                window_start TEXT,
                window_end TEXT,
                passed BOOLEAN,
                payload TEXT
            )
            """
        )
        existing = conn.execute(
            "SELECT 1 FROM research_snapshot_quality_reports "
            "WHERE report_id=?",
            [report["report_id"]],
        ).fetchone()
        if existing:
            return False
        conn.execute(
            "INSERT INTO research_snapshot_quality_reports VALUES (?,?,?,?,?,?)",
            [
                report["report_id"],
                datetime.fromisoformat(report["generated_at"]),
                report["window_start"],
                report["window_end"],
                report["passed"],
                json.dumps(
                    report,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ],
        )
        conn.commit()
        return True
    finally:
        conn.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit frozen ResearchSnapshot quality and comparisons."
    )
    parser.add_argument("--db", required=True)
    parser.add_argument("--required-days", type=int, default=10)
    parser.add_argument("--min-coverage", type=float, default=0.95)
    parser.add_argument("--save", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = generate_snapshot_quality_report(
        args.db,
        required_days=args.required_days,
        min_coverage=args.min_coverage,
    )
    if args.save:
        save_snapshot_quality_report(args.db, report)
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
