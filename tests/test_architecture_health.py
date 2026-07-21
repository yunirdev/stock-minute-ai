from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import duckdb
import pytest

from trader.bug_reporting import BugReporter
from trader.health_check import _looks_like_mojibake, check_architecture
from trader.ui_health import (
    UI_HEALTH_SCRIPT,
    UIHealthReport,
    format_health_age,
    record_ui_health,
)
from trader.watchdog import HeartbeatWatchdog


def _healthy_db(path: str, timestamp: datetime) -> None:
    con = duckdb.connect(path)
    con.execute("CREATE TABLE heartbeat (ts TIMESTAMPTZ, tick_count INTEGER, equity DOUBLE)")
    con.execute("INSERT INTO heartbeat VALUES (?, 1, 100000)", [timestamp])
    con.execute("CREATE TABLE bug_issues (fingerprint TEXT)")
    con.execute("CREATE TABLE bug_events (event_id TEXT)")
    con.execute("CREATE TABLE order_intents (intent_id TEXT)")
    con.close()


def test_mojibake_detector_catches_corrupted_visible_text():
    corrupted = chr(0xC3) * 3
    assert _looks_like_mojibake(corrupted)
    assert not _looks_like_mojibake("美股实时监控")

def test_watchdog_reads_canonical_heartbeat_table(tmp_path):
    db_path = str(tmp_path / "trade.duckdb")
    _healthy_db(db_path, datetime.now(timezone.utc))

    assert HeartbeatWatchdog(db_path=db_path).check() == []


def test_independent_health_check_accepts_fresh_runtime(tmp_path):
    now = datetime.now(timezone.utc)
    db_path = str(tmp_path / "trade.duckdb")
    heartbeat_path = tmp_path / "heartbeat.json"
    _healthy_db(db_path, now)
    heartbeat_path.write_text(json.dumps({"ts": now.isoformat()}), encoding="utf-8")

    assert check_architecture(
        db_path,
        heartbeat_path=heartbeat_path,
        now=now,
    ) == []


def test_independent_health_check_detects_stale_runtime_and_ui(tmp_path):
    now = datetime.now(timezone.utc)
    db_path = str(tmp_path / "trade.duckdb")
    _healthy_db(db_path, now - timedelta(minutes=10))

    findings = check_architecture(
        db_path,
        heartbeat_path=tmp_path / "missing.json",
        max_stale_seconds=120,
        ui_url="http://127.0.0.1:1",
        now=now,
    )

    assert {item.code for item in findings} == {"heartbeat_stale", "ui_unreachable"}


def test_browser_health_report_is_persisted(tmp_path):
    db_path = str(tmp_path / "trade.duckdb")
    report = UIHealthReport(
        kind="horizontal_overflow",
        message="Page content exceeds viewport width",
        path="/",
        width=1280,
        height=720,
        details={"overflow": 40},
    )

    fingerprint = record_ui_health(report, db_path)

    issue = BugReporter(db_path, "ui").unresolved()[0]
    assert issue.fingerprint == fingerprint
    event = BugReporter(db_path, "ui").recent_events(1)[0]
    assert event["operation"] == "ui.browser_health"
    assert "overflow" in event["context_json"]


def test_health_age_is_readable_for_long_outages():
    assert format_health_age(59) == "59s"
    assert format_health_age(600) == "10m"
    assert format_health_age(7200) == "2h"
    assert format_health_age(31 * 86400) == "31d"

def test_ui_probe_covers_runtime_and_layout_failures():
    assert "unhandledrejection" in UI_HEALTH_SCRIPT
    assert "horizontal_overflow" in UI_HEALTH_SCRIPT
    assert "ui_root_missing" in UI_HEALTH_SCRIPT
    assert "elements_clipped_by_viewport" in UI_HEALTH_SCRIPT


def test_startup_failure_is_collected(monkeypatch, tmp_path):
    import trader.main as main_module

    db_path = str(tmp_path / "trade.duckdb")
    monkeypatch.setenv("TRADE_DB_PATH", db_path)

    def fail() -> None:
        raise RuntimeError("startup boom")

    monkeypatch.setattr(main_module, "main", fail)
    with pytest.raises(RuntimeError, match="startup boom"):
        main_module._run_entrypoint()

    issue = BugReporter(db_path, "startup").unresolved()[0]
    assert issue.error_type == "RuntimeError"

def test_latest_reconciliation_surfaces_block_reason(tmp_path):
    import json

    import duckdb

    from trader.monitor_data import latest_reconciliation

    db_path = str(tmp_path / "reconciliation.duckdb")
    con = duckdb.connect(db_path)
    con.execute(
        """CREATE TABLE reconciliation_reports (
            ts TIMESTAMPTZ, ok BOOLEAN, open_orders INTEGER, positions INTEGER,
            fills_seen INTEGER, unexplained_orders TEXT,
            unexplained_positions TEXT, errors TEXT
        )"""
    )
    con.execute(
        """INSERT INTO reconciliation_reports VALUES
        (NOW(), FALSE, 1, 1, 0, ?, ?, ?)""",
        [json.dumps(["order-1"]), json.dumps(["AAPL"]), json.dumps([])],
    )
    con.close()

    report = latest_reconciliation(db_path)

    assert report is not None
    assert not report["ok"]
    assert "order-1" in report["unexplained_orders"]
    assert "AAPL" in report["unexplained_positions"]
