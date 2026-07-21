import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor

from trader.bug_reporting import BugLoggingHandler, BugReporter
from trader.monitor_data import bug_events_df, bug_issues_df


def test_deduplicates_redacts_and_reopens(tmp_path):
    db_path = str(tmp_path / "trade.duckdb")
    reporter = BugReporter(db_path, "runtime")
    message = "request api_key=secret-value failed"

    first = reporter.capture_message(
        message,
        operation="fetch",
        context={"token": "hidden", "symbol": "AAPL"},
    )
    assert reporter.resolve(first)
    second = reporter.capture_message(message, operation="fetch")

    assert first == second
    issue = reporter.unresolved()[0]
    assert issue.occurrence_count == 2
    assert issue.status == "OPEN"
    event = reporter.recent_events(1)[0]
    assert "secret-value" not in event["message"]
    assert "hidden" not in event["context_json"]


def test_captures_exception_links_and_status_changes(tmp_path):
    reporter = BugReporter(str(tmp_path / "trade.duckdb"), "runtime")
    try:
        raise RuntimeError("boom")
    except RuntimeError as exc:
        fingerprint = reporter.capture_exception(
            exc,
            operation="broker.submit",
            symbol="AAPL",
            plan_id="plan-1",
            intent_id="intent-1",
        )

    event = reporter.recent_events(1)[0]
    assert event["fingerprint"] == fingerprint
    assert event["symbol"] == "AAPL"
    assert event["plan_id"] == "plan-1"
    assert event["intent_id"] == "intent-1"
    assert "RuntimeError: boom" in event["traceback"]
    assert reporter.ignore(fingerprint)
    assert reporter.unresolved() == []
    assert reporter.reopen(fingerprint)
    assert len(reporter.unresolved()) == 1
    assert not reporter.resolve("missing")


def test_logging_handler_and_monitor_queries(tmp_path):
    db_path = str(tmp_path / "trade.duckdb")
    reporter = BugReporter(db_path, "runtime")
    handler = BugLoggingHandler(reporter)
    record = logging.LogRecord(
        "test.bug.reporting",
        logging.ERROR,
        __file__,
        1,
        "worker failed",
        (),
        None,
    )
    handler.handle(record)

    issues = bug_issues_df(db_path=db_path)
    events = bug_events_df(db_path=db_path)
    assert issues.iloc[0]["message"] == "worker failed"
    context = json.loads(events.iloc[0]["context_json"])
    assert context["logger"] == "test.bug.reporting"

def test_bug_cli_prints_unicode_events(monkeypatch, capsys, tmp_path):
    from trader import bug_cli

    db_path = str(tmp_path / "trade.duckdb")
    BugReporter(db_path, "startup").capture_message("启动失败")
    monkeypatch.setattr(
        sys,
        "argv",
        ["bug_cli", "--db", db_path, "events", "--limit", "1"],
    )

    assert bug_cli.main() == 0
    assert "启动失败" in capsys.readouterr().out

def test_concurrent_reporters_do_not_conflict(tmp_path):
    db_path = str(tmp_path / "trade.duckdb")

    def capture(_index):
        return BugReporter(db_path, "ui").capture_message(
            "Page content exceeds viewport width",
            error_type="horizontal_overflow",
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        fingerprints = list(pool.map(capture, range(20)))

    assert len(set(fingerprints)) == 1
    reporter = BugReporter(db_path, "ui")
    assert reporter.unresolved()[0].occurrence_count == 20
    assert len(reporter.recent_events(25)) == 20

def test_bug_cli_reports_canonical_resolved_status(monkeypatch, capsys, tmp_path):
    from trader import bug_cli

    db_path = str(tmp_path / "trade.duckdb")
    fingerprint = BugReporter(db_path, "runtime").capture_message("boom")
    monkeypatch.setattr(
        sys,
        "argv",
        ["bug_cli", "--db", db_path, "resolve", fingerprint],
    )

    assert bug_cli.main() == 0
    assert json.loads(capsys.readouterr().out)["status"] == "RESOLVED"