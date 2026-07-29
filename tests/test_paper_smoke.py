import pytest

from trader.audit_query import order_traces
from trader.paper_smoke import run_smoke


def test_isolated_paper_smoke_covers_execution_and_restart(tmp_path):
    db_path = tmp_path / "paper-smoke.duckdb"

    report = run_smoke(db_path)

    assert report["ok"]
    assert not report["network_used"]
    assert report["broker_type"] == "alpaca_paper"
    assert report["scenarios"] == {
        "buy_partial": "FILLED",
        "sell": "FILLED",
        "risk_rejected": "REJECTED_WITHOUT_ORDER",
        "unknown_before_restart": "UNKNOWN",
        "unknown_after_recovery": "CANCELED",
        "restart_resubmissions": 0,
    }
    assert report["trace_count"] == 4

    traces = order_traces(
        db_path,
        plan_ids=set(report["plan_ids"]),
    )
    assert len(traces) == 4
    assert all(trace["plan"] for trace in traces)
    assert all(trace["risk_events"] for trace in traces)
    submitted = [trace for trace in traces if trace["order"]]
    assert len(submitted) == 3
    assert all(trace["order"]["idempotency_key"] for trace in submitted)
    assert all(trace["order"]["order_type"] == "LMT" for trace in submitted)
    assert sum(len(trace["fills"]) for trace in submitted) == 3


def test_isolated_paper_smoke_is_repeatable_without_deleting_database(tmp_path):
    db_path = tmp_path / "repeat-smoke.duckdb"

    first = run_smoke(db_path)
    second = run_smoke(db_path)

    assert first["run_id"] != second["run_id"]
    assert first["ok"] and second["ok"]
    assert len(order_traces(db_path)) == 8


def test_paper_smoke_refuses_non_smoke_database_name(tmp_path):
    with pytest.raises(
        ValueError,
        match="SMOKE_DB_NAME_MUST_CONTAIN_SMOKE",
    ):
        run_smoke(tmp_path / "production.duckdb")
