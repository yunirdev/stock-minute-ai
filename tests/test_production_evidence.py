from datetime import date, datetime, timezone

import duckdb

from trader.production_evidence import ProductionEvidenceCoordinator


class ResearchRun:
    trading_date = "2026-07-27"
    status = "COMPLETED"
    failed_symbols = 0
    run_id = "research-1"


def _position_report():
    return {
        "observation_kind": "REAL",
        "position_mismatches": 0,
        "silent_rewrites": 0,
        "duplicate_adjustments": 0,
    }


def test_runtime_coordinator_records_real_day_once_after_cutoff(tmp_path):
    db_path = tmp_path / "trade.duckdb"
    connection = duckdb.connect(str(db_path))
    connection.execute(
        """
        CREATE TABLE order_intents(
            idempotency_key TEXT, updated_at TIMESTAMPTZ
        )
        """
    )
    connection.close()
    coordinator = ProductionEvidenceCoordinator(db_path)
    now = datetime(2026, 7, 28, 3, 0, tzinfo=timezone.utc)
    first = coordinator.tick(
        now=now,
        research_run=ResearchRun(),
        position_report=_position_report(),
        reconciliation_blocked=False,
    )
    second = coordinator.tick(
        now=now,
        research_run=ResearchRun(),
        position_report=_position_report(),
        reconciliation_blocked=False,
    )
    assert len(first["finalized_observation_ids"]) == 1
    assert second["finalized_observation_ids"] == []
    assert first["report"]["evidence_type"] == "REAL"
    assert first["report"]["observed_sessions"] == 1


def test_runtime_coordinator_backfills_missed_scheduled_day_as_failure(tmp_path):
    db_path = tmp_path / "trade.duckdb"
    coordinator = ProductionEvidenceCoordinator(db_path)
    coordinator.store.register_session(
        evidence_type="REAL",
        session_date=date(2026, 7, 23),
        schedule_version="simple-nyse-v1",
        registered_at=datetime(2026, 7, 23, 20, 0, tzinfo=timezone.utc),
    )
    result = coordinator.tick(
        now=datetime(2026, 7, 25, 3, 0, tzinfo=timezone.utc),
        research_run=None,
        position_report=None,
        reconciliation_blocked=True,
    )
    assert len(result["finalized_observation_ids"]) == 2
    reasons = {item["reason"] for item in result["report"]["failures"]}
    assert "REPORTS_INCOMPLETE" in reasons
    assert "STATE_DIFFERENCE" in reasons
    assert "UNRESOLVED_FAILURE" in reasons
