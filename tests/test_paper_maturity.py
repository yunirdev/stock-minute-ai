from datetime import date, datetime, timedelta, timezone

import pytest

from trader.paper_maturity import PaperMaturityStore

NOW = datetime(2026, 7, 27, 20, 0, tzinfo=timezone.utc)


def _record_days(
    store,
    *,
    evidence_type="SYNTHETIC",
    count=60,
    failed_index=None,
):
    start = date(2026, 1, 2)
    for index in range(count):
        session = start + timedelta(days=index)
        store.register_session(
            evidence_type=evidence_type,
            session_date=session,
            schedule_version="nyse-v1",
            registered_at=NOW,
        )
        store.record_observation(
            evidence_type=evidence_type,
            session_date=session,
            reports_complete=index != failed_index,
            unexplained_duplicate_orders=0,
            plan_rewrites=0,
            state_differences=1 if index == failed_index else 0,
            unresolved_failures=0,
            evidence_refs={
                "daily_report_ref": f"daily-{index}",
                "research_quality_ref": f"research-{index}",
                "position_quality_ref": f"position-{index}",
            },
            observed_at=NOW,
        )
    return start + timedelta(days=count - 1)


def test_sixty_session_synthetic_gate_and_real_isolation(tmp_path):
    store = PaperMaturityStore(tmp_path / "trade.duckdb")
    through = _record_days(store)
    synthetic = store.build_report(
        evidence_type="SYNTHETIC",
        through_date=through,
        created_at=NOW,
    )
    real = store.build_report(
        evidence_type="REAL",
        through_date=through,
        created_at=NOW,
    )
    assert synthetic["ready"] is True
    assert synthetic["scheduled_sessions"] == 60
    assert synthetic["observed_sessions"] == 60
    assert real["ready"] is False
    assert real["scheduled_sessions"] == 0


def test_missing_and_failed_scheduled_days_fail_gate(tmp_path):
    store = PaperMaturityStore(tmp_path / "trade.duckdb")
    through = _record_days(store, count=59, failed_index=4)
    missing_session = through + timedelta(days=1)
    store.register_session(
        evidence_type="SYNTHETIC",
        session_date=missing_session,
        schedule_version="nyse-v1",
        registered_at=NOW,
    )
    report = store.build_report(
        evidence_type="SYNTHETIC",
        through_date=missing_session,
        created_at=NOW,
    )
    reasons = {item["reason"] for item in report["failures"]}
    assert report["ready"] is False
    assert "REPORTS_INCOMPLETE" in reasons
    assert "STATE_DIFFERENCE" in reasons
    assert "MISSING_OBSERVATION" in reasons


def test_schedule_and_observation_are_immutable(tmp_path):
    store = PaperMaturityStore(tmp_path / "trade.duckdb")
    session = date(2026, 1, 2)
    store.register_session(
        evidence_type="REAL",
        session_date=session,
        schedule_version="nyse-v1",
        registered_at=NOW,
    )
    with pytest.raises(ValueError, match="SCHEDULE_REWRITE"):
        store.register_session(
            evidence_type="REAL",
            session_date=session,
            schedule_version="nyse-v2",
            registered_at=NOW,
        )
    kwargs = {
        "evidence_type": "REAL",
        "session_date": session,
        "reports_complete": True,
        "unexplained_duplicate_orders": 0,
        "plan_rewrites": 0,
        "state_differences": 0,
        "unresolved_failures": 0,
        "evidence_refs": {
            "daily_report_ref": "daily",
            "research_quality_ref": "research",
            "position_quality_ref": "position",
        },
        "observed_at": NOW,
    }
    first = store.record_observation(**kwargs)
    assert store.record_observation(**kwargs) == first
    with pytest.raises(ValueError, match="OBSERVATION_REWRITE"):
        store.record_observation(**{**kwargs, "plan_rewrites": 1})
