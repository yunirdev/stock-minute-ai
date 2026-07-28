from datetime import datetime, timezone

import pytest

from trader.paper_resilience import PaperResilienceStore, SCENARIO_SPECS

NOW = datetime(2026, 7, 27, 20, 0, tzinfo=timezone.utc)


def _record_matrix(store, *, evidence_type="SYNTHETIC", failed=""):
    for scenario, spec in SCENARIO_SPECS.items():
        store.record(
            drill_run_id="drill-1",
            evidence_type=evidence_type,
            scenario=scenario,
            actual_outcome=(
                "SUBMITTED" if scenario == failed else str(spec["outcome"])
            ),
            submit_count=(
                int(spec["submit_count"]) + (1 if scenario == failed else 0)
            ),
            unexpected_submit_count=1 if scenario == failed else 0,
            audit_ref=f"audit-{scenario}",
            recovery_ref=f"recovery-{scenario}",
            error_code="EXPECTED_FAULT",
            created_at=NOW,
        )


def test_complete_fault_matrix_passes_with_zero_unexpected_submit(tmp_path):
    store = PaperResilienceStore(tmp_path / "trade.duckdb")
    _record_matrix(store)
    report = store.build_report(
        drill_run_id="drill-1",
        evidence_type="SYNTHETIC",
    )
    assert report["passed"] is True
    assert report["observed_scenarios"] == sorted(SCENARIO_SPECS)
    assert report["unexpected_submit_count"] == 0


def test_missing_and_unexpected_submit_fail_matrix(tmp_path):
    store = PaperResilienceStore(tmp_path / "trade.duckdb")
    _record_matrix(store, failed="KILL_SWITCH")
    report = store.build_report(
        drill_run_id="drill-1",
        evidence_type="SYNTHETIC",
    )
    assert report["passed"] is False
    assert report["failed_scenarios"] == ["KILL_SWITCH"]
    assert report["unexpected_submit_count"] == 1


def test_drill_is_idempotent_and_conflicting_rewrite_fails(tmp_path):
    store = PaperResilienceStore(tmp_path / "trade.duckdb")
    kwargs = {
        "drill_run_id": "drill-1",
        "evidence_type": "REAL",
        "scenario": "MISSING_DATA",
        "actual_outcome": "BLOCKED",
        "submit_count": 0,
        "unexpected_submit_count": 0,
        "audit_ref": "audit-1",
        "recovery_ref": "recovery-1",
        "error_code": "DATA_MISSING",
        "created_at": NOW,
    }
    first = store.record(**kwargs)
    assert store.record(**kwargs)["drill_id"] == first["drill_id"]
    with pytest.raises(ValueError, match="DRILL_REWRITE"):
        store.record(**{**kwargs, "actual_outcome": "SUBMITTED"})
