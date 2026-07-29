from datetime import date, datetime, timedelta, timezone

import pytest

from trader.closed_loop_delivery import (
    ClosedLoopDeliveryStore,
    REQUIRED_SCENARIOS,
    REQUIRED_STAGE_REFS,
)
from trader.paper_maturity import PaperMaturityStore
from trader.paper_migration_signoff import (
    PaperMigrationSignoffStore,
    paper_call_graph,
)
from trader.paper_resilience import PaperResilienceStore, SCENARIO_SPECS

NOW = datetime(2026, 7, 27, 20, 0, tzinfo=timezone.utc)


def _maturity(evidence_type="SYNTHETIC"):
    return {
        "report_id": f"maturity-{evidence_type}",
        "evidence_type": evidence_type,
        "required_sessions": 60,
        "observed_sessions": 60,
        "ready": True,
    }


def _resilience(evidence_type="SYNTHETIC"):
    return {
        "drill_run_id": f"drill-{evidence_type}",
        "evidence_type": evidence_type,
        "passed": True,
    }


def _kwargs():
    return {
        "closed_loop_evidence": {
            "evidence_id": "closed-loop-1",
            "accepted": True,
        },
        "verification": {
            "pytest_passed": 374,
            "ruff_passed": True,
            "compileall_passed": True,
        },
        "documents": {
            "task_board": "docs/MIGRATION_TASK_BOARD.md",
            "closed_loop_acceptance": "docs/CLOSED_LOOP_ACCEPTANCE.md",
            "maturity_runbook": "docs/PAPER_MATURITY_RUNBOOK.md",
            "signoff_report": "docs/PAPER_MIGRATION_SIGNOFF.md",
        },
        "call_graph": paper_call_graph(),
        "limitations": ["Paper only", "Natural REAL observations pending"],
        "live_authorized": False,
        "created_at": NOW,
    }


def test_architecture_signoff_accepts_isolated_evidence_idempotently(tmp_path):
    store = PaperMigrationSignoffStore(tmp_path / "trade.duckdb")
    kwargs = {
        "signoff_type": "ARCHITECTURE",
        "maturity_report": _maturity(),
        "resilience_report": _resilience(),
        **_kwargs(),
    }
    first = store.record(**kwargs)
    second = store.record(**kwargs)
    assert first == second
    assert first["status"] == "ARCHITECTURE_READY"
    assert first["live_authorized"] is False


def test_final_signoff_requires_real_sixty_session_evidence(tmp_path):
    store = PaperMigrationSignoffStore(tmp_path / "trade.duckdb")
    with pytest.raises(ValueError, match="FINAL_REAL_EVIDENCE_REQUIRED"):
        store.record(
            signoff_type="FINAL_REAL",
            maturity_report=_maturity(),
            resilience_report=_resilience(),
            **_kwargs(),
        )
    final = store.record(
        signoff_type="FINAL_REAL",
        maturity_report=_maturity("REAL"),
        resilience_report=_resilience("REAL"),
        **_kwargs(),
    )
    assert final["status"] == "FINAL_REAL_READY"


def test_signoff_rejects_live_authorization_and_incomplete_graph(tmp_path):
    store = PaperMigrationSignoffStore(tmp_path / "trade.duckdb")
    kwargs = {
        "signoff_type": "ARCHITECTURE",
        "maturity_report": _maturity(),
        "resilience_report": _resilience(),
        **_kwargs(),
    }
    with pytest.raises(ValueError, match="LIVE_AUTHORIZATION_FORBIDDEN"):
        store.record(**{**kwargs, "live_authorized": True})
    with pytest.raises(ValueError, match="CALL_GRAPH_INCOMPLETE"):
        store.record(**{**kwargs, "call_graph": "ResearchSnapshot -> OrderIntent"})


def test_i_stage_architecture_evidence_connects_all_three_gates(tmp_path):
    db_path = tmp_path / "trade.duckdb"
    maturity_store = PaperMaturityStore(db_path)
    start = date(2026, 1, 2)
    for index in range(60):
        session = start + timedelta(days=index)
        maturity_store.register_session(
            evidence_type="SYNTHETIC",
            session_date=session,
            schedule_version="nyse-v1",
            registered_at=NOW,
        )
        maturity_store.record_observation(
            evidence_type="SYNTHETIC",
            session_date=session,
            reports_complete=True,
            unexplained_duplicate_orders=0,
            plan_rewrites=0,
            state_differences=0,
            unresolved_failures=0,
            evidence_refs={
                "daily_report_ref": f"daily-{index}",
                "research_quality_ref": f"research-{index}",
                "position_quality_ref": f"position-{index}",
            },
            observed_at=NOW,
        )
    maturity = maturity_store.build_report(
        evidence_type="SYNTHETIC",
        through_date=start + timedelta(days=59),
        created_at=NOW,
    )
    resilience_store = PaperResilienceStore(db_path)
    for scenario, spec in SCENARIO_SPECS.items():
        resilience_store.record(
            drill_run_id="i-architecture",
            evidence_type="SYNTHETIC",
            scenario=scenario,
            actual_outcome=str(spec["outcome"]),
            submit_count=int(spec["submit_count"]),
            unexpected_submit_count=0,
            audit_ref=f"audit-{scenario}",
            recovery_ref=f"recovery-{scenario}",
            error_code="EXPECTED_FAULT",
            created_at=NOW,
        )
    resilience = resilience_store.build_report(
        drill_run_id="i-architecture",
        evidence_type="SYNTHETIC",
    )
    closed_loop = ClosedLoopDeliveryStore(db_path).record(
        evidence_type="ISOLATED_PAPER",
        stage_refs={key: f"{key}-1" for key in REQUIRED_STAGE_REFS},
        scenarios=dict(REQUIRED_SCENARIOS),
        metrics={
            "data_coverage": 1.0,
            "research_success_rate": 1.0,
            "plan_count": 2.0,
            "order_success_rate": 1.0,
            "fill_rate": 1.0,
            "slippage": 0.0,
            "max_drawdown": 0.0,
            "realized_pnl": 0.0,
        },
        recovery={"API": True, "DATABASE": True, "RUNTIME": True},
        button_report={
            "action_count": 31,
            "success_covered": 31,
            "empty_covered": 31,
            "error_covered": 31,
            "busy_covered": 31,
        },
        limitations=["isolated evidence"],
        created_at=NOW,
    )
    signoff = PaperMigrationSignoffStore(db_path).record(
        signoff_type="ARCHITECTURE",
        maturity_report=maturity,
        resilience_report=resilience,
        closed_loop_evidence=closed_loop,
        **{
            key: value
            for key, value in _kwargs().items()
            if key != "closed_loop_evidence"
        },
    )
    assert signoff["status"] == "ARCHITECTURE_READY"
