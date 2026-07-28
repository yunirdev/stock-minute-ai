from datetime import datetime, timezone

import pytest

from trader.closed_loop_delivery import (
    REQUIRED_SCENARIOS,
    REQUIRED_STAGE_REFS,
    ClosedLoopDeliveryStore,
    render_call_graph,
)
from trader.paper_smoke import run_smoke

NOW = datetime(2026, 7, 27, 20, 0, tzinfo=timezone.utc)


def _refs(run_id):
    return {key: f"{key}-{run_id}" for key in REQUIRED_STAGE_REFS}


def _metrics():
    return {
        "data_coverage": 1.0,
        "research_success_rate": 1.0,
        "plan_count": 4,
        "order_success_rate": 1.0,
        "fill_rate": 1.0,
        "slippage": 0.0,
        "max_drawdown": 0.01,
        "realized_pnl": 10.0,
    }


def _buttons():
    return {
        "action_count": 31,
        "success_covered": 31,
        "empty_covered": 31,
        "error_covered": 31,
        "busy_covered": 31,
    }


def test_isolated_paper_scenarios_form_complete_frozen_delivery_evidence(tmp_path):
    smoke = run_smoke(tmp_path / "closed-loop-smoke.duckdb")
    refs = _refs(smoke["run_id"])
    store = ClosedLoopDeliveryStore(tmp_path / "delivery.duckdb")
    evidence = store.record(
        evidence_type="ISOLATED_PAPER",
        stage_refs=refs,
        scenarios=smoke["scenarios"],
        metrics=_metrics(),
        recovery={"API": True, "DATABASE": True, "RUNTIME": True},
        button_report=_buttons(),
        limitations=[
            "Paper only",
            "No live authorization",
            "No automatic strategy parameter promotion",
        ],
        created_at=NOW,
    )
    assert evidence["accepted"]
    assert store.record(
        evidence_type="ISOLATED_PAPER",
        stage_refs=refs,
        scenarios=smoke["scenarios"],
        metrics=_metrics(),
        recovery={"API": True, "DATABASE": True, "RUNTIME": True},
        button_report=_buttons(),
        limitations=evidence["limitations"],
        created_at=NOW,
    ) == evidence
    graph = render_call_graph(refs)
    assert all(value in graph for value in refs.values())


def test_missing_scenario_stage_or_recovery_fails_closed(tmp_path):
    store = ClosedLoopDeliveryStore(tmp_path / "delivery.duckdb")
    with pytest.raises(ValueError, match="STAGE_REFERENCE_MISSING"):
        store.record(
            evidence_type="ISOLATED_PAPER",
            stage_refs={},
            scenarios=REQUIRED_SCENARIOS,
            metrics=_metrics(),
            recovery={"API": True, "DATABASE": True, "RUNTIME": True},
            button_report=_buttons(),
            limitations=["Paper only"],
            created_at=NOW,
        )
    rejected = store.record(
        evidence_type="ISOLATED_PAPER",
        stage_refs=_refs("run-2"),
        scenarios=REQUIRED_SCENARIOS,
        metrics=_metrics(),
        recovery={"API": True, "DATABASE": False, "RUNTIME": True},
        button_report=_buttons(),
        limitations=["Paper only"],
        created_at=NOW,
    )
    assert not rejected["accepted"]
