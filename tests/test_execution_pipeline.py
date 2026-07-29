from datetime import datetime, timedelta, timezone

import pytest

from trader.execution_pipeline import (
    ExecutionPipelineStore,
    candidate_from_trade_plan,
)
from trader.models import (
    CandidatePlan,
    RiskVerdict,
    Side,
    TradePlan,
)

NOW = datetime(2026, 7, 27, 14, 0, tzinfo=timezone.utc)


def _candidate(*, valid_until=None, evidence=("snapshot-1",)):
    plan = TradePlan(
        plan_id="legacy-trade-plan",
        symbol="AAPL",
        side=Side.BUY,
        action="OPEN",
        entry_price=100,
        stop_loss=95,
        take_profit=115,
        qty=10,
        created_at=NOW,
    )
    return candidate_from_trade_plan(
        plan,
        decision_id="decision-1",
        strategy_version="strategy-v1",
        data_version="snapshot-hash-1",
        evidence_refs=evidence,
        valid_until=valid_until or NOW + timedelta(minutes=30),
    )


def test_candidate_final_intent_chain_has_stable_references(tmp_path):
    store = ExecutionPipelineStore(tmp_path / "trade.duckdb")
    candidate = store.register_candidate(_candidate())
    store.validate_candidate(candidate.candidate_plan_id, now=NOW)
    final = store.finalize(
        candidate.candidate_plan_id,
        risk_verdict=RiskVerdict(True, "approved", suggested_qty=8),
        risk_check_id="risk-check-1",
        risk_config_version="risk-v1",
        now=NOW + timedelta(seconds=1),
    )
    intent = store.create_order_intent(
        final.final_plan_id,
        now=NOW + timedelta(seconds=2),
    )

    assert final.side == candidate.side
    assert final.qty == 8
    assert intent.candidate_plan_id == candidate.candidate_plan_id
    assert intent.final_plan_id == final.final_plan_id
    assert intent.final_plan_version == 1
    assert intent.risk_check_id == "risk-check-1"
    assert intent.evidence_refs == ("snapshot-1",)
    assert intent.order_type == "LMT"


def test_invalid_transitions_and_duplicate_intent_fail_closed(tmp_path):
    store = ExecutionPipelineStore(tmp_path / "trade.duckdb")
    candidate = store.register_candidate(_candidate())
    with pytest.raises(ValueError, match="NOT_VALIDATED"):
        store.finalize(
            candidate.candidate_plan_id,
            risk_verdict=RiskVerdict(True, "approved", 10),
            risk_check_id="risk-1",
            risk_config_version="risk-v1",
            now=NOW,
        )
    store.validate_candidate(candidate.candidate_plan_id, now=NOW)
    with pytest.raises(ValueError, match="TRANSITION_INVALID"):
        store.validate_candidate(candidate.candidate_plan_id, now=NOW)
    final = store.finalize(
        candidate.candidate_plan_id,
        risk_verdict=RiskVerdict(True, "approved", 10),
        risk_check_id="risk-1",
        risk_config_version="risk-v1",
        now=NOW,
    )
    store.create_order_intent(final.final_plan_id, now=NOW)
    with pytest.raises(ValueError, match="TRANSITION_INVALID"):
        store.create_order_intent(final.final_plan_id, now=NOW)


def test_expiry_and_risk_rejection_fail_closed(tmp_path):
    expired_store = ExecutionPipelineStore(tmp_path / "expired.duckdb")
    expired = expired_store.register_candidate(
        _candidate(valid_until=NOW + timedelta(seconds=1))
    )
    with pytest.raises(ValueError, match="EXPIRED"):
        expired_store.validate_candidate(
            expired.candidate_plan_id,
            now=NOW + timedelta(seconds=2),
        )

    risk_store = ExecutionPipelineStore(tmp_path / "risk.duckdb")
    candidate = risk_store.register_candidate(_candidate())
    risk_store.validate_candidate(candidate.candidate_plan_id, now=NOW)
    with pytest.raises(ValueError, match="RISK_NOT_APPROVED"):
        risk_store.finalize(
            candidate.candidate_plan_id,
            risk_verdict=RiskVerdict(False, "blocked", 0),
            risk_check_id="risk-2",
            risk_config_version="risk-v1",
            now=NOW,
        )


def test_evidence_and_direction_are_mandatory():
    with pytest.raises(ValueError, match="EVIDENCE_REQUIRED"):
        _candidate(evidence=())
    with pytest.raises(ValueError, match="DIRECTION_INVALID"):
        CandidatePlan(
            **{
                **_candidate().__dict__,
                "candidate_plan_id": "bad-direction",
                "side": Side.SELL,
            }
        )
