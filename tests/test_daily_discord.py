from datetime import datetime, timezone
from types import SimpleNamespace

from trader.daily_discord import (
    build_daily_research_message,
    build_signal_report_message,
)
from trader.models import Side
from trader.signal_reports import SignalState


def test_daily_research_message_contains_batch_and_ranked_pick():
    run = SimpleNamespace(
        trading_date="2026-07-27",
        run_id="research-1",
        status="COMPLETED",
        completed_symbols=1,
        failed_symbols=0,
        data_cutoff=datetime(2026, 7, 26, tzinfo=timezone.utc),
    )
    item = SimpleNamespace(
        status="COMPLETED",
        rank=1,
        symbol="AAPL",
        recommendation="BUY",
        ai_score=82,
        screening_score=75,
        thesis="test thesis",
    )
    note = build_daily_research_message(run, [item])
    assert "research-1" in note.body
    assert any("AAPL" in key for key in note.fields)
    assert "82.0" in next(value for key, value in note.fields.items() if "AAPL" in key)


def test_signal_message_uses_canonical_entry_stop_and_target():
    report = SimpleNamespace(
        state=SignalState.READY,
        side=Side.BUY,
        symbol="AAPL",
        ai_run_id="research-1",
        entry_low=99,
        entry_high=101,
        chase_limit=102,
        stop_loss=95,
        take_profit=110,
        reasons=["AI_EVIDENCE_VALID"],
        model_weight_pct=10,
        model_risk_pct=0.5,
        risk_reward=2,
        ai_score=82,
        plan_id="p1",
        realized_pnl=None,
    )
    note = build_signal_report_message(report)
    assert "$99.00" in note.body
    assert "$95.00" in note.body
    assert "$110.00" in note.body
    assert note.plan_id == "p1"
