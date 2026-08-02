"""周报自动化 + 晨报承接信息。"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from trader.teams.maintenance import (
    _default_period_label,
    should_send_weekly_report,
)

_ET = ZoneInfo("America/New_York")


def _et(day, hour, minute):
    return datetime(2026, 8, day, hour, minute, tzinfo=_ET).astimezone(timezone.utc)


# ── 周报调度 ─────────────────────────────────────────────────────────────────
# 2026-08-07 是周五，08-06 周四，08-08 周六


def test_fires_friday_after_the_close():
    assert should_send_weekly_report(_et(7, 16, 30), None)


def test_not_before_the_close():
    assert not should_send_weekly_report(_et(7, 15, 0), None)


def test_not_on_other_weekdays():
    assert not should_send_weekly_report(_et(6, 16, 30), None)
    assert not should_send_weekly_report(_et(8, 16, 30), None)


def test_once_per_week():
    year, week, _ = datetime(2026, 8, 7, tzinfo=_ET).isocalendar()
    label = f"{year}-W{week:02d}"
    assert not should_send_weekly_report(_et(7, 17, 0), label)


def test_late_start_still_catches_up():
    """引擎周五傍晚才起来，这一周的体检不该就此消失。"""
    assert should_send_weekly_report(_et(7, 18, 0), None)


def test_too_late_is_skipped():
    assert not should_send_weekly_report(_et(7, 22, 0), None)


# ── 去重身份 ─────────────────────────────────────────────────────────────────


def test_weekly_label_is_iso_week():
    label = _default_period_label(24 * 7)
    assert label.count("-W") == 1


def test_daily_label_is_a_date():
    label = _default_period_label(24)
    assert len(label) == 10 and label.count("-") == 2


def test_manual_run_shares_the_identity_of_the_auto_run():
    """周五下午手动点一次维护分析，不该和自动周报各推一条。"""
    assert _default_period_label(24 * 7) == _default_period_label(24 * 8)


# ── 晨报承接 ─────────────────────────────────────────────────────────────────


def test_carryover_lists_stops_and_waiting_plans(tmp_path, monkeypatch):
    import trader.morning_brief as mb
    from trader.models import Side
    from trader.signal_reports import SignalReport, SignalState, SignalStore

    db = str(tmp_path / "carry.duckdb")
    store = SignalStore(db)
    now = datetime.now(timezone.utc)

    def _mk(symbol, state, version):
        return SignalReport(
            signal_id=f"sig-{symbol}", version=version, symbol=symbol, state=state,
            side=Side.BUY, strategy="orb", timeframe="5m", market_regime="bull",
            market_price=101.0, market_data_at=now, generated_at=now,
            valid_until=now + timedelta(hours=6), entry_low=100.0, entry_high=102.0,
            chase_limit=103.0, stop_loss=98.0, take_profit=110.0, risk_reward=2.5,
            model_weight_pct=5.0, model_risk_pct=0.5, ai_score=78.0,
            ai_run_id="run-1", ai_contributors=["quant"], decision_id="d",
            plan_id=f"plan-{symbol}",
        )

    store._insert(_mk("NVDA", SignalState.ENTERED, 1))
    store._insert(_mk("SNOW", SignalState.READY, 1))

    text = mb._build_carryover_section(db)
    assert "NVDA" in text and "98.00" in text and "110.00" in text
    assert "SNOW" in text and "100.00" in text


def test_carryover_is_empty_when_nothing_is_open(tmp_path):
    import trader.morning_brief as mb
    from trader.signal_reports import SignalStore

    db = str(tmp_path / "empty.duckdb")
    SignalStore(db)
    assert mb._build_carryover_section(db) == ""


def test_automation_notice_states_whether_the_engine_will_trade():
    """读者有权在开盘前知道引擎今天会不会自己下单。"""
    import trader.morning_brief as mb

    assert "自动交易已开启" in mb._build_automation_notice(True)
    assert "自动交易已关闭" in mb._build_automation_notice(False)


def test_automation_notice_stays_silent_when_state_is_unknown(monkeypatch):
    """宁可不说，也不能显示一个可能与实际相反的自动化状态。"""
    import trader.morning_brief as mb

    monkeypatch.delenv("AUTO_TRADE_PAPER", raising=False)
    assert mb._build_automation_notice(None) == ""


def test_automation_notice_falls_back_to_env(monkeypatch):
    import trader.morning_brief as mb

    monkeypatch.setenv("AUTO_TRADE_PAPER", "true")
    assert "自动交易已开启" in mb._build_automation_notice(None)
