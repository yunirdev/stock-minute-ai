"""RiskEngine's circuit breaker (daily drawdown halt + consecutive-failure
halt) had zero regression coverage despite gating every plan through
evaluate_plan(). These tests pin down the halt/reset state machine so a
future refactor can't silently break the trip condition or the one
sanctioned way to clear it (reset_halt).
"""
from trader.config import RiskConfig, TradingConfig
from trader.risk_engine import RiskEngine


def _engine(**risk_kwargs) -> RiskEngine:
    return RiskEngine(TradingConfig(risk=RiskConfig(**risk_kwargs)))


def test_check_equity_does_nothing_before_daily_start_is_set():
    engine = _engine(daily_drawdown_limit_pct=0.03)
    engine.check_equity(0.0)
    assert engine.is_halted is False


def test_check_equity_halts_on_daily_drawdown_breach():
    engine = _engine(daily_drawdown_limit_pct=0.03)
    engine.set_daily_start(100_000.0)
    engine.check_equity(96_000.0)  # -4% < -3% limit
    assert engine.is_halted is True
    assert "回撤" in engine.halt_reason


def test_check_equity_does_not_halt_within_limit():
    engine = _engine(daily_drawdown_limit_pct=0.03)
    engine.set_daily_start(100_000.0)
    engine.check_equity(98_000.0)  # -2% within -3% limit
    assert engine.is_halted is False


def test_record_failure_halts_after_threshold():
    engine = _engine(max_consecutive_failures=3)
    engine.record_failure()
    engine.record_failure()
    assert engine.is_halted is False
    engine.record_failure()
    assert engine.is_halted is True
    assert engine.consecutive_failures == 3


def test_record_success_resets_failure_counter():
    engine = _engine(max_consecutive_failures=3)
    engine.record_failure()
    engine.record_failure()
    engine.record_success()
    assert engine.consecutive_failures == 0
    engine.record_failure()
    engine.record_failure()
    assert engine.is_halted is False


def test_reset_halt_clears_halt_and_failure_counter():
    engine = _engine(max_consecutive_failures=1)
    engine.record_failure()
    assert engine.is_halted is True
    engine.reset_halt()
    assert engine.is_halted is False
    assert engine.halt_reason == ""
    assert engine.consecutive_failures == 0


def test_evaluate_plan_rejects_everything_once_halted():
    from trader.models import Side, TradePlan

    engine = _engine(max_consecutive_failures=1)
    engine.record_failure()
    assert engine.is_halted is True

    plan = TradePlan(
        plan_id="p1", symbol="AAPL", side=Side.BUY, action="OPEN",
        entry_price=100.0, stop_loss=90.0, take_profit=120.0, qty=10.0,
    )
    verdict = engine.evaluate_plan(plan, 100_000.0, {})
    assert verdict.approved is False
    assert "熔断" in verdict.reason
