from types import SimpleNamespace

import pytest

from trader.config import RiskConfig, TradingConfig
from trader.models import Side, TradePlan
from trader.order_lifecycle import OrderIntentStore
from trader.risk_engine import RiskEngine
from trader.runtime import Runtime


def _config(db_path: str) -> TradingConfig:
    return TradingConfig(
        db_path=db_path,
        broker_type="alpaca_paper",
        auto_trade_paper=True,
        risk=RiskConfig(
            max_position_pct=0.20,
            max_trade_risk_pct=0.005,
        ),
    )


def _plan(
    *,
    side: Side = Side.BUY,
    entry: float = 100.0,
    stop: float = 90.0,
    qty: float = 50.0,
) -> TradePlan:
    return TradePlan(
        plan_id="risk-plan",
        symbol="AAPL",
        side=side,
        action="OPEN",
        entry_price=entry,
        stop_loss=stop,
        take_profit=120.0 if side == Side.BUY else 80.0,
        qty=qty,
    )


def test_trade_risk_at_configured_boundary_is_approved(tmp_path):
    risk = RiskEngine(_config(str(tmp_path / "trade.duckdb")))

    verdict = risk.evaluate_plan(_plan(qty=50.0), 100_000.0, {})

    assert verdict.approved


@pytest.mark.parametrize(
    "plan",
    [
        _plan(qty=50.0001),
        _plan(side=Side.SELL, stop=110.0, qty=50.0001),
    ],
)
def test_trade_risk_above_limit_is_rejected(tmp_path, plan):
    risk = RiskEngine(_config(str(tmp_path / "trade.duckdb")))

    verdict = risk.evaluate_plan(plan, 100_000.0, {})

    assert not verdict.approved
    assert "单笔止损风险" in verdict.reason


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("entry_price", None),
        ("entry_price", float("nan")),
        ("stop_loss", float("inf")),
        ("qty", float("nan")),
    ],
)
def test_invalid_trade_plan_numbers_fail_closed(tmp_path, field, value):
    risk = RiskEngine(_config(str(tmp_path / "trade.duckdb")))
    plan = _plan()
    setattr(plan, field, value)

    verdict = risk.evaluate_plan(plan, 100_000.0, {})

    assert not verdict.approved
    assert "无效" in verdict.reason


def test_invalid_equity_and_risk_configuration_fail_closed(tmp_path):
    config = _config(str(tmp_path / "trade.duckdb"))
    risk = RiskEngine(config)

    assert not risk.evaluate_plan(_plan(), float("nan"), {}).approved

    config.risk.max_trade_risk_pct = float("nan")
    assert not risk.evaluate_plan(_plan(), 100_000.0, {}).approved


def test_execution_pipeline_rechecks_trade_risk_before_broker_submit(
    tmp_path,
):
    db_path = str(tmp_path / "trade.duckdb")
    config = _config(db_path)

    class Broker:
        submissions = 0

        def place_order(self, _intent):
            self.submissions += 1
            return "broker-order"

    runtime = Runtime.__new__(Runtime)
    runtime._cfg = config
    runtime._risk = RiskEngine(config)
    runtime._order_store = OrderIntentStore(db_path)
    runtime._broker = Broker()
    runtime._kill = SimpleNamespace(engaged=lambda: False)
    runtime._reconciliation_blocked = False
    runtime._open_orders = {}
    runtime._live_plans = {}
    runtime._bug_reporter = SimpleNamespace(capture_exception=lambda *_, **__: None)

    plan = _plan(qty=100.0)
    runtime._execute_via_pipeline(plan, 100_000.0, {})

    assert runtime._broker.submissions == 0
    assert runtime._order_store.list_all() == []
    assert plan.status == "REJECTED"
