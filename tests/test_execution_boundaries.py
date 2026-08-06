import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from trader.broker.alpaca import AlpacaBroker
from trader.config import RiskConfig, TradingConfig
from trader.models import OrderIntent, Side, TradePlan
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


def _plan() -> TradePlan:
    return TradePlan(
        plan_id="safety-plan",
        symbol="AAPL",
        side=Side.BUY,
        action="OPEN",
        entry_price=100.0,
        stop_loss=99.0,
        take_profit=102.0,
        qty=10.0,
    )


def _runtime(config, broker, store, *, kill_engaged=False) -> Runtime:
    runtime = Runtime.__new__(Runtime)
    runtime._cfg = config
    runtime._risk = RiskEngine(config)
    runtime._order_store = store
    runtime._broker = broker
    runtime._kill = SimpleNamespace(engaged=lambda: kill_engaged)
    runtime._reconciliation_blocked = False
    runtime._open_orders = {}
    runtime._live_plans = {}
    runtime._bug_reporter = SimpleNamespace(
        capture_exception=lambda *_, **__: None
    )
    return runtime


def test_unknown_broker_type_fails_closed():
    with pytest.raises(ValueError, match="UNSUPPORTED_BROKER_TYPE"):
        TradingConfig(broker_type="custom")


def test_runtime_submit_boundary_rechecks_paper_mode(tmp_path):
    db_path = str(tmp_path / "trade.duckdb")
    config = _config(db_path)
    config.broker_type = "alpaca_live"
    broker = SimpleNamespace(calls=0)

    def place_order(_intent):
        broker.calls += 1
        return "must-not-submit"

    broker.place_order = place_order
    store = OrderIntentStore(db_path)
    runtime = _runtime(config, broker, store)

    runtime._execute_via_pipeline(_plan(), 100_000.0, {})

    assert broker.calls == 0
    assert store.list_all() == []


def test_runtime_submit_boundary_rechecks_kill_switch(tmp_path):
    db_path = str(tmp_path / "trade.duckdb")
    config = _config(db_path)
    broker = SimpleNamespace(calls=0)

    def place_order(_intent):
        broker.calls += 1
        return "must-not-submit"

    broker.place_order = place_order
    store = OrderIntentStore(db_path)
    runtime = _runtime(config, broker, store, kill_engaged=True)

    runtime._execute_via_pipeline(_plan(), 100_000.0, {})

    assert broker.calls == 0
    assert store.list_all() == []


def test_runtime_constructs_only_lmt_intents(tmp_path):
    db_path = str(tmp_path / "trade.duckdb")
    config = _config(db_path)
    captured = []
    broker = SimpleNamespace(
        place_order=lambda intent: captured.append(intent) or "broker-1"
    )
    store = OrderIntentStore(db_path)
    runtime = _runtime(config, broker, store)

    runtime._execute_via_pipeline(_plan(), 100_000.0, {})

    assert len(captured) == 1
    assert captured[0].order_type == "LMT"
    # 入场限价单带 marketable-limit 缓冲（+0.15%），不再是原始 entry_price
    # 原样提交——reference_price 才是那个"干净"的计划价。
    assert captured[0].limit_price == 100.15
    assert captured[0].reference_price == 100.0


def test_alpaca_adapter_rejects_live_order_submission():
    broker = AlpacaBroker.__new__(AlpacaBroker)
    broker._paper = False
    intent = OrderIntent(
        intent_id="intent-live",
        signal_id="plan-live",
        symbol="AAPL",
        side=Side.BUY,
        qty=1.0,
        order_type="LMT",
        limit_price=100.0,
    )

    with pytest.raises(RuntimeError, match="LIVE_ORDER_SUBMISSION_DISABLED"):
        broker.place_order(intent)


def test_alpaca_adapter_rejects_non_limit_order():
    broker = AlpacaBroker.__new__(AlpacaBroker)
    broker._paper = True
    intent = OrderIntent(
        intent_id="intent-market",
        signal_id="plan-market",
        symbol="AAPL",
        side=Side.BUY,
        qty=1.0,
        order_type="MKT",
        limit_price=None,
    )

    with pytest.raises(ValueError, match="AUTOMATIC_EXECUTION_REQUIRES_LMT"):
        broker.place_order(intent)


def test_alpaca_position_and_cash_failures_are_not_reported_as_empty():
    class FailingClient:
        def get_all_positions(self):
            raise ConnectionError("position API unavailable")

        def get_account(self):
            raise ConnectionError("account API unavailable")

    broker = AlpacaBroker.__new__(AlpacaBroker)
    broker._client = FailingClient()

    with pytest.raises(ConnectionError, match="position API unavailable"):
        broker.get_positions()
    with pytest.raises(ConnectionError, match="account API unavailable"):
        broker.get_account_cash()


def test_runtime_is_the_only_production_order_submission_caller():
    trader_root = Path(__file__).parents[1] / "trader"
    callers = []
    for path in trader_root.rglob("*.py"):
        tree = ast.parse(
            path.read_text(encoding="utf-8-sig"),
            filename=str(path),
        )
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "place_order"
            ):
                callers.append(path.relative_to(trader_root).as_posix())

    assert callers == ["runtime.py"]


def test_runtime_submission_is_reached_only_from_execution_pipeline():
    runtime_path = Path(__file__).parents[1] / "trader" / "runtime.py"
    tree = ast.parse(
        runtime_path.read_text(encoding="utf-8-sig"),
        filename=str(runtime_path),
    )
    direct_call_owners = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != "Runtime":
            continue
        for function in node.body:
            if not isinstance(
                function,
                (ast.FunctionDef, ast.AsyncFunctionDef),
            ):
                continue
            if any(
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "_submit_pipeline_intent"
                for call in ast.walk(function)
            ):
                direct_call_owners.append(function.name)

    assert direct_call_owners == ["_execute_via_pipeline"]
