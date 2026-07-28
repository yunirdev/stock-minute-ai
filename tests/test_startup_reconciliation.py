from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import duckdb

from trader.config import TradingConfig
from trader.models import Fill, OrderIntent, Position, Side
from trader.order_lifecycle import (
    OrderIntentStore,
    OrderLifecycle,
    idempotency_key,
)
from trader.portfolio import Portfolio
from trader.runtime import Runtime


def _config(db_path: str) -> TradingConfig:
    return TradingConfig(
        db_path=db_path,
        broker_type="alpaca_paper",
        auto_trade_paper=True,
    )


def _intent_store_with_order(
    db_path: str,
    *,
    state: OrderLifecycle,
    broker_order_id: str | None = None,
) -> tuple[OrderIntentStore, str]:
    store = OrderIntentStore(db_path)
    key = idempotency_key("plan-1", "AAPL", "BUY", 5.0, 100.0, "OPEN")
    intent = OrderIntent(
        intent_id="intent-1",
        signal_id="plan-1",
        symbol="AAPL",
        side=Side.BUY,
        qty=5.0,
        order_type="LMT",
        limit_price=100.0,
        plan_id="plan-1",
    )
    store.persist(intent, key, "plan-1", state=state)
    if broker_order_id:
        store.update(key, broker_order_id=broker_order_id)
    return store, key


class _AuditCapture:
    def __init__(self):
        self.reports = []

    def log_reconciliation(self, report):
        self.reports.append(report)


def _runtime(config, broker, store, portfolio=None):
    runtime = Runtime.__new__(Runtime)
    runtime._broker = broker
    runtime._order_store = store
    runtime._portfolio = portfolio or Portfolio(config)
    runtime._audit = _AuditCapture()
    runtime._signal_store = SimpleNamespace(apply_fill=lambda *_: None)
    runtime._open_orders = {}
    runtime._bug_reporter = SimpleNamespace(capture_exception=lambda *_, **__: None)
    runtime._reconciliation_blocked = False
    return runtime


def test_portfolio_restores_positions_from_durable_fill_deltas(tmp_path):
    db_path = str(tmp_path / "trade.duckdb")
    config = _config(db_path)
    portfolio = Portfolio(config)
    portfolio.apply_fill(
        Fill(
            "broker-1",
            "intent-1",
            "AAPL",
            Side.BUY,
            5.0,
            100.0,
            datetime.now(timezone.utc),
        )
    )

    restarted = Portfolio(config)

    assert restarted.positions["AAPL"].qty == 5.0
    assert restarted.positions["AAPL"].avg_entry_px == 100.0


def test_broker_baseline_preserves_old_fills_and_replays_only_newer_ones(
    tmp_path,
):
    db_path = str(tmp_path / "trade.duckdb")
    config = _config(db_path)
    before = datetime.now(timezone.utc) - timedelta(minutes=2)
    baseline_at = before + timedelta(minutes=1)
    portfolio = Portfolio(config)
    portfolio.apply_fill(
        Fill(
            "legacy-paper-order",
            "legacy-intent",
            "QQQ",
            Side.BUY,
            14,
            700,
            before,
        )
    )
    baseline_id = portfolio.record_broker_baseline(
        positions=[],
        cash=100_000,
        observed_at=baseline_at,
        reason="LEGACY_PAPER_BROKER_RETIREMENT",
        evidence={"broker_type": "alpaca_paper"},
    )
    duplicate_id = portfolio.record_broker_baseline(
        positions=[],
        cash=100_000,
        observed_at=baseline_at,
        reason="LEGACY_PAPER_BROKER_RETIREMENT",
        evidence={"broker_type": "alpaca_paper"},
    )

    assert baseline_id == duplicate_id
    assert portfolio.positions == {}
    assert portfolio.cash == 100_000

    portfolio.apply_fill(
        Fill(
            "alpaca-order-after-baseline",
            "alpaca-intent",
            "AAPL",
            Side.BUY,
            2,
            100,
            baseline_at + timedelta(minutes=1),
        )
    )
    restarted = Portfolio(config)
    assert restarted.positions["AAPL"].qty == 2
    assert "QQQ" not in restarted.positions

    connection = duckdb.connect(db_path, read_only=True)
    try:
        assert connection.execute(
            "SELECT count(*) FROM fills"
        ).fetchone()[0] == 2
        assert connection.execute(
            "SELECT count(*) FROM portfolio_reconciliation_baselines"
        ).fetchone()[0] == 1
    finally:
        connection.close()


def test_startup_recovers_open_order_by_client_order_id(tmp_path):
    db_path = str(tmp_path / "trade.duckdb")
    config = _config(db_path)
    store, key = _intent_store_with_order(
        db_path,
        state=OrderLifecycle.SENDING,
    )
    client_id = store.get_by_key(key)["client_order_id"]

    broker = SimpleNamespace(
        get_open_orders=lambda: [
            {
                "id": "broker-open",
                "client_order_id": client_id,
                "status": "new",
            }
        ],
        get_positions=lambda: [],
        get_recent_fills=lambda: [],
    )
    runtime = _runtime(config, broker, store)

    runtime._run_reconciliation()

    row = store.get_by_key(key)
    assert not runtime._reconciliation_blocked
    assert "broker-open" in runtime._open_orders
    assert row["broker_order_id"] == "broker-open"
    assert row["state"] == OrderLifecycle.OPEN.value


def test_startup_applies_known_recent_fill_before_position_comparison(tmp_path):
    db_path = str(tmp_path / "trade.duckdb")
    config = _config(db_path)
    store, key = _intent_store_with_order(
        db_path,
        state=OrderLifecycle.OPEN,
        broker_order_id="broker-filled",
    )
    fill = Fill(
        "broker-filled",
        "",
        "AAPL",
        Side.BUY,
        5.0,
        100.0,
        datetime.now(timezone.utc),
    )
    broker = SimpleNamespace(
        get_open_orders=lambda: [],
        get_positions=lambda: [
            Position("AAPL", qty=5.0, avg_entry_px=100.0)
        ],
        get_recent_fills=lambda: [fill],
    )
    runtime = _runtime(config, broker, store)

    runtime._run_reconciliation()

    assert not runtime._reconciliation_blocked
    assert runtime._portfolio.positions["AAPL"].qty == 5.0
    assert store.get_by_key(key)["filled_qty"] == 5.0
    assert store.get_by_key(key)["remaining_qty"] == 0.0
    assert store.get_by_key(key)["state"] == OrderLifecycle.FILLED.value


def test_startup_blocks_unexplained_broker_position_and_audits_reason(tmp_path):
    db_path = str(tmp_path / "trade.duckdb")
    config = _config(db_path)
    store = OrderIntentStore(db_path)
    broker = SimpleNamespace(
        get_open_orders=lambda: [],
        get_positions=lambda: [
            Position("MSFT", qty=3.0, avg_entry_px=400.0)
        ],
        get_recent_fills=lambda: [],
    )
    runtime = _runtime(config, broker, store)

    runtime._run_reconciliation()

    assert runtime._reconciliation_blocked
    assert runtime._audit.reports[-1].unexplained_positions == ["MSFT"]


def test_startup_blocks_position_quantity_mismatch_and_audits_reason(tmp_path):
    db_path = str(tmp_path / "trade.duckdb")
    config = _config(db_path)
    store = OrderIntentStore(db_path)
    portfolio = Portfolio(config)
    portfolio.apply_fill(
        Fill(
            "broker-filled",
            "intent-1",
            "AAPL",
            Side.BUY,
            5.0,
            100.0,
            datetime.now(timezone.utc),
        )
    )
    broker = SimpleNamespace(
        get_open_orders=lambda: [],
        get_positions=lambda: [
            Position("AAPL", qty=4.0, avg_entry_px=100.0)
        ],
        get_recent_fills=lambda: [],
    )
    runtime = _runtime(config, broker, store, portfolio)

    runtime._run_reconciliation()

    assert runtime._reconciliation_blocked
    assert runtime._audit.reports[-1].unexplained_positions == [
        "AAPL:broker=4,local=5"
    ]


def test_startup_blocks_unexplained_recent_fill_and_audits_reason(tmp_path):
    db_path = str(tmp_path / "trade.duckdb")
    config = _config(db_path)
    store = OrderIntentStore(db_path)
    fill = Fill(
        "unknown-order",
        "",
        "AAPL",
        Side.BUY,
        1.0,
        100.0,
        datetime.now(timezone.utc),
    )
    broker = SimpleNamespace(
        get_open_orders=lambda: [],
        get_positions=lambda: [],
        get_recent_fills=lambda: [fill],
    )
    runtime = _runtime(config, broker, store)

    runtime._run_reconciliation()

    assert runtime._reconciliation_blocked
    assert runtime._audit.reports[-1].errors == [
        "UNEXPLAINED_FILL:unknown-order"
    ]


def test_startup_blocks_and_audits_broker_api_failure(tmp_path):
    db_path = str(tmp_path / "trade.duckdb")
    config = _config(db_path)
    store = OrderIntentStore(db_path)

    def fail():
        raise ConnectionError("broker unavailable")

    broker = SimpleNamespace(
        get_open_orders=fail,
        get_positions=lambda: [],
        get_recent_fills=lambda: [],
    )
    runtime = _runtime(config, broker, store)

    runtime._run_reconciliation()

    assert runtime._reconciliation_blocked
    assert runtime._audit.reports[-1].errors == ["ConnectionError"]
