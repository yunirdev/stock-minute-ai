from datetime import datetime, timezone

from trader.models import Fill, Side
from trader.order_lifecycle import (
    OrderIntentStore,
    OrderLifecycle,
    client_order_id,
    idempotency_key,
    reconcile_broker,
)


def test_idempotency_and_client_order_id_are_stable():
    key = idempotency_key("plan-1", "AAPL", "BUY", 2, 100, "OPEN")
    assert key == idempotency_key("plan-1", "AAPL", "BUY", 2, 100, "OPEN")
    assert client_order_id(key) == client_order_id(key)


def test_store_enforces_one_intent_per_key(tmp_path):
    store = OrderIntentStore(str(tmp_path / "trade.duckdb"))
    intent = type("Intent", (), {"intent_id": "i1", "signal_id": "p1", "symbol": "AAPL", "side": Side.BUY, "qty": 1.0, "limit_price": 100.0, "order_type": "LMT", "tif": "DAY"})()
    key = idempotency_key("p1", "AAPL", "BUY", 1, 100, "OPEN")
    first = store.persist(intent, key, "p1")
    intent2 = type("Intent", (), {"intent_id": "i2", "signal_id": "p1", "symbol": "AAPL", "side": Side.BUY, "qty": 1.0, "limit_price": 100.0, "order_type": "LMT", "tif": "DAY"})()
    second = store.persist(intent2, key, "p1")
    assert first["intent_id"] == second["intent_id"] == "i1"
    assert len(store.list_all()) == 1


def test_reconcile_blocks_unexplained_broker_order():
    class Broker:
        def get_open_orders(self):
            return [{"id": "b1", "client_order_id": "unknown"}]

        def get_positions(self):
            return []

        def get_recent_fills(self):
            return []

    report = reconcile_broker(Broker(), [])
    assert not report.ok
    assert report.unexplained_orders == ["unknown"]


def test_reconcile_success_with_known_order():
    class Broker:
        def get_open_orders(self):
            return [{"id": "b1", "client_order_id": "cid"}]

        def get_positions(self):
            return []

        def get_recent_fills(self):
            return []

    assert reconcile_broker(Broker(), [{"client_order_id": "cid"}]).ok


def test_lifecycle_has_unknown_and_partial_states():
    assert OrderLifecycle.UNKNOWN.value == "UNKNOWN"
    assert OrderLifecycle.PARTIALLY_FILLED.value == "PARTIALLY_FILLED"
    assert OrderLifecycle.CANCELED.value == "CANCELED"


def test_portfolio_fill_application_is_cumulative_and_idempotent(tmp_path):
    from trader.config import TradingConfig
    from trader.portfolio import Portfolio

    portfolio = Portfolio(TradingConfig(db_path=str(tmp_path / "trade.duckdb")))
    fill = Fill("b1", "i1", "AAPL", Side.BUY, 2, 100, datetime.now(timezone.utc))
    portfolio.apply_fill(fill)
    portfolio.apply_fill(Fill("b1", "i1", "AAPL", Side.BUY, 2, 100, datetime.now(timezone.utc)))
    assert portfolio.positions["AAPL"].qty == 2
