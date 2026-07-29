"""AlpacaBroker.get_account_equity must raise on failure, not return 0.0.

A swallowed exception here reads as "$0 equity" to RiskEngine.check_equity,
which computes a -100% daily drawdown and permanently trips the circuit
breaker (no auto-recovery) — this actually happened in production after a
single Alpaca API read timeout. get_positions/get_account_cash already
raise; this brings get_account_equity in line with them.
"""
from types import SimpleNamespace

import pytest

from trader.broker.alpaca import AlpacaBroker


def _broker(client) -> AlpacaBroker:
    broker = AlpacaBroker.__new__(AlpacaBroker)
    broker._client = client
    broker._paper = True
    return broker


def test_get_account_equity_returns_float_on_success():
    client = SimpleNamespace(get_account=lambda: SimpleNamespace(equity="100590.88"))
    broker = _broker(client)
    assert broker.get_account_equity() == pytest.approx(100590.88)


def test_get_account_equity_raises_on_failure_instead_of_returning_zero():
    def get_account():
        raise TimeoutError("Read timed out")

    client = SimpleNamespace(get_account=get_account)
    broker = _broker(client)
    with pytest.raises(TimeoutError):
        broker.get_account_equity()


def test_get_account_equity_matches_sibling_methods_error_behavior():
    """get_positions/get_account_cash/get_account_equity must all raise the
    same way on a broker failure, so Runtime's single except-and-skip
    handler in _tick() catches every one of them identically.
    """

    class _BrokenClient:
        def get_account(self):
            raise ConnectionError("boom")

        def get_all_positions(self):
            raise ConnectionError("boom")

    broker = _broker(_BrokenClient())
    for method_name in ("get_account_equity", "get_account_cash", "get_positions"):
        with pytest.raises(ConnectionError):
            getattr(broker, method_name)()


def test_get_fill_raises_on_query_failure_instead_of_returning_none():
    """A None return here reads as 'confirmed no fill' to _poll_orders,
    which could mark an actually-filled order done without ever applying
    it — a silent, permanent fill loss (worse than the equity bug: no
    circuit breaker fires, nothing visibly alerts).
    """

    def get_order_by_id(_id):
        raise TimeoutError("Read timed out")

    client = SimpleNamespace(get_order_by_id=get_order_by_id)
    broker = _broker(client)
    with pytest.raises(TimeoutError):
        broker.get_fill("order-1")


def test_get_fill_still_returns_none_for_genuinely_unfilled_order():
    """Distinguish the legitimate case (order exists, nothing filled yet)
    from a query failure — only the latter must raise.
    """
    order = SimpleNamespace(
        id="order-1", filled_qty="0", symbol="AAPL", side=SimpleNamespace(value="buy"),
        filled_avg_price=None, filled_at=None, status=SimpleNamespace(value="new"),
    )
    client = SimpleNamespace(get_order_by_id=lambda _id: order)
    broker = _broker(client)
    assert broker.get_fill("order-1") is None


def test_get_fill_returns_fill_on_success():
    order = SimpleNamespace(
        id="order-1", filled_qty="10", symbol="AAPL", side=SimpleNamespace(value="buy"),
        filled_avg_price="150.5", filled_at=None,
        status=SimpleNamespace(value="filled"),
    )
    client = SimpleNamespace(get_order_by_id=lambda _id: order)
    broker = _broker(client)
    fill = broker.get_fill("order-1")
    assert fill is not None
    assert fill.filled_qty == 10.0
    assert fill.avg_price == 150.5


def test_get_order_status_raises_on_failure_instead_of_returning_failed():
    """A silent OrderStatus.FAILED return is a landmine: a future _poll_orders
    branch treating FAILED as a real rejection would wipe live plan state on
    a transient network error. Match get_positions/get_account_cash/get_fill.
    """

    def get_order_by_id(_id):
        raise ConnectionError("boom")

    client = SimpleNamespace(get_order_by_id=get_order_by_id)
    broker = _broker(client)
    with pytest.raises(ConnectionError):
        broker.get_order_status("order-1")


def test_cancel_order_raises_on_failure_instead_of_returning_false():
    """A bare False is indistinguishable from 'broker confirmed not
    cancelled' — a caller retrying on False could double-submit if the
    cancel had actually gone through server-side before the response failed.
    """

    def cancel_order_by_id(_id):
        raise TimeoutError("Read timed out")

    client = SimpleNamespace(cancel_order_by_id=cancel_order_by_id)
    broker = _broker(client)
    with pytest.raises(TimeoutError):
        broker.cancel_order("order-1")


def test_cancel_order_returns_true_on_success():
    client = SimpleNamespace(cancel_order_by_id=lambda _id: None)
    broker = _broker(client)
    assert broker.cancel_order("order-1") is True
