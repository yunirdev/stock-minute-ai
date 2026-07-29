from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd

from trader.data_hub import (
    DataDomain,
    DataEnvelope,
    DataHub,
    DataStatus,
    SourceRegistry,
)
from trader.data_hub_sources import (
    compare_market_envelopes,
    register_alpaca_broker_facts,
    register_market_sources,
)
from trader.models import Fill, Position, Side


NOW = datetime(2026, 7, 26, 16, tzinfo=timezone.utc)


def _bars(price=100.0, count=2):
    return pd.DataFrame(
        {
            "timestamp_utc": pd.date_range(
                NOW,
                periods=count,
                freq="5min",
                tz="UTC",
            ),
            "open": [price] * count,
            "high": [price + 1] * count,
            "low": [price - 1] * count,
            "close": [price] * count,
            "volume": [1_000] * count,
        }
    )


def test_alpaca_market_is_primary_and_execution_eligible():
    registry = SourceRegistry()
    register_market_sources(
        registry,
        alpaca_loader=lambda *_: _bars(),
        local_cache_loader=lambda *_: _bars(99),
        yahoo_loader=lambda *_: _bars(98),
    )
    hub = DataHub(registry, clock=lambda: NOW.replace(minute=10))
    try:
        result = hub.fetch(
            DataDomain.MARKET,
            "AAPL",
            params={"timeframe": "5m"},
        )
    finally:
        hub.close()

    assert result.status == DataStatus.OK
    assert result.source_id == "alpaca_market"
    assert result.payload["execution_eligible"] is True
    assert result.metadata["upstream"] == "alpaca"


def test_local_and_yahoo_fallbacks_are_explicitly_not_execution_prices():
    registry = SourceRegistry()
    register_market_sources(
        registry,
        alpaca_loader=lambda *_: [],
        local_cache_loader=lambda *_: _bars(99),
        yahoo_loader=lambda *_: _bars(98),
    )
    hub = DataHub(registry, clock=lambda: NOW.replace(minute=10))
    try:
        local = hub.fetch(DataDomain.MARKET, "AAPL")
    finally:
        hub.close()
    assert local.status == DataStatus.DEGRADED
    assert local.source_id == "local_bar_cache"
    assert local.payload["execution_eligible"] is False
    assert local.failures[0].code == "DATA_MARKET_BARS_EMPTY"

    registry = SourceRegistry()
    register_market_sources(
        registry,
        alpaca_loader=lambda *_: [],
        local_cache_loader=lambda *_: [],
        yahoo_loader=lambda *_: _bars(98),
    )
    hub = DataHub(registry, clock=lambda: NOW.replace(minute=10))
    try:
        yahoo = hub.fetch(DataDomain.MARKET, "AAPL")
    finally:
        hub.close()
    assert yahoo.status == DataStatus.DEGRADED
    assert yahoo.source_id == "yahoo_market_fallback"
    assert yahoo.payload["execution_eligible"] is False
    assert yahoo.quality_score == 0.6


def test_alpaca_broker_facts_are_one_authoritative_envelope():
    fill = Fill(
        "order-1",
        "intent-1",
        "AAPL",
        Side.BUY,
        1.0,
        100.0,
        NOW,
    )
    broker = SimpleNamespace(
        get_account_equity=lambda: 100_000.0,
        get_positions=lambda: [Position("AAPL", 1.0, 100.0)],
        get_open_orders=lambda: [{"id": "order-2"}],
        get_recent_fills=lambda: [fill],
    )
    registry = SourceRegistry()
    register_alpaca_broker_facts(registry, broker)
    hub = DataHub(registry)
    try:
        result = hub.fetch(DataDomain.BROKER, "ACCOUNT")
    finally:
        hub.close()

    assert result.status == DataStatus.OK
    assert result.source_id == "alpaca_broker_facts"
    assert result.payload["equity"] == 100_000.0
    assert result.payload["positions"][0]["symbol"] == "AAPL"
    assert result.payload["recent_fills"][0]["order_id"] == "order-1"
    assert result.metadata["authoritative"] is True


def _envelope(source, price, bars):
    return DataEnvelope(
        request_id="request",
        domain=DataDomain.MARKET,
        key="AAPL",
        source_id=source,
        status=DataStatus.OK,
        payload={"last_price": price, "bars": [None] * bars},
        as_of=NOW,
        fetched_at=NOW,
        expires_at=NOW,
        quality_score=1.0,
    )


def test_market_double_read_classifies_price_and_coverage_difference():
    comparison = compare_market_envelopes(
        _envelope("alpaca", 100.0, 10),
        _envelope("yahoo", 99.0, 8),
        price_tolerance_bps=5.0,
    )

    assert comparison["comparable"]
    assert comparison["classification"] == "DIFFERENT"
    assert [item["field"] for item in comparison["differences"]] == [
        "last_price",
        "bar_count",
    ]
