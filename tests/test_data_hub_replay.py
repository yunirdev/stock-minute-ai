from datetime import datetime, timedelta, timezone

import pandas as pd

from trader.data_cache import fetch_alpaca_bars_window
from trader.data_hub_quality import DataHubQualityStore, SourceReadMetrics
from trader.data_hub_replay import (
    DataHubReplayStore,
    HistoricalDataHubReplayRunner,
)
from trader.data_hub_sources import market_adapter
from trader.data_hub import DataDomain, DataHub, SourceRegistry, SourceSpec
from trader.data_hub_quality import observe_double_read

NOW = datetime(2026, 7, 26, 20, tzinfo=timezone.utc)


def _daily_frame(symbol, *, changed_date="", change=0.0):
    dates = pd.bdate_range(
        "2026-06-22",
        periods=25,
        tz="America/New_York",
    )
    rows = []
    for index, timestamp in enumerate(dates):
        trading_date = timestamp.date().isoformat()
        close = 100.0 + index
        if trading_date == changed_date:
            close += change
        rows.append(
            {
                "symbol": symbol,
                "timestamp_utc": timestamp.tz_convert("UTC"),
                "open": close - 0.5,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": 1_000.0 + index,
            }
        )
    return pd.DataFrame(rows)


def _live_observation(store):
    raw = _daily_frame("AAPL").tail(2)
    registry = SourceRegistry()
    registry.register(
        SourceSpec(
            source_id="source",
            domain=DataDomain.MARKET,
            adapter=market_adapter(
                lambda *_: raw,
                upstream="alpaca",
                execution_eligible=True,
                quality_score=1.0,
            ),
            priority=0,
            timeout_seconds=1.0,
            ttl_seconds=0.0,
        )
    )
    hub = DataHub(registry, clock=lambda: NOW)
    try:
        primary = hub.fetch(DataDomain.MARKET, "AAPL")
        shadow = hub.fetch(DataDomain.MARKET, "AAPL")
    finally:
        hub.close()
    observation = observe_double_read(
        primary,
        shadow,
        observed_at=NOW,
        primary_metrics=SourceReadMetrics("primary", 10.0),
        shadow_metrics=SourceReadMetrics("shadow", 20.0),
        trading_date="2026-07-24",
    )
    store.save_observation(observation)


def test_twenty_day_replay_accelerates_shadow_delivery_not_cutover(tmp_path):
    quality_store = DataHubQualityStore(tmp_path / "quality.duckdb")
    _live_observation(quality_store)
    frames = {
        symbol: _daily_frame(symbol)
        for symbol in ("AAPL", "MSFT")
    }
    runner = HistoricalDataHubReplayRunner(
        store=DataHubReplayStore(quality_store.db_path),
        local_loader=lambda symbol, _timeframe: frames[symbol],
        alpaca_loader=lambda symbol, _timeframe, _start, _end: frames[
            symbol
        ],
        clock=lambda: NOW,
    )

    result = runner.run(["AAPL", "MSFT"])

    assert result.report["passed"]
    assert result.report["observed_days"] == 20
    assert result.report["comparisons"] == 40
    assert result.report["differences"] == 0
    assert result.report["accelerated_shadow_delivery_ready"]
    assert not result.report["execution_cutover_ready"]
    assert not result.report["can_replace_live_observation_window"]
    assert not result.report["execution_input_switched"]
    assert result.saved_observations == 40


def test_replay_rejects_historical_value_difference(tmp_path):
    quality_store = DataHubQualityStore(tmp_path / "quality.duckdb")
    _live_observation(quality_store)
    local = {
        symbol: _daily_frame(symbol)
        for symbol in ("AAPL", "MSFT")
    }
    changed_date = local["AAPL"].iloc[-1]["timestamp_utc"].tz_convert(
        "America/New_York"
    ).date().isoformat()
    remote = {
        "AAPL": _daily_frame(
            "AAPL",
            changed_date=changed_date,
            change=1.0,
        ),
        "MSFT": local["MSFT"],
    }
    result = HistoricalDataHubReplayRunner(
        store=DataHubReplayStore(quality_store.db_path),
        local_loader=lambda symbol, _timeframe: local[symbol],
        alpaca_loader=lambda symbol, _timeframe, _start, _end: remote[
            symbol
        ],
        clock=lambda: NOW,
    ).run(["AAPL", "MSFT"])

    assert not result.report["passed"]
    assert result.report["mismatched_comparisons"] == 1
    assert result.report["differences"] == 4
    assert not result.report["accelerated_shadow_delivery_ready"]


def test_explicit_alpaca_window_validation(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "trader.data_cache._alpaca_fetch_bars",
        lambda *args: calls.append(args) or pd.DataFrame(),
    )

    fetch_alpaca_bars_window(
        "aapl",
        "1d",
        NOW - timedelta(days=2),
        NOW,
    )

    assert calls[0][0:2] == ("AAPL", "1d")
