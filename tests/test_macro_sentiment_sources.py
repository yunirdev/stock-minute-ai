from datetime import datetime, timedelta, timezone

import pytest

from trader.data_hub import DataDomain, DataHub, DataStatus, SourceRegistry
from trader.macro_sentiment_sources import (
    FredSeriesSpec,
    register_macro_sentiment_sources,
)

NOW = datetime(2026, 7, 26, 18, tzinfo=timezone.utc)
SERIES = (
    FredSeriesSpec("DGS10", 7 * 86_400),
    FredSeriesSpec("UNRATE", 45 * 86_400),
)


def _fred(*, dgs_as_of="2026-07-25", include_unrate=True):
    payload = {
        "DGS10": [
            {
                "date": "2026-07-25",
                "value": "4.25",
                "as_of": dgs_as_of,
            }
        ]
    }
    if include_unrate:
        payload["UNRATE"] = [
            {
                "date": "2026-07-01",
                "value": "4.1",
                "as_of": "2026-07-10",
            }
        ]
    return payload


def _stocktwits():
    return [
        {
            "id": "st-1",
            "symbol": "AAPL",
            "body": "Structured source sentiment",
            "created_at": NOW - timedelta(hours=1),
            "entities": {"sentiment": {"basic": "Bullish"}},
            "likes": {"total": 4},
        }
    ]


def _reddit():
    return [
        {
            "id": "rd-1",
            "title": "Unlabelled discussion",
            "created_utc": (NOW - timedelta(hours=2)).timestamp(),
            "score": 120,
            "num_comments": 30,
            "subreddit": "stocks",
        }
    ]


def _polymarket():
    return [
        {
            "id": "pm-1",
            "question": "Will the Fed cut rates?",
            "updatedAt": (NOW - timedelta(hours=3)).isoformat(),
            "outcomes": '["Yes", "No"]',
            "outcomePrices": '["0.62", "0.38"]',
            "liquidity": 50_000,
            "volume": 200_000,
        }
    ]


def _register(registry, **overrides):
    defaults = {
        "fred_loader": lambda _: _fred(),
        "stocktwits_loader": lambda _: _stocktwits(),
        "reddit_loader": lambda _: _reddit(),
        "polymarket_loader": lambda _: _polymarket(),
        "fred_series_specs": SERIES,
        **overrides,
    }
    register_macro_sentiment_sources(registry, **defaults)


def test_fred_fresh_complete_series_are_authoritative_research_facts():
    registry = SourceRegistry()
    _register(registry)
    hub = DataHub(registry, clock=lambda: NOW)
    try:
        result = hub.fetch(DataDomain.MACRO, "US")
    finally:
        hub.close()

    assert result.status == DataStatus.OK
    assert result.source_id == "fred_macro"
    assert result.payload["coverage"] == 1.0
    assert result.payload["missing_series"] == []
    assert result.payload["stale_series"] == []
    assert [item["series_id"] for item in result.payload["observations"]] == [
        "DGS10",
        "UNRATE",
    ]
    assert all(
        item["source"] == "fred"
        and item["freshness"] == "FRESH"
        for item in result.payload["observations"]
    )
    assert result.payload["broker_fact_eligible"] is False
    assert result.payload["execution_eligible"] is False


def test_fred_staleness_and_missing_series_reduce_quality_explicitly():
    registry = SourceRegistry()
    _register(
        registry,
        fred_loader=lambda _: _fred(
            dgs_as_of="2026-07-01",
            include_unrate=False,
        ),
    )
    hub = DataHub(registry, clock=lambda: NOW)
    try:
        result = hub.fetch(DataDomain.MACRO, "US")
    finally:
        hub.close()

    assert result.status == DataStatus.DEGRADED
    assert result.failure_code == "SOURCE_QUALITY_DEGRADED"
    assert result.payload["coverage"] == 0.5
    assert result.payload["missing_series"] == ["UNRATE"]
    assert result.payload["stale_series"] == ["DGS10"]
    assert result.quality_score == pytest.approx(0.2)
    assert result.metadata["low_quality"] is True


def test_social_and_prediction_signals_mark_unlabelled_reddit_low_quality():
    registry = SourceRegistry()
    _register(registry)
    hub = DataHub(registry, clock=lambda: NOW)
    try:
        result = hub.fetch(DataDomain.SENTIMENT, "AAPL")
    finally:
        hub.close()

    assert result.status == DataStatus.DEGRADED
    assert result.payload["directive_capability"] == "RESEARCH_ONLY"
    assert result.payload["broker_fact_eligible"] is False
    assert result.payload["execution_eligible"] is False
    by_source = {
        item["source"]: item for item in result.payload["signals"]
    }
    assert by_source["stocktwits"]["sentiment_score"] == 1.0
    assert by_source["reddit"]["sentiment_score"] is None
    assert by_source["reddit"]["quality_label"] == "LOW"
    assert by_source["polymarket"]["probability"] == pytest.approx(0.62)
    assert "reddit" in result.payload["low_quality_sources"]


def test_social_staleness_coverage_and_partial_failure_are_visible():
    def unavailable(_request):
        raise ConnectionError("offline")

    registry = SourceRegistry()
    _register(
        registry,
        stocktwits_loader=unavailable,
        reddit_loader=lambda _: [
            {
                **_reddit()[0],
                "created_utc": (NOW - timedelta(days=2)).timestamp(),
            }
        ],
        polymarket_loader=lambda _: [],
    )
    hub = DataHub(registry, clock=lambda: NOW)
    try:
        result = hub.fetch(DataDomain.SENTIMENT, "AAPL")
    finally:
        hub.close()

    assert result.status == DataStatus.DEGRADED
    statuses = {
        item["source_id"]: item for item in result.payload["source_statuses"]
    }
    assert statuses["stocktwits"]["status"] == "FAILED"
    assert statuses["stocktwits"]["failure_code"] == "ConnectionError"
    assert statuses["reddit"]["status"] == "STALE"
    assert statuses["reddit"]["fresh_count"] == 0
    assert statuses["polymarket"]["status"] == "LOW_COVERAGE"
    assert result.payload["coverage"]["polymarket"] == 0.0


def test_all_social_source_failures_fail_explicitly():
    def unavailable(_request):
        raise TimeoutError("offline")

    registry = SourceRegistry()
    _register(
        registry,
        stocktwits_loader=unavailable,
        reddit_loader=unavailable,
        polymarket_loader=unavailable,
    )
    hub = DataHub(registry, clock=lambda: NOW)
    try:
        result = hub.fetch(DataDomain.SENTIMENT, "AAPL")
    finally:
        hub.close()

    assert result.status == DataStatus.FAILED
    assert result.failures[0].code == "DATA_SENTIMENT_ALL_SOURCES_FAILED"
