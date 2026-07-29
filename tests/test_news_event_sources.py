from datetime import datetime, timedelta, timezone

from trader.data_hub import DataDomain, DataHub, DataStatus, SourceRegistry
from trader.news_event_sources import register_news_event_sources

NOW = datetime(2026, 7, 26, 18, tzinfo=timezone.utc)


def _news(
    *,
    title="Apple launches product",
    published_at=NOW - timedelta(hours=1),
    symbol="AAPL",
    source_id="item-1",
):
    return {
        "id": source_id,
        "kind": "news",
        "symbol": symbol,
        "title": title,
        "summary": "A sourced report.",
        "published_at": published_at,
        "url": "https://example.com/story?tracking=1",
    }


def _earnings(*, source_time="2026-07-28T12:00:00-04:00", title="AAPL Earnings"):
    return {
        "id": f"earnings-{source_time}",
        "kind": "calendar",
        "category": "earnings",
        "symbol": "AAPL",
        "title": title,
        "event_at": source_time,
        "as_of": NOW,
    }


def _register(registry, **overrides):
    def empty(_request):
        return []

    loaders = {
        "finnhub_loader": empty,
        "nasdaq_loader": empty,
        "wallstreetcn_loader": empty,
        "yahoo_loader": empty,
        "rss_loader": empty,
        **overrides,
    }
    register_news_event_sources(registry, **loaders)


def test_cross_source_duplicates_merge_provenance_and_remain_research_only():
    registry = SourceRegistry()
    _register(
        registry,
        finnhub_loader=lambda _: [_news()],
        yahoo_loader=lambda _: [
            _news(
                title="APPLE launches product!",
                source_id="yahoo-1",
            )
        ],
    )
    hub = DataHub(registry, clock=lambda: NOW)
    try:
        result = hub.fetch(DataDomain.NEWS, "AAPL")
    finally:
        hub.close()

    assert result.status == DataStatus.OK
    assert len(result.payload["items"]) == 1
    item = result.payload["items"][0]
    assert item["source"] == "finnhub"
    assert item["sources"] == ["finnhub", "yahoo_news"]
    assert item["execution_eligible"] is False
    assert result.payload["directive_capability"] == "RESEARCH_ONLY"
    assert result.metadata["deduplicated_count"] == 1


def test_time_window_is_inclusive_and_excludes_news_outside_boundary():
    since = NOW - timedelta(hours=2)
    until = NOW
    registry = SourceRegistry()
    _register(
        registry,
        rss_loader=lambda _: [
            _news(title="at start", published_at=since, source_id="start"),
            _news(title="at end", published_at=until, source_id="end"),
            _news(
                title="too old",
                published_at=since - timedelta(microseconds=1),
                source_id="old",
            ),
            _news(
                title="too new",
                published_at=until + timedelta(microseconds=1),
                source_id="new",
            ),
        ],
    )
    hub = DataHub(registry, clock=lambda: NOW)
    try:
        result = hub.fetch(
            DataDomain.NEWS,
            "AAPL",
            params={"since": since, "until": until},
        )
    finally:
        hub.close()

    assert [item["title"] for item in result.payload["items"]] == [
        "at start",
        "at end",
    ]
    assert result.payload["window"] == {
        "since": since.isoformat(),
        "until": until.isoformat(),
    }


def test_yahoo_nested_content_and_rss_dates_normalize_without_llm_fill():
    registry = SourceRegistry()
    _register(
        registry,
        yahoo_loader=lambda _: [
            {
                "id": "yf-1",
                "content": {
                    "title": "Yahoo sourced headline",
                    "summary": "Yahoo source summary",
                    "pubDate": "2026-07-26T17:00:00Z",
                    "canonicalUrl": {
                        "url": "https://finance.yahoo.com/news/story?guccounter=1"
                    },
                },
            }
        ],
        rss_loader=lambda _: [
            {
                "id": "rss-1",
                "title": "RSS sourced headline",
                "summary": "RSS source summary",
                "published": "Sun, 26 Jul 2026 16:00:00 GMT",
                "link": "https://example.com/rss/story",
            }
        ],
    )
    hub = DataHub(registry, clock=lambda: NOW)
    try:
        result = hub.fetch(DataDomain.NEWS, "AAPL")
    finally:
        hub.close()

    assert [item["source"] for item in result.payload["items"]] == [
        "rss_news",
        "yahoo_news",
    ]
    assert result.payload["items"][1]["url"] == (
        "https://finance.yahoo.com/news/story"
    )
    assert result.payload["directive_capability"] == "RESEARCH_ONLY"


def test_calendar_conflict_uses_source_priority_and_is_auditable():
    registry = SourceRegistry()
    _register(
        registry,
        nasdaq_loader=lambda _: [
            _earnings(
                source_time="2026-07-28T08:00:00-04:00",
                title="AAPL Earnings Calendar",
            )
        ],
        finnhub_loader=lambda _: [
            _earnings(
                source_time="2026-07-28T16:30:00-04:00",
            )
        ],
    )
    hub = DataHub(registry, clock=lambda: NOW)
    try:
        result = hub.fetch(DataDomain.NEWS, "AAPL")
    finally:
        hub.close()

    assert result.status == DataStatus.DEGRADED
    assert result.failure_code == "SOURCE_QUALITY_DEGRADED"
    assert len(result.payload["items"]) == 1
    assert result.payload["items"][0]["source"] == "nasdaq_calendar"
    assert result.payload["items"][0]["sources"] == [
        "nasdaq_calendar",
        "finnhub",
    ]
    assert result.payload["conflicts"][0]["selected_source"] == (
        "nasdaq_calendar"
    )
    assert result.payload["conflicts"][0]["differences"][0]["field"] == (
        "event_at"
    )


def test_partial_source_failure_degrades_without_hiding_available_news():
    def unavailable(_request):
        raise ConnectionError("offline")

    registry = SourceRegistry()
    _register(
        registry,
        finnhub_loader=unavailable,
        wallstreetcn_loader=lambda _: [_news(title="available")],
    )
    hub = DataHub(registry, clock=lambda: NOW)
    try:
        result = hub.fetch(DataDomain.NEWS, "AAPL")
    finally:
        hub.close()

    assert result.status == DataStatus.DEGRADED
    assert result.quality_score == 0.8
    assert result.payload["items"][0]["source"] == "wallstreetcn"
    statuses = {
        item["source_id"]: item for item in result.payload["source_statuses"]
    }
    assert statuses["finnhub"]["status"] == "FAILED"
    assert statuses["finnhub"]["failure_code"] == "ConnectionError"


def test_all_source_failures_are_explicit():
    def unavailable(_request):
        raise TimeoutError("offline")

    registry = SourceRegistry()
    _register(
        registry,
        finnhub_loader=unavailable,
        nasdaq_loader=unavailable,
        wallstreetcn_loader=unavailable,
        yahoo_loader=unavailable,
        rss_loader=unavailable,
    )
    hub = DataHub(registry, clock=lambda: NOW)
    try:
        result = hub.fetch(DataDomain.NEWS, "AAPL")
    finally:
        hub.close()

    assert result.status == DataStatus.FAILED
    assert result.failures[0].code == "DATA_NEWS_ALL_SOURCES_FAILED"
