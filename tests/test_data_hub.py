import time
from datetime import datetime, timedelta, timezone

import pytest

from trader.data_hub import (
    AdapterResult,
    DataDomain,
    DataHub,
    DataStatus,
    SourceRegistry,
    SourceSpec,
)


NOW = datetime(2026, 7, 26, 16, tzinfo=timezone.utc)


class _Clock:
    def __init__(self):
        self.value = NOW

    def __call__(self):
        return self.value


def _spec(
    source_id,
    adapter,
    *,
    priority=0,
    timeout=0.1,
    ttl=60,
    max_stale=0,
):
    return SourceSpec(
        source_id=source_id,
        domain=DataDomain.MARKET,
        adapter=adapter,
        priority=priority,
        timeout_seconds=timeout,
        ttl_seconds=ttl,
        max_stale_seconds=max_stale,
        required_fields=("price",),
    )


def test_registry_orders_sources_and_rejects_duplicate():
    registry = SourceRegistry()
    registry.register(_spec("fallback", lambda _: None, priority=10))
    registry.register(_spec("primary", lambda _: None, priority=0))

    assert [
        source.source_id
        for source in registry.sources_for(DataDomain.MARKET)
    ] == ["primary", "fallback"]
    with pytest.raises(ValueError, match="DATA_SOURCE_DUPLICATE"):
        registry.register(_spec("primary", lambda _: None))


def test_fresh_cache_uses_ttl_and_refreshes_after_expiry():
    clock = _Clock()
    calls = []

    def adapter(_request):
        calls.append(clock.value)
        return AdapterResult({"price": 100 + len(calls)}, clock.value)

    registry = SourceRegistry()
    registry.register(_spec("primary", adapter, ttl=30))
    hub = DataHub(registry, clock=clock)
    try:
        first = hub.fetch(DataDomain.MARKET, "aapl")
        clock.value += timedelta(seconds=20)
        cached = hub.fetch(DataDomain.MARKET, "AAPL")
        clock.value += timedelta(seconds=11)
        refreshed = hub.fetch(DataDomain.MARKET, "AAPL")
    finally:
        hub.close()

    assert first.status == DataStatus.OK
    assert not first.cache_hit
    assert cached.cache_hit
    assert cached.payload == first.payload
    assert not refreshed.cache_hit
    assert refreshed.payload["price"] == 102
    assert len(calls) == 2


def test_timeout_uses_explicit_lower_priority_fallback():
    def timeout(_request):
        time.sleep(0.05)
        return AdapterResult({"price": 999}, NOW)

    registry = SourceRegistry()
    registry.register(_spec("primary", timeout, timeout=0.005))
    registry.register(
        _spec(
            "fallback",
            lambda _: AdapterResult({"price": 101}, NOW, 0.8),
            priority=10,
        )
    )
    hub = DataHub(registry, clock=lambda: NOW)
    try:
        result = hub.fetch(DataDomain.MARKET, "AAPL")
    finally:
        hub.close()

    assert result.status == DataStatus.DEGRADED
    assert result.source_id == "fallback"
    assert result.fallback_from == "primary"
    assert result.failure_code == "FALLBACK_SOURCE"
    assert result.failures[0].code == "SOURCE_TIMEOUT"


def test_invalid_primary_quality_falls_back_and_all_fail_is_visible():
    registry = SourceRegistry()
    registry.register(
        _spec(
            "primary",
            lambda _: AdapterResult({"wrong": 1}, NOW),
        )
    )
    registry.register(
        _spec(
            "fallback",
            lambda _: AdapterResult({"price": 100}, NOW + timedelta(seconds=1)),
            priority=10,
        )
    )
    hub = DataHub(registry, clock=lambda: NOW)
    try:
        result = hub.fetch(DataDomain.MARKET, "AAPL")
        missing = hub.fetch(DataDomain.NEWS, "AAPL")
    finally:
        hub.close()

    assert result.status == DataStatus.FAILED
    assert [failure.code for failure in result.failures] == [
        "DATA_SOURCE_REQUIRED_FIELDS_MISSING",
        "DATA_SOURCE_AS_OF_FUTURE",
    ]
    assert missing.status == DataStatus.FAILED
    assert missing.failures[0].code == "NO_REGISTERED_SOURCE"


def test_stale_cache_is_only_returned_as_explicit_degradation():
    clock = _Clock()
    fail = {"value": False}

    def adapter(_request):
        if fail["value"]:
            raise ConnectionError("offline")
        return AdapterResult({"price": 100}, clock.value)

    registry = SourceRegistry()
    registry.register(
        _spec(
            "primary",
            adapter,
            ttl=10,
            max_stale=60,
        )
    )
    hub = DataHub(registry, clock=clock)
    try:
        hub.fetch(DataDomain.MARKET, "AAPL")
        fail["value"] = True
        clock.value += timedelta(seconds=20)
        stale = hub.fetch(DataDomain.MARKET, "AAPL")
    finally:
        hub.close()

    assert stale.status == DataStatus.DEGRADED
    assert stale.cache_hit
    assert stale.failure_code == "STALE_CACHE_FALLBACK"
    assert stale.quality_score == 0.5
    assert stale.failures[0].code == "ConnectionError"
