from datetime import datetime, timezone

import pytest

from trader.corporate_data_sources import (
    SecEdgarClient,
    SlidingWindowRateLimiter,
    register_sec_edgar_corporate,
    sec_edgar_corporate_adapter,
)
from trader.data_hub import DataDomain, DataHub, DataStatus, SourceRegistry

NOW = datetime(2026, 7, 26, 18, tzinfo=timezone.utc)


def _bundle(*, facts=True, disclosure=True, insider=True, revision=False):
    fact_entries = []
    if facts:
        fact_entries.append(
            {
                "start": "2026-01-01",
                "end": "2026-03-31",
                "val": 100,
                "accn": "0000320193-26-000001",
                "fy": 2026,
                "fp": "Q1",
                "form": "10-Q",
                "filed": "2026-04-25",
            }
        )
        if revision:
            fact_entries.append(
                {
                    "start": "2026-01-01",
                    "end": "2026-03-31",
                    "val": 105,
                    "accn": "0000320193-26-000002",
                    "fy": 2026,
                    "fp": "Q1",
                    "form": "10-Q/A",
                    "filed": "2026-04-27",
                }
            )
    forms = []
    accessions = []
    accepted = []
    filing_dates = []
    reports = []
    items = []
    documents = []
    if disclosure:
        forms.append("8-K")
        accessions.append("0000320193-26-000003")
        accepted.append("20260428123000")
        filing_dates.append("2026-04-28")
        reports.append("2026-04-28")
        items.append("2.02")
        documents.append("event.htm")
    if insider:
        forms.append("4/A")
        accessions.append("0000320193-26-000004")
        accepted.append("20260429120000")
        filing_dates.append("2026-04-29")
        reports.append("2026-04-28")
        items.append("")
        documents.append("ownership.xml")
    return {
        "companyfacts": {
            "entityName": "Apple Inc.",
            "facts": {
                "us-gaap": {
                    "RevenueFromContractWithCustomerExcludingAssessedTax": {
                        "label": "Revenue",
                        "description": "Revenue for the period.",
                        "units": {"USD": fact_entries},
                    }
                }
            },
        },
        "submissions": {
            "name": "Apple Inc.",
            "filings": {
                "recent": {
                    "accessionNumber": accessions,
                    "form": forms,
                    "acceptanceDateTime": accepted,
                    "filingDate": filing_dates,
                    "reportDate": reports,
                    "items": items,
                    "primaryDocument": documents,
                }
            },
        },
    }


def test_sec_source_and_as_of_are_preserved_for_each_fact_section():
    registry = SourceRegistry()
    client = SecEdgarClient(
        user_agent="stock-minute-ai test@example.com",
        json_loader=lambda *_: {},
    )
    client.fetch_bundle = lambda *_: _bundle()
    register_sec_edgar_corporate(
        registry,
        client=client,
        cik_resolver=lambda symbol: "320193" if symbol == "AAPL" else None,
    )
    hub = DataHub(registry, clock=lambda: NOW)
    try:
        result = hub.fetch(DataDomain.CORPORATE, "aapl")
    finally:
        hub.close()

    assert result.status == DataStatus.OK
    assert result.source_id == "sec_edgar_corporate"
    assert result.as_of == datetime(
        2026,
        4,
        29,
        12,
        tzinfo=timezone.utc,
    )
    assert result.payload["financial_facts"][0]["source"] == (
        "sec_edgar_companyfacts"
    )
    assert result.payload["disclosures"][0]["source"] == (
        "sec_edgar_submissions"
    )
    assert result.payload["insider_filings"][0]["form"] == "4/A"
    assert result.payload["fact_generation"] == "SOURCE_ONLY"
    assert result.metadata["llm_fact_fill"] is False


def test_financial_revision_selects_latest_and_keeps_full_history():
    adapter = sec_edgar_corporate_adapter(
        lambda *_: _bundle(revision=True)
    )
    registry = SourceRegistry()
    client = SecEdgarClient(
        user_agent="stock-minute-ai test@example.com",
        json_loader=lambda *_: {},
    )
    client.fetch_bundle = lambda *_: _bundle(revision=True)
    register_sec_edgar_corporate(registry, client=client)
    hub = DataHub(registry, clock=lambda: NOW)
    try:
        result = hub.fetch(
            DataDomain.CORPORATE,
            "AAPL",
            params={"cik": "0000320193"},
        )
    finally:
        hub.close()

    fact = result.payload["financial_facts"][0]
    assert fact["value"] == 105
    assert fact["form"] == "10-Q/A"
    assert fact["is_amendment"] is True
    assert fact["revision"] == 1
    assert [item["value"] for item in fact["revisions"]] == [100, 105]
    assert adapter is not None


def test_missing_sections_degrade_and_all_missing_fails_explicitly():
    registry = SourceRegistry()
    client = SecEdgarClient(
        user_agent="stock-minute-ai test@example.com",
        json_loader=lambda *_: {},
    )
    client.fetch_bundle = lambda *_: _bundle(
        facts=False,
        disclosure=True,
        insider=False,
    )
    register_sec_edgar_corporate(registry, client=client)
    hub = DataHub(registry, clock=lambda: NOW)
    try:
        degraded = hub.fetch(
            DataDomain.CORPORATE,
            "AAPL",
            params={"cik": "320193"},
        )
    finally:
        hub.close()

    assert degraded.status == DataStatus.DEGRADED
    assert degraded.failure_code == "SOURCE_QUALITY_DEGRADED"
    assert degraded.quality_score == pytest.approx(0.6)
    assert degraded.payload["missing_sections"] == [
        "financial_facts",
        "insider_filings",
    ]

    registry = SourceRegistry()
    client.fetch_bundle = lambda *_: _bundle(
        facts=False,
        disclosure=False,
        insider=False,
    )
    register_sec_edgar_corporate(registry, client=client)
    hub = DataHub(registry, clock=lambda: NOW)
    try:
        failed = hub.fetch(
            DataDomain.CORPORATE,
            "AAPL",
            params={"cik": "320193"},
        )
    finally:
        hub.close()

    assert failed.status == DataStatus.FAILED
    assert failed.failures[0].code == "DATA_CORPORATE_FACTS_EMPTY"


def test_sec_rate_limit_is_fail_fast_and_recovers_after_window():
    clock = {"value": 0.0}
    limiter = SlidingWindowRateLimiter(
        max_requests=2,
        window_seconds=1.0,
        clock=lambda: clock["value"],
    )

    limiter.acquire()
    limiter.acquire()
    with pytest.raises(ValueError, match="DATA_SOURCE_RATE_LIMITED"):
        limiter.acquire()

    clock["value"] = 1.0
    limiter.acquire()


def test_sec_client_requires_identity_and_limits_each_http_request(monkeypatch):
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    with pytest.raises(ValueError, match="DATA_SEC_USER_AGENT_REQUIRED"):
        SecEdgarClient()

    calls = []
    client = SecEdgarClient(
        user_agent="stock-minute-ai test@example.com",
        json_loader=lambda url, *_: calls.append(url) or {},
    )
    bundle = client.fetch_bundle("AAPL", "320193")

    assert bundle["cik"] == "0000320193"
    assert len(calls) == 2
    assert "/companyfacts/CIK0000320193.json" in calls[0]
    assert "/submissions/CIK0000320193.json" in calls[1]
