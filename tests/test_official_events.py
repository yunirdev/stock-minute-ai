from __future__ import annotations


def test_official_events_parse_core_free_sources():
    from trader.official_events import get_official_events

    pages = {
        "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm": """
            <div>2026 FOMC Meetings</div>
            <div>June</div>
            <div>16-17*</div>
        """,
        "https://www.bls.gov/schedule/2026/06_sched.htm": """
            <td>10</td>
            <td>Consumer Price Index</td>
            <td>May 2026</td>
            <td>08:30 AM</td>
        """,
        "https://www.bea.gov/news/schedule": """
            <div>June 26</div>
            <div>08:30 AM</div>
            <div>Personal Income and Outlays, May 2026</div>
        """,
        "https://www.eia.gov/petroleum/supply/weekly/": """
            <span>Next Release Date: June 24, 2026</span>
        """,
    }

    result = get_official_events(
        "2026-06-01",
        "2026-06-30",
        fetcher=lambda url: pages[url],
    )

    titles = {event.title for event in result.events}
    assert "FOMC 官方日历" in titles
    assert "CPI 官方发布" in titles
    assert "Personal Income and Outlays, May 2026" in titles
    assert "EIA 原油库存官方发布" in titles
    assert {source.state for source in result.sources} == {"ok"}


def test_official_events_report_source_errors_without_faking_empty_calendar():
    from trader.official_events import get_official_events

    def failing_fetcher(url: str) -> str:
        if "bls.gov" in url:
            raise RuntimeError("blocked")
        return ""

    result = get_official_events(
        "2026-06-01",
        "2026-06-30",
        fetcher=failing_fetcher,
    )

    states = {source.name: source.state for source in result.sources}
    assert states["official_bls_calendar"] == "error"
    assert "official_bls_calendar" in states


def test_calendar_events_prefers_official_duplicate_over_nasdaq(monkeypatch):
    import trader.calendar_events as ce
    from trader.official_events import (
        OfficialEvent,
        OfficialEventsResult,
        OfficialSourceStatus,
    )

    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    monkeypatch.setattr(
        ce,
        "_now_pacific",
        lambda: ce.datetime(2026, 6, 17, 7, tzinfo=ce._PACIFIC_TZ),
    )
    monkeypatch.setattr(
        ce,
        "get_official_events",
        lambda *_: OfficialEventsResult(
            events=[
                OfficialEvent(
                    date="2026-06-17",
                    time_str="14:00 ET",
                    title="FOMC 官方日历",
                    impact="critical",
                    category="fomc",
                    source="official_fed_fomc",
                )
            ],
            sources=[
                OfficialSourceStatus("official_fed_fomc", "ok"),
                OfficialSourceStatus("official_bls_calendar", "ok"),
                OfficialSourceStatus("official_bea_calendar", "ok"),
            ],
        ),
    )
    monkeypatch.setattr(
        ce,
        "_fetch_nasdaq_economic_with_status",
        lambda *_: (
            [
                ce.CalendarEvent(
                    date="2026-06-17",
                    time_str="14:00 ET",
                    title_cn="FOMC 事件",
                    title_en="FOMC Statement",
                    impact="critical",
                    category="fomc",
                    source="nasdaq_economic",
                )
            ],
            None,
        ),
    )

    result = ce.get_upcoming_events_with_status(symbols=[], days=1)

    assert len(result.events) == 1
    assert result.events[0].source == "official_fed_fomc"
    assert result.source_state("official_fed_fomc") == "ok"
    assert result.source_state("nasdaq_economic") == "ok"
    assert result.has_source_issue is False
