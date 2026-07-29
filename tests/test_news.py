from datetime import datetime, timezone

from trader.models import NewsEvent
from trader.news import NewsEventStore, poll_all_sources

NOW = datetime(2026, 7, 28, 14, 0, tzinfo=timezone.utc)


def _event(event_id: str, symbol: str = "AAPL") -> NewsEvent:
    return NewsEvent(
        event_id=event_id,
        kind="news",
        symbol=symbol,
        title=f"headline {event_id}",
        summary="summary",
        url="https://example.com",
        severity=0.5,
        ts=NOW,
        source="test",
    )


def test_record_batch_persists_and_dedups_by_event_id(tmp_path):
    store = NewsEventStore(tmp_path / "trade.duckdb")
    store.record_batch([_event("e1"), _event("e2")], recorded_at=NOW)
    # Re-recording the same event_id must not raise or duplicate.
    store.record_batch([_event("e1")], recorded_at=NOW)

    import duckdb

    conn = duckdb.connect(str(tmp_path / "trade.duckdb"), read_only=True)
    rows = conn.execute("SELECT event_id FROM news_events ORDER BY event_id").fetchall()
    conn.close()
    assert [r[0] for r in rows] == ["e1", "e2"]


def test_record_batch_with_empty_list_is_a_noop(tmp_path):
    store = NewsEventStore(tmp_path / "trade.duckdb")
    assert store.record_batch([], recorded_at=NOW) == 0


class _StubSource:
    def __init__(self, events=None, error: Exception | None = None):
        self._events = events or []
        self._error = error

    def poll(self, since: datetime):
        if self._error:
            raise self._error
        return self._events


def test_poll_all_sources_persists_combined_results(tmp_path):
    store = NewsEventStore(tmp_path / "trade.duckdb")
    sources = [
        ("a", _StubSource([_event("a1")]), NOW),
        ("b", _StubSource([_event("b1")]), NOW),
    ]
    events = poll_all_sources(sources, store, now=NOW)
    assert {e.event_id for e in events} == {"a1", "b1"}

    import duckdb

    conn = duckdb.connect(str(tmp_path / "trade.duckdb"), read_only=True)
    rows = conn.execute("SELECT event_id FROM news_events").fetchall()
    conn.close()
    assert {r[0] for r in rows} == {"a1", "b1"}


def test_poll_all_sources_tolerates_one_source_failing(tmp_path):
    store = NewsEventStore(tmp_path / "trade.duckdb")
    sources = [
        ("broken", _StubSource(error=RuntimeError("rate limited")), NOW),
        ("ok", _StubSource([_event("ok1")]), NOW),
    ]
    events = poll_all_sources(sources, store, now=NOW)
    assert [e.event_id for e in events] == ["ok1"]


def test_poll_all_sources_without_store_still_returns_events():
    sources = [("a", _StubSource([_event("a1")]), NOW)]
    events = poll_all_sources(sources, None, now=NOW)
    assert [e.event_id for e in events] == ["a1"]
