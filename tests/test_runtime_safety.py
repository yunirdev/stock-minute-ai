from __future__ import annotations

from datetime import datetime, timedelta, timezone


def test_intraday_latest_bar_freshness_rejects_stale_bars():
    from trader.models import Bar
    from trader.runtime import _latest_bar_is_fresh

    now = datetime(2026, 6, 22, 16, 0, tzinfo=timezone.utc)
    stale = Bar(
        symbol="AAPL",
        timestamp=now - timedelta(days=1),
        open=100,
        high=101,
        low=99,
        close=100,
        volume=1_000,
    )

    assert not _latest_bar_is_fresh(stale, "5m", now=now)


def test_intraday_latest_bar_freshness_allows_feed_delay():
    from trader.models import Bar
    from trader.runtime import _latest_bar_is_fresh

    now = datetime(2026, 6, 22, 16, 0, tzinfo=timezone.utc)
    delayed = Bar(
        symbol="AAPL",
        timestamp=now - timedelta(minutes=30),
        open=100,
        high=101,
        low=99,
        close=100,
        volume=1_000,
    )

    assert _latest_bar_is_fresh(delayed, "5m", now=now)
