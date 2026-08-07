"""Bar cache feed provenance tagging and mismatch detection.

2026-08-06: a batch of cached bars turned out to hold IEX single-venue volume
(2-7% of the real consolidated figure) instead of the sip feed .env is
configured for. It sat on disk undetected for months because data_cache is
strictly local-first and never re-verifies a file's content -- nothing
recorded which feed a file was fetched with, so nothing could ever notice a
mismatch. These tests cover the fix: every write tags the feed it came from,
and describe_cached_bars (which feeds the research snapshot quality report)
flags a mismatch against the currently configured feed.
"""
from __future__ import annotations

import pandas as pd
import pytest

from trader import data_cache


def _frame(rows: int = 60, symbol: str = "AAPL") -> pd.DataFrame:
    ts = pd.date_range("2026-08-01", periods=rows, freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "symbol": [symbol] * rows,
            "timestamp_utc": ts,
            "timestamp": ts,
            "open": [1.0] * rows,
            "high": [1.0] * rows,
            "low": [1.0] * rows,
            "close": [1.0] * rows,
            "volume": [1.0] * rows,
        }
    )


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    bars_dir = tmp_path / "bars"
    bars_dir.mkdir()
    monkeypatch.setattr(data_cache, "_BARS_DIR", bars_dir)
    monkeypatch.setattr(data_cache, "_CACHE", {})
    return bars_dir


def _configured(monkeypatch, feed: str) -> None:
    monkeypatch.setattr(
        data_cache, "_alpaca_creds", lambda: ("key", "secret", feed)
    )


# --- writers tag what they wrote --------------------------------------------


def test_fetch_and_save_tags_alpaca_rows_with_the_configured_feed(
    isolated_cache, monkeypatch
):
    _configured(monkeypatch, "sip")
    monkeypatch.setattr(data_cache, "_alpaca_fetch_full", lambda s, tf: _frame())

    df = data_cache.fetch_and_save("AAPL", "30m")

    assert set(df["source_feed"]) == {"sip"}


def test_fetch_and_save_tags_yfinance_fallback_distinctly(isolated_cache, monkeypatch):
    _configured(monkeypatch, "sip")
    monkeypatch.setattr(data_cache, "_alpaca_fetch_full", lambda s, tf: pd.DataFrame())
    monkeypatch.setattr(data_cache, "_yf_fetch", lambda s, tf, period: _frame())

    df = data_cache.fetch_and_save("AAPL", "30m")

    assert set(df["source_feed"]) == {"yfinance"}


def test_fetch_and_save_1m_is_always_tagged_yfinance(isolated_cache, monkeypatch):
    _configured(monkeypatch, "sip")
    monkeypatch.setattr(data_cache, "_yf_fetch", lambda s, tf, period: _frame())

    df = data_cache.fetch_and_save("AAPL", "1m")

    assert set(df["source_feed"]) == {"yfinance"}


def test_upsert_bars_tags_with_the_configured_feed(isolated_cache, monkeypatch):
    _configured(monkeypatch, "sip")

    data_cache.upsert_bars("AAPL", "5m", _frame())

    saved = data_cache._load_from_disk("AAPL", "5m")
    assert set(saved["source_feed"]) == {"sip"}


# --- describe_cached_bars flags a stale-feed file ---------------------------


def test_describe_flags_mismatch_between_cached_and_configured_feed(
    isolated_cache, monkeypatch
):
    tagged = _frame().assign(source_feed="iex")
    data_cache._save_to_disk("AAPL", "5m", tagged)
    _configured(monkeypatch, "sip")

    result = data_cache.describe_cached_bars("AAPL", "5m")

    assert result["status"] == "DEGRADED"
    assert result["failure_code"] == "BAR_FEED_STALE"
    assert result["metadata"]["cached_feed"] == "iex"
    assert result["metadata"]["feed_mismatch"] is True


def test_describe_is_ok_when_cached_feed_matches_configured(isolated_cache, monkeypatch):
    tagged = _frame().assign(source_feed="sip")
    data_cache._save_to_disk("AAPL", "5m", tagged)
    _configured(monkeypatch, "sip")

    result = data_cache.describe_cached_bars("AAPL", "5m")

    assert result["status"] == "OK"
    assert result["failure_code"] == ""
    assert result["metadata"]["feed_mismatch"] is False


def test_describe_never_flags_yfinance_rows_regardless_of_configured_feed(
    isolated_cache, monkeypatch
):
    """The feed concept is meaningless for yfinance -- it has no venue
    selection, so a "mismatch" against whatever Alpaca feed is configured
    would be pure noise."""
    tagged = _frame().assign(source_feed="yfinance")
    data_cache._save_to_disk("AAPL", "1m", tagged)
    _configured(monkeypatch, "sip")

    result = data_cache.describe_cached_bars("AAPL", "1m")

    assert result["status"] == "OK"
    assert result["metadata"]["feed_mismatch"] is False


def test_describe_treats_untagged_legacy_file_as_suspicious(isolated_cache, monkeypatch):
    """Files written before this fix carry no source_feed column at all --
    exactly the case that let the IEX-volume batch go unnoticed. Missing
    provenance must not be treated as trustworthy by default."""
    untagged = _frame()  # no source_feed column, as every pre-fix file is
    data_cache._save_to_disk("AAPL", "5m", untagged)
    _configured(monkeypatch, "sip")

    result = data_cache.describe_cached_bars("AAPL", "5m")

    assert result["status"] == "DEGRADED"
    assert result["failure_code"] == "BAR_FEED_STALE"
    assert result["metadata"]["cached_feed"] == ""


def test_describe_does_not_downgrade_below_the_history_length_check(
    isolated_cache, monkeypatch
):
    """A too-short file is already DEGRADED with BAR_HISTORY_SHORT; a feed
    mismatch on top of that must not produce a second, contradictory
    failure_code for the same status."""
    tagged = _frame(rows=10).assign(source_feed="iex")
    data_cache._save_to_disk("AAPL", "5m", tagged)
    _configured(monkeypatch, "sip")

    result = data_cache.describe_cached_bars("AAPL", "5m")

    assert result["status"] == "DEGRADED"
    assert result["failure_code"] == "BAR_HISTORY_SHORT"
