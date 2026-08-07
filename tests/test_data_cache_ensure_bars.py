"""ensure_bars: top up the local cache for symbols the Runtime never traded.

The Runtime only writes bars for its own --symbols at its own --tf, so the
research universe is routinely wider than anything the cache has seen. Without
a top-up, screening runs on symbols with no price data at all and every such
snapshot records local_bar_cache=BAR_CACHE_EMPTY.
"""
from __future__ import annotations

import pandas as pd
import pytest

from trader import data_cache


def _frame(rows: int, symbol: str = "AAPL") -> pd.DataFrame:
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
def stub_cache(monkeypatch, tmp_path):
    """Isolate the cache and record what would have been fetched/persisted.

    get_bars is stubbed, so nothing here actually touches disk -- without a
    default, _file_age_seconds would see "file does not exist" (age=inf) and
    treat every symbol as stale. Default to "just fetched" so existing tests
    keep describing "this data is fine" unless a test overrides it to cover
    the staleness path itself.
    """
    calls = {"fetched": [], "saved": []}

    monkeypatch.setattr(data_cache, "_BARS_DIR", tmp_path)
    monkeypatch.setattr(data_cache, "_CACHE", {})
    monkeypatch.setattr(data_cache, "_file_age_seconds", lambda s, tf: 0.0)

    def fake_window(symbol, timeframe, start, end):
        calls["fetched"].append((symbol, timeframe))
        return calls.get("response", _frame(100, symbol))

    def fake_upsert(symbol, timeframe, df):
        calls["saved"].append((symbol, timeframe, len(df)))

    monkeypatch.setattr(data_cache, "fetch_alpaca_bars_window", fake_window)
    monkeypatch.setattr(data_cache, "upsert_bars", fake_upsert)
    return calls


def test_downloads_only_symbols_with_insufficient_cache(stub_cache, monkeypatch):
    # AAPL already has plenty; ZZZZ has nothing.
    monkeypatch.setattr(
        data_cache,
        "get_bars",
        lambda s, tf: _frame(300) if s == "AAPL" else pd.DataFrame(),
    )

    result = data_cache.ensure_bars(["AAPL", "ZZZZ"], "5m")

    assert result["sufficient"] == ["AAPL"]
    assert result["filled"] == ["ZZZZ"]
    assert stub_cache["fetched"] == [("ZZZZ", "5m")]  # AAPL never hit the network


def test_tops_up_cache_that_is_too_short(stub_cache, monkeypatch):
    """Below the DEGRADED threshold counts as needing a top-up, not just empty."""
    monkeypatch.setattr(data_cache, "get_bars", lambda s, tf: _frame(10))

    result = data_cache.ensure_bars(["AAPL"], "5m", min_rows=40)

    assert result["filled"] == ["AAPL"]
    assert stub_cache["saved"] == [("AAPL", "5m", 100)]


def test_uses_a_bounded_window_not_full_history(stub_cache, monkeypatch):
    """Full history for 5m is hundreds of thousands of bars; the cache only ever
    holds a recent window, so the top-up must stay bounded."""
    captured = {}

    def fake_window(symbol, timeframe, start, end):
        captured["span_days"] = (end - start).days
        return _frame(100, symbol)

    monkeypatch.setattr(data_cache, "fetch_alpaca_bars_window", fake_window)
    monkeypatch.setattr(data_cache, "get_bars", lambda s, tf: pd.DataFrame())

    data_cache.ensure_bars(["AAPL"], "5m", lookback_days=30)

    assert captured["span_days"] == 30


@pytest.mark.parametrize("timeframe,expected", [("5m", 30), ("1d", 400)])
def test_lookback_window_scales_so_min_rows_is_reachable(
    stub_cache, monkeypatch, timeframe, expected
):
    """A daily bar only yields ~21 rows in 30 days, so a fixed window would keep
    the cache permanently under threshold and re-download on every single run."""
    captured = {}

    def fake_window(symbol, tf, start, end):
        captured["span_days"] = (end - start).days
        return _frame(300, symbol)

    monkeypatch.setattr(data_cache, "fetch_alpaca_bars_window", fake_window)
    monkeypatch.setattr(data_cache, "get_bars", lambda s, tf: pd.DataFrame())

    data_cache.ensure_bars(["AAPL"], timeframe)

    assert captured["span_days"] == expected


def test_thin_cache_above_degraded_line_still_tops_up(stub_cache, monkeypatch):
    """120 rows of 5m clears describe_cached_bars' 40-row DEGRADED threshold but
    is far too little to build strategy statistics from -- that gap is exactly
    why min_rows defaults well above 40."""
    monkeypatch.setattr(data_cache, "get_bars", lambda s, tf: _frame(120))

    result = data_cache.ensure_bars(["NVDA"], "5m")

    assert result["filled"] == ["NVDA"]


def test_download_failure_is_reported_but_not_raised(stub_cache, monkeypatch):
    """A research batch must not die because one symbol failed to download."""
    def boom(symbol, timeframe, start, end):
        raise RuntimeError("alpaca down")

    monkeypatch.setattr(data_cache, "fetch_alpaca_bars_window", boom)
    monkeypatch.setattr(data_cache, "get_bars", lambda s, tf: pd.DataFrame())

    result = data_cache.ensure_bars(["AAPL", "MSFT"], "5m")

    assert result["failed"] == ["AAPL", "MSFT"]
    assert result["filled"] == []


def test_empty_response_counts_as_failure_and_saves_nothing(stub_cache, monkeypatch):
    monkeypatch.setattr(
        data_cache, "fetch_alpaca_bars_window", lambda *a, **k: pd.DataFrame()
    )
    monkeypatch.setattr(data_cache, "get_bars", lambda s, tf: pd.DataFrame())

    result = data_cache.ensure_bars(["AAPL"], "5m")

    assert result["failed"] == ["AAPL"]
    assert stub_cache["saved"] == []


def test_blank_symbols_are_skipped(stub_cache, monkeypatch):
    monkeypatch.setattr(data_cache, "get_bars", lambda s, tf: _frame(300))

    result = data_cache.ensure_bars(["AAPL", "", "  "], "5m")

    assert result["sufficient"] == ["AAPL"]


# --- staleness: row count alone is not enough -------------------------------
#
# _FILE_MAX_AGE existed in this module for a long time but nothing ever read
# it -- _ensure_loaded's own docstring said staleness was "left to manual
# refresh". That gap is exactly how a batch of IEX-scale volume (see
# test_data_cache_feed_tag.py) sat on disk undetected for months: the files
# had plenty of rows, so nothing ever asked whether the rows were current.


def test_refetches_a_fresh_row_count_that_is_past_its_max_age(stub_cache, monkeypatch):
    monkeypatch.setattr(data_cache, "get_bars", lambda s, tf: _frame(300))
    old_age = data_cache._FILE_MAX_AGE["5m"] + 3600  # one hour past the limit
    monkeypatch.setattr(data_cache, "_file_age_seconds", lambda s, tf: old_age)

    result = data_cache.ensure_bars(["AAPL"], "5m")

    assert result["filled"] == ["AAPL"]
    assert result["stale"] == ["AAPL"]
    assert result["sufficient"] == []


def test_does_not_refetch_a_fresh_file_with_enough_rows(stub_cache, monkeypatch):
    monkeypatch.setattr(data_cache, "get_bars", lambda s, tf: _frame(300))
    monkeypatch.setattr(
        data_cache, "_file_age_seconds", lambda s, tf: data_cache._FILE_MAX_AGE["5m"] - 60
    )

    result = data_cache.ensure_bars(["AAPL"], "5m")

    assert result["sufficient"] == ["AAPL"]
    assert result["filled"] == []
    assert stub_cache["fetched"] == []


def test_staleness_check_is_per_timeframe(stub_cache, monkeypatch):
    """1m tolerates only 8h before refresh; 1d tolerates 24h -- the same
    absolute age must not be judged the same way across timeframes."""
    age = 12 * 3600  # stale for 1m (8h), fresh for 1d (24h)
    monkeypatch.setattr(data_cache, "get_bars", lambda s, tf: _frame(300))
    monkeypatch.setattr(data_cache, "_file_age_seconds", lambda s, tf: age)

    result_1m = data_cache.ensure_bars(["AAPL"], "1m")
    result_1d = data_cache.ensure_bars(["AAPL"], "1d")

    assert result_1m["stale"] == ["AAPL"]
    assert result_1d["sufficient"] == ["AAPL"]
