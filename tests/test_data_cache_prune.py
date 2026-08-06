"""prune_bar_cache: deletes bar files for symbols that fell out of the
watchlist.

data/bars/ is shared by several independent consumers (the selection pools, the
Runtime --symbols universe, daily_research, market scan, manual UI downloads),
and the caller only ever knows about its own list. Every other consumer's needs
are therefore protected inside prune_bar_cache itself, so no caller can delete
bars that something else is still relying on -- data_cache never re-fetches on
its own, so a wrong deletion is silent and permanent until someone re-downloads.
"""
from __future__ import annotations

import os
import time

import duckdb
import pytest

from trader import data_cache


def _write_bar_file(bars_dir, symbol: str, timeframe: str = "30m", *, age_days: float = 0.0):
    # Content doesn't matter to prune_bar_cache -- it only parses the filename
    # and deletes/keeps the file, never reads it as parquet.
    path = bars_dir / f"{symbol}_{timeframe}.parquet"
    path.write_bytes(b"placeholder")
    if age_days:
        old = time.time() - age_days * 86400
        os.utime(path, (old, old))
    return path


def _make_trade_db(path, *, fills=(), order_intents=()) -> None:
    con = duckdb.connect(str(path))
    try:
        con.execute("CREATE TABLE fills (symbol TEXT, side TEXT, filled_qty DOUBLE)")
        for symbol, side, qty in fills:
            con.execute("INSERT INTO fills VALUES (?, ?, ?)", [symbol, side, qty])
        con.execute("CREATE TABLE order_intents (symbol TEXT, state TEXT)")
        for symbol, state in order_intents:
            con.execute("INSERT INTO order_intents VALUES (?, ?)", [symbol, state])
    finally:
        con.close()


def _make_ai_db(path, rows=()) -> None:
    # trading_date is VARCHAR in the real ai_states.duckdb, holding ISO date
    # strings. Mirror that exactly -- a DATE column here would let a query that
    # cannot run against production pass its test.
    con = duckdb.connect(str(path))
    try:
        con.execute(
            "CREATE TABLE research_snapshots (symbol TEXT, trading_date VARCHAR)"
        )
        for symbol, trading_date in rows:
            con.execute(
                "INSERT INTO research_snapshots VALUES (?, ?)",
                [symbol, trading_date.isoformat()],
            )
    finally:
        con.close()


@pytest.fixture
def isolated_bars(tmp_path, monkeypatch):
    bars_dir = tmp_path / "bars"
    bars_dir.mkdir()
    monkeypatch.setattr(data_cache, "_BARS_DIR", bars_dir)
    monkeypatch.setattr(data_cache, "_CACHE", {})
    # Neutralize the ambient environment so each test states its own universe.
    monkeypatch.delenv("SYMBOLS", raising=False)
    return bars_dir


@pytest.fixture
def missing_dbs(tmp_path):
    """Paths that do not exist -- protection sources degrade to empty."""
    return {
        "db_path": tmp_path / "no-trade.duckdb",
        "ai_db_path": tmp_path / "no-ai.duckdb",
    }


def test_prune_removes_symbols_outside_watchlist(isolated_bars, missing_dbs):
    _write_bar_file(isolated_bars, "AAPL", age_days=30)
    _write_bar_file(isolated_bars, "ZZZZ", age_days=30)  # kicked out of the watchlist

    result = data_cache.prune_bar_cache(["AAPL"], **missing_dbs)

    assert "ZZZZ_30m.parquet" in result["removed_files"]
    assert not (isolated_bars / "ZZZZ_30m.parquet").exists()
    assert (isolated_bars / "AAPL_30m.parquet").exists()


def test_prune_protects_open_position_even_if_not_in_watchlist(isolated_bars, tmp_path):
    _write_bar_file(isolated_bars, "TSLA", age_days=30)
    db_path = tmp_path / "trade.duckdb"
    _make_trade_db(db_path, fills=[("TSLA", "BUY", 10.0)])

    result = data_cache.prune_bar_cache(
        [], db_path=db_path, ai_db_path=tmp_path / "no-ai.duckdb"
    )

    assert (isolated_bars / "TSLA_30m.parquet").exists()
    assert "TSLA" in result["kept_symbols"]


def test_prune_protects_open_order_even_if_not_in_watchlist(isolated_bars, tmp_path):
    _write_bar_file(isolated_bars, "MSFT", age_days=30)
    db_path = tmp_path / "trade.duckdb"
    _make_trade_db(db_path, order_intents=[("MSFT", "ACKNOWLEDGED")])

    data_cache.prune_bar_cache(
        [], db_path=db_path, ai_db_path=tmp_path / "no-ai.duckdb"
    )

    assert (isolated_bars / "MSFT_30m.parquet").exists()


def test_prune_does_not_protect_closed_position_or_terminal_order(isolated_bars, tmp_path):
    _write_bar_file(isolated_bars, "NFLX", age_days=30)
    db_path = tmp_path / "trade.duckdb"
    _make_trade_db(
        db_path,
        fills=[("NFLX", "BUY", 5.0), ("NFLX", "SELL", 5.0)],  # net flat
        order_intents=[("NFLX", "FILLED")],  # terminal state
    )

    data_cache.prune_bar_cache(
        [], db_path=db_path, ai_db_path=tmp_path / "no-ai.duckdb"
    )

    assert not (isolated_bars / "NFLX_30m.parquet").exists()


def test_prune_protects_configured_symbols_universe(isolated_bars, missing_dbs, monkeypatch):
    """The regression that made this necessary: the Runtime trades --symbols,
    which has nothing to do with the selection pools the caller passes in.
    Deleting QQQ here would have silently halted trading on it."""
    monkeypatch.setenv("SYMBOLS", "SPY,QQQ,AAPL")
    _write_bar_file(isolated_bars, "QQQ", age_days=30)
    _write_bar_file(isolated_bars, "ZZZZ", age_days=30)

    result = data_cache.prune_bar_cache(["NVDA"], **missing_dbs)

    assert (isolated_bars / "QQQ_30m.parquet").exists()
    assert not (isolated_bars / "ZZZZ_30m.parquet").exists()
    assert "QQQ" in result["kept_symbols"]


def test_prune_protects_recently_researched_symbols(isolated_bars, tmp_path):
    """Runtime's universe is a CLI argument and never hits disk, so recent
    research snapshots stand in for it."""
    from datetime import date, timedelta

    ai_db = tmp_path / "ai_states.duckdb"
    _make_ai_db(ai_db, [("AMD", date.today() - timedelta(days=3))])
    _write_bar_file(isolated_bars, "AMD", age_days=30)

    data_cache.prune_bar_cache(
        [], db_path=tmp_path / "no-trade.duckdb", ai_db_path=ai_db
    )

    assert (isolated_bars / "AMD_30m.parquet").exists()


def test_prune_ignores_research_older_than_the_window(isolated_bars, tmp_path):
    from datetime import date, timedelta

    ai_db = tmp_path / "ai_states.duckdb"
    _make_ai_db(ai_db, [("OLDY", date.today() - timedelta(days=400))])
    _write_bar_file(isolated_bars, "OLDY", age_days=30)

    data_cache.prune_bar_cache(
        [], db_path=tmp_path / "no-trade.duckdb", ai_db_path=ai_db
    )

    assert not (isolated_bars / "OLDY_30m.parquet").exists()


def test_prune_keeps_freshly_downloaded_files_within_grace_window(isolated_bars, missing_dbs):
    """A symbol downloaded manually in the UI has not landed in any list yet;
    deleting it right away would undo the download the user just asked for."""
    _write_bar_file(isolated_bars, "FRESH", age_days=1)
    _write_bar_file(isolated_bars, "STALE", age_days=30)

    result = data_cache.prune_bar_cache([], grace_days=7, **missing_dbs)

    assert (isolated_bars / "FRESH_30m.parquet").exists()
    assert "FRESH_30m.parquet" in result["kept_by_grace"]
    assert not (isolated_bars / "STALE_30m.parquet").exists()


def test_prune_grace_window_can_be_disabled(isolated_bars, missing_dbs):
    _write_bar_file(isolated_bars, "FRESH", age_days=0)

    data_cache.prune_bar_cache([], grace_days=0, **missing_dbs)

    assert not (isolated_bars / "FRESH_30m.parquet").exists()


def test_prune_missing_db_fails_safe_to_watchlist_only(isolated_bars, missing_dbs):
    _write_bar_file(isolated_bars, "GOOGL", age_days=30)

    result = data_cache.prune_bar_cache([], **missing_dbs)

    assert not (isolated_bars / "GOOGL_30m.parquet").exists()
    assert result["kept_symbols"] == []
