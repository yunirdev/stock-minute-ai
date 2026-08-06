"""prune_bar_cache: deletes bar files for symbols that fell out of the
watchlist, but never touches a symbol that still has an open position or a
non-terminal order — that safety check happens inside prune_bar_cache itself
so every caller gets it for free."""
from __future__ import annotations

import duckdb
import pytest

from trader import data_cache


def _write_bar_file(bars_dir, symbol: str, timeframe: str = "30m") -> None:
    # Content doesn't matter to prune_bar_cache — it only parses the
    # filename and deletes/keeps the file, never reads it as parquet.
    (bars_dir / f"{symbol}_{timeframe}.parquet").write_bytes(b"placeholder")


def _make_trade_db(path, *, fills=(), order_intents=()) -> None:
    con = duckdb.connect(str(path))
    try:
        con.execute(
            "CREATE TABLE fills (symbol TEXT, side TEXT, filled_qty DOUBLE)"
        )
        for symbol, side, qty in fills:
            con.execute(
                "INSERT INTO fills VALUES (?, ?, ?)", [symbol, side, qty]
            )
        con.execute(
            "CREATE TABLE order_intents (symbol TEXT, state TEXT)"
        )
        for symbol, state in order_intents:
            con.execute(
                "INSERT INTO order_intents VALUES (?, ?)", [symbol, state]
            )
    finally:
        con.close()


@pytest.fixture
def isolated_bars(tmp_path, monkeypatch):
    bars_dir = tmp_path / "bars"
    bars_dir.mkdir()
    monkeypatch.setattr(data_cache, "_BARS_DIR", bars_dir)
    monkeypatch.setattr(data_cache, "_CACHE", {})
    return bars_dir


def test_prune_removes_symbols_outside_watchlist(isolated_bars, tmp_path):
    _write_bar_file(isolated_bars, "AAPL")
    _write_bar_file(isolated_bars, "ZZZZ")  # kicked out of the watchlist

    result = data_cache.prune_bar_cache(["AAPL"], db_path=tmp_path / "missing.duckdb")

    assert "ZZZZ_30m.parquet" in result["removed_files"]
    assert not (isolated_bars / "ZZZZ_30m.parquet").exists()
    assert (isolated_bars / "AAPL_30m.parquet").exists()


def test_prune_protects_open_position_even_if_not_in_watchlist(isolated_bars, tmp_path):
    _write_bar_file(isolated_bars, "TSLA")
    db_path = tmp_path / "trade.duckdb"
    _make_trade_db(db_path, fills=[("TSLA", "BUY", 10.0)])

    result = data_cache.prune_bar_cache([], db_path=db_path)

    assert (isolated_bars / "TSLA_30m.parquet").exists()
    assert "TSLA" in result["kept_symbols"]


def test_prune_protects_open_order_even_if_not_in_watchlist(isolated_bars, tmp_path):
    _write_bar_file(isolated_bars, "MSFT")
    db_path = tmp_path / "trade.duckdb"
    _make_trade_db(db_path, order_intents=[("MSFT", "ACKNOWLEDGED")])

    data_cache.prune_bar_cache([], db_path=db_path)

    assert (isolated_bars / "MSFT_30m.parquet").exists()


def test_prune_does_not_protect_closed_position_or_terminal_order(isolated_bars, tmp_path):
    _write_bar_file(isolated_bars, "NFLX")
    db_path = tmp_path / "trade.duckdb"
    _make_trade_db(
        db_path,
        fills=[("NFLX", "BUY", 5.0), ("NFLX", "SELL", 5.0)],  # net flat
        order_intents=[("NFLX", "FILLED")],  # terminal state
    )

    data_cache.prune_bar_cache([], db_path=db_path)

    assert not (isolated_bars / "NFLX_30m.parquet").exists()


def test_prune_missing_db_fails_safe_to_watchlist_only(isolated_bars, tmp_path):
    _write_bar_file(isolated_bars, "GOOGL")

    result = data_cache.prune_bar_cache([], db_path=tmp_path / "does_not_exist.duckdb")

    assert not (isolated_bars / "GOOGL_30m.parquet").exists()
    assert result["kept_symbols"] == []
