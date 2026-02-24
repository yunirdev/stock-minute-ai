"""Fetch historical bar data from Alpaca and write to DuckDB.

Usage:
    python -m ingest.history
    # or
    python ingest/history.py

Reads from .env:
    ALPACA_API_KEY
    ALPACA_API_SECRET
    ALPACA_DATA_FEED   (iex | sip | delayed_sip)
    SYMBOLS            (comma-separated, e.g. AAPL,MSFT,NVDA)
    HISTORY_START      (e.g. 2016-01-01)
    HISTORY_CHUNK_DAYS (chunk size for minute-bar fetches, e.g. 20)
    DB_PATH            (path to DuckDB file, e.g. market.duckdb)
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import os
from datetime import datetime, timedelta, timezone

import duckdb
import pandas as pd
from dotenv import load_dotenv

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed, Adjustment


def _ensure_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS minute_bars (
            symbol TEXT,
            ts TIMESTAMPTZ,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume BIGINT,
            source TEXT,
            PRIMARY KEY(symbol, ts, source)
        );
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS daily_bars (
            symbol TEXT,
            dt DATE,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume BIGINT,
            source TEXT,
            PRIMARY KEY(symbol, dt, source)
        );
        """
    )


def _upsert_daily(con: duckdb.DuckDBPyConnection, df: pd.DataFrame, source: str) -> int:
    df = df[["symbol", "timestamp", "open", "high", "low", "close", "volume"]].copy()
    df["dt"] = df["timestamp"].apply(
        lambda t: t.date() if hasattr(t, "date") else t
    )
    df["source"] = source
    con.register("_tmp_daily", df)
    con.execute(
        """
        INSERT INTO daily_bars(symbol, dt, open, high, low, close, volume, source)
        SELECT symbol, dt, open, high, low, close, volume::BIGINT, source
        FROM _tmp_daily
        ON CONFLICT(symbol, dt, source) DO UPDATE SET
            open=excluded.open,
            high=excluded.high,
            low=excluded.low,
            close=excluded.close,
            volume=excluded.volume;
        """
    )
    con.unregister("_tmp_daily")
    return len(df)


def _upsert_minute(con: duckdb.DuckDBPyConnection, df: pd.DataFrame, source: str) -> int:
    df = df[["symbol", "timestamp", "open", "high", "low", "close", "volume"]].copy()
    df = df.rename(columns={"timestamp": "ts"})
    df["source"] = source
    con.register("_tmp_minute", df)
    con.execute(
        """
        INSERT INTO minute_bars(symbol, ts, open, high, low, close, volume, source)
        SELECT symbol, ts, open, high, low, close, volume::BIGINT, source
        FROM _tmp_minute
        ON CONFLICT(symbol, ts, source) DO UPDATE SET
            open=excluded.open,
            high=excluded.high,
            low=excluded.low,
            close=excluded.close,
            volume=excluded.volume;
        """
    )
    con.unregister("_tmp_minute")
    return len(df)


def _parse_feed(feed_str: str) -> DataFeed:
    feed_str = feed_str.lower()
    mapping = {
        "iex": DataFeed.IEX,
        "sip": DataFeed.SIP,
        "delayed_sip": DataFeed.DELAYED_SIP,
    }
    return mapping.get(feed_str, DataFeed.IEX)


def main() -> None:
    load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

    api_key = os.getenv("ALPACA_API_KEY", "")
    api_secret = os.getenv("ALPACA_API_SECRET", "")
    if not api_key or not api_secret:
        print("ERROR: Missing ALPACA_API_KEY / ALPACA_API_SECRET in .env")
        sys.exit(1)

    symbols = [
        s.strip().upper()
        for s in os.getenv("SYMBOLS", "AAPL,MSFT").split(",")
        if s.strip()
    ]
    history_start = os.getenv("HISTORY_START", "2020-01-01")
    chunk_days = int(os.getenv("HISTORY_CHUNK_DAYS", "20"))
    db_path = os.getenv("DB_PATH", "market.duckdb")
    feed_str = os.getenv("ALPACA_DATA_FEED", os.getenv("ALPACA_FEED", "iex"))
    feed = _parse_feed(feed_str)

    print(f"DB_PATH : {db_path}")
    print(f"Symbols : {symbols}")
    print(f"Start   : {history_start}")
    print(f"Feed    : {feed_str}")

    client = StockHistoricalDataClient(api_key, api_secret)
    con = duckdb.connect(db_path)
    _ensure_schema(con)

    start_dt = datetime.strptime(history_start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.now(timezone.utc)

    # ── Daily bars ────────────────────────────────────────────────────────────
    print(f"\nFetching daily bars {history_start} → today ...")
    try:
        req = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day,
            start=start_dt,
            end=end_dt,
            feed=feed,
            adjustment=Adjustment.RAW,
        )
        bars = client.get_stock_bars(req)
        df = bars.df
        if df is not None and not df.empty:
            df = df.reset_index()
            count = _upsert_daily(con, df, "alpaca")
            print(f"  daily_bars: {count} rows upserted")
        else:
            print("  daily_bars: no data returned")
    except Exception as exc:
        print(f"  daily_bars: ERROR – {exc}")

    # ── Minute bars (chunked to avoid rate-limit / memory issues) ─────────────
    print(f"\nFetching minute bars in {chunk_days}-day chunks ...")
    total_minute = 0
    cur = start_dt
    while cur < end_dt:
        chunk_end = min(cur + timedelta(days=chunk_days), end_dt)
        print(f"  {cur.date()} → {chunk_end.date()} ...", end=" ", flush=True)
        try:
            req = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=TimeFrame.Minute,
                start=cur,
                end=chunk_end,
                feed=feed,
            )
            bars = client.get_stock_bars(req)
            df = bars.df
            if df is not None and not df.empty:
                df = df.reset_index()
                count = _upsert_minute(con, df, "alpaca")
                total_minute += count
                print(f"{count} rows")
            else:
                print("no data")
        except Exception as exc:
            print(f"ERROR – {exc}")
        cur = chunk_end

    print(f"\n  minute_bars total: {total_minute} rows upserted")
    con.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
