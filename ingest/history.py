from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

import duckdb
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv


def init_db(db_path: str):
    con = duckdb.connect(db_path)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS minute_bars (
          symbol TEXT,
          ts TIMESTAMP,
          open DOUBLE,
          high DOUBLE,
          low DOUBLE,
          close DOUBLE,
          volume DOUBLE,
          PRIMARY KEY(symbol, ts)
        );
        """
    )
    con.close()


def upsert_df(db_path: str, df):
    if df is None or len(df) == 0:
        return
    con = duckdb.connect(db_path)
    con.execute(
        """
        INSERT INTO minute_bars(symbol, ts, open, high, low, close, volume)
        SELECT symbol, timestamp, open, high, low, close, volume FROM df
        ON CONFLICT(symbol, ts) DO UPDATE SET
          open=excluded.open,
          high=excluded.high,
          low=excluded.low,
          close=excluded.close,
          volume=excluded.volume;
        """
    )
    con.close()


def fetch_history(symbols: List[str], days: int = 5, db_path: str = "data.duckdb", feed: str = "iex"):
    load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")
    key = os.getenv("ALPACA_API_KEY", "")
    secret = os.getenv("ALPACA_API_SECRET", "")
    if not key or not secret:
        raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_API_SECRET")

    client = StockHistoricalDataClient(key, secret)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Minute,
        start=start,
        end=end,
        feed=feed,
        adjustment="raw",
    )

    bars = client.get_stock_bars(req)
    df = bars.df.reset_index()  # columns: symbol, timestamp, open, high, low, close, volume, ...
    df = df[["symbol", "timestamp", "open", "high", "low", "close", "volume"]]

    init_db(db_path)
    upsert_df(db_path, df)
    print(f"Inserted/updated {len(df)} rows for {symbols} from {start} to {end}.")


if __name__ == "__main__":
    symbols = os.getenv("SYMBOLS", "AAPL,MSFT").split(",")
    symbols = [s.strip().upper() for s in symbols if s.strip()]
    feed = os.getenv("ALPACA_FEED", "iex")
    fetch_history(symbols, days=5, db_path="data.duckdb", feed=feed)