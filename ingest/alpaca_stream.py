import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import aiohttp
import duckdb
from dotenv import load_dotenv


@dataclass
class Bar:
    symbol: str
    timestamp: str  # ISO8601
    open: float
    high: float
    low: float
    close: float
    volume: float


class AlpacaBarStreamer:
    """
    Streams minute bars from Alpaca Market Data WebSocket and:
      1) pushes latest bar per symbol into an in-memory dict
      2) optionally calls on_bar(bar) callback
      3) persists to DuckDB
    """
    def __init__(self, symbols: List[str], feed: str, db_path: str = "data.duckdb"):
        load_dotenv()
        self.api_key = os.getenv("ALPACA_API_KEY", "")
        self.api_secret = os.getenv("ALPACA_API_SECRET", "")
        if not self.api_key or not self.api_secret:
            raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_API_SECRET in .env")

        self.symbols = symbols
        self.feed = feed
        self.ws_url = "wss://stream.data.alpaca.markets/v2/test" if feed == "test" else f"wss://stream.data.alpaca.markets/v2/{feed}"
        self.latest: Dict[str, Bar] = {}
        self.db_path = db_path
        self._stop = False
        self.on_bar = None  # optional callback

        self._init_db()

    def _init_db(self):
        con = duckdb.connect(self.db_path)
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

    def stop(self):
        self._stop = True

    def _upsert_bar(self, bar: Bar):
        con = duckdb.connect(self.db_path)
        con.execute(
            """
            INSERT INTO minute_bars(symbol, ts, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, ts) DO UPDATE SET
              open=excluded.open,
              high=excluded.high,
              low=excluded.low,
              close=excluded.close,
              volume=excluded.volume;
            """,
            [bar.symbol, bar.timestamp, bar.open, bar.high, bar.low, bar.close, bar.volume],
        )
        con.close()

    async def run(self):
        # Auth can be done via message after connect (docs).
        # Subscribe message: {"action":"subscribe","bars":["AAPL",...]} (docs).
        # Messages are arrays like [{"T":"success",...}, ...] (docs).
        auth_msg = {"action": "auth", "key": self.api_key, "secret": self.api_secret}
        sub_msg = {"action": "subscribe", "bars": self.symbols}

        backoff = 1
        while not self._stop:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(self.ws_url, heartbeat=20) as ws:
                        await ws.send_str(json.dumps(auth_msg))
                        await ws.send_str(json.dumps(sub_msg))

                        backoff = 1  # reset on success
                        async for msg in ws:
                            if self._stop:
                                break

                            if msg.type == aiohttp.WSMsgType.TEXT:
                                data = json.loads(msg.data)
                                # data is a list of messages
                                for item in data:
                                    t = item.get("T")
                                    if t in ("success", "subscription"):
                                        continue
                                    if t == "error":
                                        raise RuntimeError(f"Alpaca error: {item}")
                                    if t == "b":  # bar message type in Alpaca stream
                                        bar = Bar(
                                            symbol=item["S"],
                                            timestamp=item["t"],
                                            open=float(item["o"]),
                                            high=float(item["h"]),
                                            low=float(item["l"]),
                                            close=float(item["c"]),
                                            volume=float(item["v"]),
                                        )
                                        self.latest[bar.symbol] = bar
                                        self._upsert_bar(bar)
                                        if self.on_bar:
                                            self.on_bar(bar)

                            elif msg.type == aiohttp.WSMsgType.ERROR:
                                raise RuntimeError(f"WebSocket error: {msg.data}")

            except Exception as e:
                # simple reconnect with exponential backoff
                print(f"[stream] disconnected: {e}")
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)