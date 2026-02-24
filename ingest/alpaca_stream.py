import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List

import aiohttp
import duckdb
from dotenv import load_dotenv


@dataclass
class Bar:
    symbol: str
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class AlpacaBarStreamer:
    def __init__(self, symbols: List[str], feed: str, db_path: str = "market.duckdb"):
        load_dotenv()
        self.api_key = os.getenv("ALPACA_API_KEY", "")
        self.api_secret = os.getenv("ALPACA_API_SECRET", "")
        if not self.api_key or not self.api_secret:
            raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_API_SECRET in .env")

        self.symbols = [s.upper() for s in symbols]
        self.feed = feed
        self.db_path = db_path

        if feed == "test":
            self.ws_url = "wss://stream.data.alpaca.markets/v2/test"
        else:
            self.ws_url = f"wss://stream.data.alpaca.markets/v2/{feed}"

        self.latest: Dict[str, Bar] = {}
        self._stop = False
        self.on_bar = None

        self._init_db()

    def _init_db(self):
        con = duckdb.connect(self.db_path)
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
        con.close()

    def stop(self):
        self._stop = True

    def _upsert_bar(self, bar: Bar):
        con = duckdb.connect(self.db_path)
        con.execute(
            """
            INSERT INTO minute_bars(symbol, ts, open, high, low, close, volume, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, ts, source) DO UPDATE SET
                open=excluded.open,
                high=excluded.high,
                low=excluded.low,
                close=excluded.close,
                volume=excluded.volume;
            """,
            [
                bar.symbol,
                bar.timestamp,
                bar.open,
                bar.high,
                bar.low,
                bar.close,
                int(bar.volume),
                "alpaca",
            ],
        )
        con.close()

    async def run(self):
        auth_msg = {"action": "auth", "key": self.api_key, "secret": self.api_secret}
        sub_msg = {"action": "subscribe", "bars": self.symbols}

        backoff = 1
        while not self._stop:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(self.ws_url, heartbeat=20) as ws:
                        await ws.send_str(json.dumps(auth_msg))
                        await ws.send_str(json.dumps(sub_msg))

                        backoff = 1
                        async for msg in ws:
                            if self._stop:
                                break

                            if msg.type == aiohttp.WSMsgType.TEXT:
                                data = json.loads(msg.data)
                                for item in data:
                                    t = item.get("T")
                                    if t in ("success", "subscription"):
                                        continue
                                    if t == "error":
                                        raise RuntimeError(f"Alpaca error: {item}")
                                    if t == "b":
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
                print(f"[stream] disconnected: {e}")
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)