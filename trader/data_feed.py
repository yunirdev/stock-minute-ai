"""
data_feed.py
Market data provider using Alpaca REST API.

Fetches historical bars for indicator computation and latest prices
for real-time portfolio valuation.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List

import pandas as pd
import requests

from .config import TradingConfig
from .models import Bar

logger = logging.getLogger(__name__)

_ALPACA_BASE = "https://data.alpaca.markets/v2"

_TF_MAP: Dict[str, str] = {
    "1m": "1Min",
    "5m": "5Min",
    "15m": "15Min",
    "30m": "30Min",
    "1h": "1Hour",
    "1d": "1Day",
}


class AlpacaDataFeed:
    """Fetches bars and latest prices from Alpaca Data REST API v2."""

    def __init__(self, config: TradingConfig) -> None:
        self._cfg = config
        self._headers = {
            "APCA-API-KEY-ID": config.alpaca_api_key,
            "APCA-API-SECRET-KEY": config.alpaca_secret_key,
        }
        self._tf = _TF_MAP.get(config.timeframe, "5Min")

    def fetch_bars(self, symbol: str, n_bars: int = 120) -> List[Bar]:
        """Return up to *n_bars* recent bars for *symbol*."""
        # Free Alpaca plan: SIP data has 15-min delay; use 20-min buffer to avoid 403
        end = datetime.now(timezone.utc) - timedelta(minutes=20)
        # Request a wider window; Alpaca skips after-hours gaps
        lookback_days = max(2, n_bars // 78 + 3)
        start = end - timedelta(days=lookback_days)

        params = {
            "timeframe": self._tf,
            "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "limit": n_bars * 2,
            "feed": self._cfg.alpaca_feed,
            "sort": "asc",
        }
        url = f"{_ALPACA_BASE}/stocks/{symbol}/bars"
        try:
            resp = requests.get(
                url, headers=self._headers, params=params, timeout=15)
            resp.raise_for_status()
            raw = resp.json().get("bars", [])
        except Exception as exc:
            logger.error("Alpaca fetch_bars %s: %s", symbol, exc)
            return []

        bars: List[Bar] = []
        for b in raw[-n_bars:]:
            bars.append(Bar(
                symbol=symbol,
                timestamp=pd.Timestamp(b["t"]).to_pydatetime(),
                open=float(b["o"]),
                high=float(b["h"]),
                low=float(b["l"]),
                close=float(b["c"]),
                volume=float(b["v"]),
            ))
        logger.debug("fetch_bars %s: got %d bars", symbol, len(bars))
        return bars

    def get_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Return {symbol: latest_close} for each symbol in *symbols*."""
        if not symbols:
            return {}
        url = f"{_ALPACA_BASE}/stocks/bars/latest"
        params = {
            "symbols": ",".join(symbols),
            "feed": self._cfg.alpaca_feed,
        }
        try:
            resp = requests.get(
                url, headers=self._headers, params=params, timeout=10)
            resp.raise_for_status()
            bars = resp.json().get("bars", {})
            return {sym: float(data["c"]) for sym, data in bars.items()}
        except Exception as exc:
            logger.error("Alpaca get_latest_prices: %s", exc)
            return {}
