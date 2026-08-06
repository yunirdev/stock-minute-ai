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

_TIMEFRAME_MINUTES: Dict[str, int] = {
    "1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60,
}
_RTH_MINUTES_PER_DAY = 390  # 6.5h 常规交易时段


def _bars_per_trading_day(timeframe: str) -> int:
    """一个常规交易日大致能攒出多少根这个周期的 bar。

    lookback_days 原来写死用 78 除（78 = 390/5，是 5 分钟 bar 一天的根数），
    换成 30m/1h 等其它周期时这个常数就不对了——30 分钟一天只有 13 根，用 78
    去除会把回看窗口算得远小于实际需要的天数，导致拉回来的 bar 数量长期不够
    filter（`len(bars) < 30`）要求的下限，尤其在 IEX 之类成交没那么密的 feed
    下更明显。这里按实际周期分钟数动态算，"1d" 之类未收录的周期按 1 根/天
    退化（回看天数直接约等于要求的根数）。
    """
    minutes = _TIMEFRAME_MINUTES.get(timeframe)
    if not minutes:
        return 1
    return max(1, _RTH_MINUTES_PER_DAY // minutes)


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
        bars_per_day = _bars_per_trading_day(self._cfg.timeframe)
        lookback_days = max(2, n_bars // bars_per_day + 3)
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
            raw = resp.json().get("bars") or []
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
