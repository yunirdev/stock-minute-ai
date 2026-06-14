"""
data_feed_yfinance.py
Market data via Yahoo Finance (yfinance).

Free, no API key required. Suitable for development and paper trading.
"""
from __future__ import annotations

import logging
import socket
from datetime import datetime, timezone
from typing import Dict, List

import pandas as pd

from .config import TradingConfig
from .models import Bar

logger = logging.getLogger(__name__)

# Global socket timeout — prevents yfinance / urllib3 from blocking forever
# when an underlying connection stalls (which would otherwise hang the
# scheduler tick because daemon threads cannot be cancelled cleanly).
socket.setdefaulttimeout(20)

_TF_MAP = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "1d": "1d",
}

# yfinance requires a minimum period for each interval
_PERIOD_MAP = {
    "1m":  "7d",
    "5m":  "60d",
    "15m": "60d",
    "30m": "60d",
    "1h":  "730d",
    "1d":  "5y",
}


class YFinanceDataFeed:
    """Fetches bars and latest prices using Yahoo Finance (yfinance)."""

    def __init__(self, config: TradingConfig) -> None:
        import yfinance as yf  # noqa: F401 — ensure installed
        self._cfg = config
        self._yf_tf = _TF_MAP.get(config.timeframe, "5m")
        self._period = _PERIOD_MAP.get(config.timeframe, "60d")

    def fetch_bars(self, symbol: str, n_bars: int = 120) -> List[Bar]:
        """Return up to *n_bars* recent bars for *symbol*.

        Runs the actual yfinance call in a *daemon* thread; if it does not
        complete within the timeout window the thread is abandoned (rather
        than waited on inside an executor's __exit__) so the scheduler tick
        can never be blocked indefinitely.
        """
        import threading
        import yfinance as yf

        holder: Dict[str, object] = {"df": None, "err": None}

        def _do_fetch() -> None:
            try:
                ticker = yf.Ticker(symbol)
                holder["df"] = ticker.history(
                    period=self._period,
                    interval=self._yf_tf,
                    auto_adjust=True,
                    prepost=False,
                )
            except Exception as exc:  # noqa: BLE001
                holder["err"] = exc

        th = threading.Thread(target=_do_fetch, daemon=True, name=f"yf-bars-{symbol}")
        th.start()
        th.join(timeout=25)
        if th.is_alive():
            logger.warning("YFinanceDataFeed fetch_bars %s: timeout, abandoning thread", symbol)
            return []
        if holder["err"] is not None:
            logger.error("YFinanceDataFeed fetch_bars %s: %s", symbol, holder["err"])
            return []
        df = holder["df"]

        if df is None or df.empty:
            logger.warning("YFinanceDataFeed: no data for %s", symbol)
            return []

        df = df.tail(n_bars)
        bars: List[Bar] = []
        for ts, row in df.iterrows():
            try:
                dt: datetime = ts.to_pydatetime()
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                bars.append(Bar(
                    symbol=symbol,
                    timestamp=dt,
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=float(row["Volume"]),
                ))
            except Exception:
                continue
        logger.debug("YFinanceDataFeed fetch_bars %s: got %d bars", symbol, len(bars))
        return bars

    def get_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Return {symbol: latest_close} using fast_info, with per-symbol timeout.

        Uses *daemon* threads instead of a ThreadPoolExecutor context manager so
        that a stuck yfinance call (e.g. socket hang inside urllib3) cannot
        block the scheduler tick. Hung threads are simply abandoned; the next
        tick will retry.
        """
        if not symbols:
            return {}
        import threading
        import yfinance as yf

        result: Dict[str, float] = {}
        result_lock = threading.Lock()

        def _fetch_one(symbol: str) -> None:
            try:
                t = yf.Ticker(symbol)
                price = getattr(t.fast_info, "last_price", None)
                if price is None:
                    hist = t.history(period="1d", interval="1m", auto_adjust=True)
                    price = float(hist["Close"].iloc[-1]) if not hist.empty else None
                if price is not None:
                    with result_lock:
                        result[symbol] = float(price)
            except Exception as exc:
                logger.error("YFinanceDataFeed get_latest_prices %s: %s", symbol, exc)

        threads = [
            threading.Thread(target=_fetch_one, args=(s,), daemon=True, name=f"yf-price-{s}")
            for s in symbols
        ]
        for th in threads:
            th.start()
        for th in threads:
            th.join(timeout=15)
            if th.is_alive():
                logger.warning("get_latest_prices: thread %s timed out, abandoning", th.name)
        return result
