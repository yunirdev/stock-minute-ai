"""
portfolio.py
In-memory portfolio state with DuckDB-backed persistence.

Tracks cash, positions, fills, and equity snapshots.
"""
from __future__ import annotations

import logging
import time
from typing import Dict

import duckdb

from .config import TradingConfig
from .models import Fill, Position, Side, utc_now

logger = logging.getLogger(__name__)


class Portfolio:

    def __init__(self, config: TradingConfig) -> None:
        self._cash: float = config.initial_capital
        self._positions: Dict[str, Position] = {}
        self._realized_pnl: float = 0.0
        self._db_path = config.db_path
        self._init_db()
        logger.info("Portfolio 初始化: 本金 %.2f, db=%s",
                    config.initial_capital, config.db_path)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _connect(self):
        """Open a fresh DuckDB connection, retrying on transient lock errors."""
        for _attempt in range(5):
            try:
                return duckdb.connect(self._db_path)
            except Exception:
                if _attempt == 4:
                    raise
                time.sleep(0.1 * (_attempt + 1))

    def _init_db(self) -> None:
        conn = self._connect()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fills (
                order_id    TEXT,
                intent_id   TEXT,
                symbol      TEXT,
                side        TEXT,
                filled_qty  DOUBLE,
                avg_price   DOUBLE,
                fill_time   TIMESTAMP,
                fee         DOUBLE
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS equity_snapshots (
                ts              TIMESTAMP,
                cash            DOUBLE,
                total_equity    DOUBLE,
                unrealized_pnl  DOUBLE,
                realized_pnl    DOUBLE
            )
        """)
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def apply_fill(self, fill: Fill) -> None:
        """Update cash and position based on a confirmed fill."""
        qty = fill.filled_qty
        px = fill.avg_price
        symbol = fill.symbol

        if fill.side == Side.BUY:
            cost = qty * px + fill.fee
            self._cash -= cost
            if symbol in self._positions:
                pos = self._positions[symbol]
                new_qty = pos.qty + qty
                pos.avg_entry_px = (pos.avg_entry_px * pos.qty + px * qty) / new_qty
                pos.qty = new_qty
                pos.last_updated = utc_now()
            else:
                self._positions[symbol] = Position(
                    symbol=symbol,
                    qty=qty,
                    avg_entry_px=px,
                )
        else:  # SELL
            proceeds = qty * px - fill.fee
            self._cash += proceeds
            if symbol in self._positions:
                pos = self._positions[symbol]
                trade_pnl = (px - pos.avg_entry_px) * qty
                pos.realized_pnl += trade_pnl
                self._realized_pnl += trade_pnl
                pos.qty -= qty
                pos.last_updated = utc_now()
                if pos.qty <= 0:
                    del self._positions[symbol]

        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO fills VALUES (?,?,?,?,?,?,?,?)",
                [fill.order_id, fill.intent_id, fill.symbol, fill.side.value,
                 fill.filled_qty, fill.avg_price, fill.fill_time, fill.fee],
            )
            conn.commit()
        finally:
            conn.close()
        logger.info("Fill applied: %s %s %.0f @ %.4f  cash=%.2f",
                    fill.side.value, symbol, qty, px, self._cash)

    def update_market_prices(self, prices: Dict[str, float]) -> None:
        """Refresh unrealized PnL from latest prices."""
        for symbol, price in prices.items():
            if symbol in self._positions:
                pos = self._positions[symbol]
                pos.unrealized_pnl = (price - pos.avg_entry_px) * pos.qty
                pos.last_updated = utc_now()

    def _position_market_value(self, prices: Dict[str, float]) -> float:
        """Return marked-to-market position value."""
        total = 0.0
        for symbol, pos in self._positions.items():
            price = prices.get(symbol, pos.avg_entry_px)
            total += pos.qty * price
        return total

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get_equity(self, prices: Dict[str, float]) -> float:
        self.update_market_prices(prices)
        return self._cash + self._position_market_value(prices)

    def snapshot_equity(self, prices: Dict[str, float]) -> float:
        """Compute equity and persist a snapshot row."""
        self.update_market_prices(prices)
        unrealized = sum(p.unrealized_pnl for p in self._positions.values())
        equity = self._cash + self._position_market_value(prices)
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO equity_snapshots VALUES (?,?,?,?,?)",
                [utc_now(), self._cash, equity,
                 unrealized, self._realized_pnl],
            )
            conn.commit()
        finally:
            conn.close()
        return equity

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def positions(self) -> Dict[str, Position]:
        return self._positions

    @property
    def realized_pnl(self) -> float:
        return self._realized_pnl

    def close(self) -> None:
        pass  # No persistent connection to close
