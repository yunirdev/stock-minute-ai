"""
portfolio.py
In-memory portfolio state with DuckDB-backed persistence.

Tracks cash, positions, fills, and equity snapshots.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from datetime import datetime, timezone
from typing import Dict

import duckdb

from .config import TradingConfig
from .models import Fill, Position, Side, utc_now

logger = logging.getLogger(__name__)


class Portfolio:

    def __init__(self, config: TradingConfig) -> None:
        self._initial_capital = float(config.initial_capital)
        self._cash: float = self._initial_capital
        self._positions: Dict[str, Position] = {}
        self._realized_pnl: float = 0.0
        self._db_path = config.db_path
        self._init_db()
        self._restore_from_fills()
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
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS portfolio_reconciliation_baselines (
                baseline_id TEXT PRIMARY KEY,
                observed_at TIMESTAMPTZ,
                broker_positions_json TEXT,
                broker_cash DOUBLE,
                reason TEXT,
                evidence_json TEXT,
                created_at TIMESTAMPTZ
            )
            """
        )
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def _apply_position_delta(
        self,
        *,
        symbol: str,
        side: Side,
        qty: float,
        price: float,
        fee: float,
    ) -> None:
        if side == Side.BUY:
            self._cash -= qty * price + fee
            if symbol in self._positions:
                pos = self._positions[symbol]
                new_qty = pos.qty + qty
                pos.avg_entry_px = (
                    pos.avg_entry_px * pos.qty + price * qty
                ) / new_qty
                pos.qty = new_qty
                pos.last_updated = utc_now()
            else:
                self._positions[symbol] = Position(
                    symbol=symbol,
                    qty=qty,
                    avg_entry_px=price,
                )
            return

        self._cash += qty * price - fee
        if symbol in self._positions:
            pos = self._positions[symbol]
            trade_pnl = (price - pos.avg_entry_px) * qty
            pos.realized_pnl += trade_pnl
            self._realized_pnl += trade_pnl
            pos.qty -= qty
            pos.last_updated = utc_now()
            if pos.qty <= 0:
                del self._positions[symbol]

    def _restore_from_fills(self) -> None:
        """Rebuild in-memory positions from durable incremental fill rows."""
        self._cash = self._initial_capital
        self._positions = {}
        self._realized_pnl = 0.0
        conn = self._connect()
        try:
            baseline = conn.execute(
                """
                SELECT observed_at, broker_positions_json, broker_cash
                FROM portfolio_reconciliation_baselines
                ORDER BY observed_at DESC, baseline_id DESC
                LIMIT 1
                """
            ).fetchone()
            if baseline is None:
                rows = conn.execute(
                    """
                    SELECT symbol, side, filled_qty, avg_price, fee
                    FROM fills
                    ORDER BY fill_time
                    """
                ).fetchall()
            else:
                observed_at, positions_json, broker_cash = baseline
                self._cash = float(broker_cash)
                for item in json.loads(positions_json or "[]"):
                    symbol = str(item["symbol"]).strip().upper()
                    quantity = float(item["qty"])
                    if not symbol or quantity == 0:
                        continue
                    self._positions[symbol] = Position(
                        symbol=symbol,
                        qty=quantity,
                        avg_entry_px=float(item["avg_entry_px"]),
                    )
                rows = conn.execute(
                    """
                    SELECT symbol, side, filled_qty, avg_price, fee
                    FROM fills
                    WHERE fill_time > ?
                    ORDER BY fill_time
                    """,
                    [observed_at],
                ).fetchall()
        finally:
            conn.close()
        for symbol, side, qty, price, fee in rows:
            qty_value = float(qty or 0.0)
            price_value = float(price or 0.0)
            if qty_value <= 0 or price_value <= 0:
                continue
            self._apply_position_delta(
                symbol=str(symbol),
                side=Side(str(side)),
                qty=qty_value,
                price=price_value,
                fee=float(fee or 0.0),
            )

    def record_broker_baseline(
        self,
        *,
        positions: list[Position],
        cash: float,
        observed_at: datetime,
        reason: str,
        evidence: dict,
    ) -> str:
        """Persist an explicit broker-authoritative migration baseline."""
        if observed_at.tzinfo is None or observed_at.utcoffset() is None:
            raise ValueError("PORTFOLIO_BASELINE_TIME_TZ_REQUIRED")
        cash_value = float(cash)
        if not math.isfinite(cash_value) or cash_value < 0:
            raise ValueError("PORTFOLIO_BASELINE_CASH_INVALID")
        reason_value = str(reason).strip()
        if not reason_value:
            raise ValueError("PORTFOLIO_BASELINE_REASON_REQUIRED")
        normalized: list[dict] = []
        seen: set[str] = set()
        for position in positions:
            symbol = str(position.symbol).strip().upper()
            quantity = float(position.qty)
            average = float(position.avg_entry_px)
            if (
                not symbol
                or symbol in seen
                or not math.isfinite(quantity)
                or not math.isfinite(average)
                or quantity == 0
                or average <= 0
            ):
                raise ValueError("PORTFOLIO_BASELINE_POSITION_INVALID")
            seen.add(symbol)
            normalized.append(
                {
                    "symbol": symbol,
                    "qty": quantity,
                    "avg_entry_px": average,
                }
            )
        normalized.sort(key=lambda item: item["symbol"])
        positions_json = json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        evidence_json = json.dumps(
            evidence,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=str,
        )
        raw = "|".join(
            (
                observed_at.astimezone(timezone.utc).isoformat(),
                positions_json,
                f"{cash_value:.8f}",
                reason_value,
                evidence_json,
            )
        )
        baseline_id = "portfolio-baseline-" + hashlib.sha256(
            raw.encode()
        ).hexdigest()[:24]
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO portfolio_reconciliation_baselines VALUES
                (?,?,?,?,?,?,?)
                ON CONFLICT (baseline_id) DO NOTHING
                """,
                [
                    baseline_id,
                    observed_at,
                    positions_json,
                    cash_value,
                    reason_value,
                    evidence_json,
                    datetime.now(timezone.utc),
                ],
            )
            conn.commit()
        finally:
            conn.close()
        self._restore_from_fills()
        return baseline_id

    def apply_fill(self, fill: Fill) -> float:
        """Apply only the newly observed cumulative fill quantity."""
        conn = self._connect()
        try:
            prior = conn.execute("SELECT COALESCE(SUM(filled_qty), 0) FROM fills WHERE order_id=?", [fill.order_id]).fetchone()[0]
        finally:
            conn.close()
        delta = max(0.0, float(fill.filled_qty) - float(prior))
        if delta <= 0:
            return 0.0
        qty = delta
        px = fill.avg_price
        symbol = fill.symbol

        self._apply_position_delta(
            symbol=symbol,
            side=fill.side,
            qty=qty,
            price=px,
            fee=fill.fee,
        )

        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO fills VALUES (?,?,?,?,?,?,?,?)",
                [fill.order_id, fill.intent_id, fill.symbol, fill.side.value,
                 delta, fill.avg_price, fill.fill_time, fill.fee],
            )
            conn.commit()
        finally:
            conn.close()
        logger.info("Fill applied: %s %s %.0f @ %.4f  cash=%.2f",
                    fill.side.value, symbol, qty, px, self._cash)
        return delta

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

    def snapshot_external_equity(self, equity: float, cash: float = 0.0,
                                 unrealized: float = 0.0, realized: float = 0.0) -> float:
        """写入由外部 broker（如 Alpaca）报告的权益快照。

        Alpaca 模式下账本以 broker 为准，本地不自算权益，但仍写 equity_snapshots
        表，让 monitor / 审计照常读取。"""
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO equity_snapshots VALUES (?,?,?,?,?)",
                [utc_now(), cash, equity, unrealized, realized],
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
