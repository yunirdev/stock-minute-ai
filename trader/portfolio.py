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

    _LEGACY_TS_SUFFIX = "__legacy_naive_ts"

    def _migrate_naive_timestamp_table(
        self, conn, *, table: str, converted_columns: str, ts_column: str,
    ) -> dict:
        """Same fix as AuditLog's heartbeat/signals/orders/risk_events migration:
        a naive TIMESTAMP column silently stores a tz-aware Python datetime as
        *local* wall-clock time, dropping the UTC offset. `fills.fill_time` and
        `equity_snapshots.ts` had the same bug — currently it happens not to
        surface visibly because DuckDB applies the same implicit local-time
        conversion to query bind parameters too, so filtering "since" with a
        tz-aware value still matches, but it's one code path away from silently
        breaking (different host timezone, a reader that treats the column as
        real UTC, etc). Migrate in place, preserving data."""
        pending: dict = {}
        row = conn.execute(
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_name = ? AND column_name = ?",
            [table, ts_column],
        ).fetchone()
        if row and row[0] != "TIMESTAMP WITH TIME ZONE":
            conn.execute(f"ALTER TABLE {table} RENAME TO {table}{self._LEGACY_TS_SUFFIX}")
            pending[table] = converted_columns
        return pending

    def _finish_naive_timestamp_migration(self, conn, table: str, pending: dict) -> None:
        converted_columns = pending.pop(table, None)
        if converted_columns is None:
            return
        legacy = f"{table}{self._LEGACY_TS_SUFFIX}"
        conn.execute(f"INSERT INTO {table} SELECT {converted_columns} FROM {legacy}")
        conn.execute(f"DROP TABLE {legacy}")
        logger.info("portfolio: migrated %s to TIMESTAMPTZ (data preserved)", table)

    def _init_db(self) -> None:
        conn = self._connect()
        pending = self._migrate_naive_timestamp_table(
            conn,
            table="fills",
            converted_columns=(
                "order_id, intent_id, symbol, side, filled_qty, avg_price, "
                "timezone(current_setting('TimeZone'), fill_time), fee"
            ),
            ts_column="fill_time",
        )
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fills (
                order_id    TEXT,
                intent_id   TEXT,
                symbol      TEXT,
                side        TEXT,
                filled_qty  DOUBLE,
                avg_price   DOUBLE,
                fill_time   TIMESTAMPTZ,
                fee         DOUBLE
            )
        """)
        self._finish_naive_timestamp_migration(conn, "fills", pending)

        pending = self._migrate_naive_timestamp_table(
            conn,
            table="equity_snapshots",
            converted_columns=(
                "timezone(current_setting('TimeZone'), ts), cash, total_equity, "
                "unrealized_pnl, realized_pnl"
            ),
            ts_column="ts",
        )
        conn.execute("""
            CREATE TABLE IF NOT EXISTS equity_snapshots (
                ts              TIMESTAMPTZ,
                cash            DOUBLE,
                total_equity    DOUBLE,
                unrealized_pnl  DOUBLE,
                realized_pnl    DOUBLE
            )
        """)
        self._finish_naive_timestamp_migration(conn, "equity_snapshots", pending)
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
        """多空对称的持仓更新。Position.qty 有符号（正=多头，负=空头，见
        models.py 的约定）；side/qty 只描述这一笔成交本身（方向+数量，都是
        正的 qty），仓位方向变化完全靠"这笔成交跟现有仓位是同向还是反向"来
        判断，不是靠 side==BUY/SELL 本身——BUY 既可能是开多/加多，也可能是
        回补空头；SELL 既可能是平多/减多，也可能是开空/加空。

        现金流公式不用跟着方向分支：BUY 花钱、SELL 收钱，跟这笔成交是在加
        仓还是平仓、多头还是空头都无关（卖空同样是先收到现金，回补时再付
        出去）。
        """
        if side == Side.BUY:
            self._cash -= qty * price + fee
            delta = qty
        else:
            self._cash += qty * price - fee
            delta = -qty

        pos = self._positions.get(symbol)
        prior_qty = pos.qty if pos is not None else 0.0
        new_qty = prior_qty + delta
        same_direction_or_flat = prior_qty == 0.0 or (prior_qty > 0) == (delta > 0)

        if same_direction_or_flat:
            # 开新仓，或者顺着原方向加仓（多头再买、空头再卖）——按加权平均
            # 结转成本价，跟原来 BUY-only 那支逻辑完全一致，只是不再要求
            # prior_qty 一定是正数。
            if pos is None:
                self._positions[symbol] = Position(
                    symbol=symbol, qty=new_qty, avg_entry_px=price,
                )
            else:
                pos.avg_entry_px = (
                    pos.avg_entry_px * abs(prior_qty) + price * qty
                ) / abs(new_qty)
                pos.qty = new_qty
                pos.last_updated = utc_now()
            return

        # 反着原方向的成交——多头被卖出，或空头被买入回补。先把能对冲掉的
        # 部分结算实现盈亏；如果这笔量超过现有仓位（穿仓），原方向仓位在
        # 这里已经全部结清，剩下的部分按当前价开一张全新的反方向仓位——
        # 不是原来那样"qty<=0 就直接删记录"，那样会把穿仓之后真实存在的
        # 反向仓位凭空丢掉。
        closing_qty = min(qty, abs(prior_qty))
        if prior_qty > 0:
            trade_pnl = (price - pos.avg_entry_px) * closing_qty
        else:
            trade_pnl = (pos.avg_entry_px - price) * closing_qty
        pos.realized_pnl += trade_pnl
        self._realized_pnl += trade_pnl

        if new_qty == 0:
            del self._positions[symbol]
        elif (prior_qty > 0) == (new_qty > 0):
            # 没穿仓，只是减仓——剩余部分成本价不变。
            pos.qty = new_qty
            pos.last_updated = utc_now()
        else:
            # 穿仓反向：原方向已经在上面全部结算完，剩下的部分是全新的反向
            # 仓位，成本价就是这笔成交价，不能沿用原来那个方向的成本价。
            self._positions[symbol] = Position(
                symbol=symbol, qty=new_qty, avg_entry_px=price,
            )

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
