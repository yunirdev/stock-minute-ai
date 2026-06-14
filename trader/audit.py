"""
audit.py
Append-only audit log stored in DuckDB.

Every signal, risk decision, order submission, and status change is written
here so there is a complete trace: signal → risk → order → fill.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from .config import TradingConfig
from .models import OrderIntent, RiskVerdict, Signal

logger = logging.getLogger(__name__)

# Sidecar file written by the engine every tick — readable by Streamlit without
# any DuckDB connection (avoids cross-process file-lock contention on Windows).
_HEARTBEAT_FILE = Path(__file__).resolve().parents[1] / "logs" / "heartbeat.json"
_HEARTBEAT_FILE.parent.mkdir(parents=True, exist_ok=True)


class AuditLog:

    def __init__(self, config: TradingConfig) -> None:
        self._db_path = config.db_path
        self._init_db()

    # ------------------------------------------------------------------
    # Internal helpers
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
            CREATE TABLE IF NOT EXISTS signals (
                signal_id   TEXT,
                symbol      TEXT,
                strategy    TEXT,
                side        TEXT,
                exec_price  DOUBLE,
                timeframe   TEXT,
                signal_time TIMESTAMP,
                bar_close   DOUBLE,
                metadata    TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                intent_id       TEXT,
                signal_id       TEXT,
                symbol          TEXT,
                side            TEXT,
                qty             DOUBLE,
                order_type      TEXT,
                limit_price     DOUBLE,
                risk_tag        TEXT,
                created_at      TIMESTAMP,
                broker_order_id TEXT,
                status          TEXT,
                updated_at      TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS risk_events (
                ts          TIMESTAMP,
                symbol      TEXT,
                strategy    TEXT,
                side        TEXT,
                verdict     TEXT,
                reason      TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS heartbeat (
                ts          TIMESTAMP,
                tick_count  INTEGER,
                equity      DOUBLE
            )
        """)
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

    def log_signal(self, s: Signal) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO signals VALUES (?,?,?,?,?,?,?,?,?)",
                [s.signal_id, s.symbol, s.strategy, s.side.value,
                 s.exec_price, s.timeframe, s.signal_time,
                 s.bar_close, json.dumps(s.metadata)],
            )
            conn.commit()
        finally:
            conn.close()

    def log_order(
        self,
        intent: OrderIntent,
        broker_order_id: str = "",
        status: str = "SUBMITTED",
    ) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                [intent.intent_id, intent.signal_id, intent.symbol,
                 intent.side.value, intent.qty, intent.order_type,
                 intent.limit_price, intent.risk_tag, intent.created_at,
                 broker_order_id, status, datetime.now(timezone.utc)],
            )
            conn.commit()
        finally:
            conn.close()

    def update_order_status(
        self,
        intent_id: str,
        status: str,
        broker_order_id: str = "",
    ) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE orders SET status=?, updated_at=?, broker_order_id=? "
                "WHERE intent_id=?",
                [status, datetime.now(timezone.utc), broker_order_id, intent_id],
            )
            conn.commit()
        finally:
            conn.close()

    def log_risk_event(self, signal: Signal, verdict: RiskVerdict) -> None:
        v = "APPROVED" if verdict.approved else "BLOCKED"
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO risk_events VALUES (?,?,?,?,?,?)",
                [datetime.now(timezone.utc), signal.symbol, signal.strategy,
                 signal.side.value, v, verdict.reason],
            )
            conn.commit()
        finally:
            conn.close()

    def log_heartbeat(self, tick_count: int, equity: float) -> None:
        """Overwrite the single-row heartbeat table on every tick."""
        now = datetime.now(timezone.utc)
        # Write to DuckDB (best-effort — may fail if another process holds the lock)
        try:
            conn = self._connect()
            conn.execute("DELETE FROM heartbeat")
            conn.execute(
                "INSERT INTO heartbeat VALUES (?,?,?)",
                [now, tick_count, equity],
            )
            conn.commit()
            conn.close()
        except Exception as _exc:
            logger.debug("heartbeat DuckDB write skipped: %s", _exc)
        # Always write JSON sidecar — no file locking, readable by Streamlit
        try:
            _HEARTBEAT_FILE.write_text(
                json.dumps({
                    "ts": now.isoformat(),
                    "tick_count": tick_count,
                    "equity": equity,
                }),
                encoding="utf-8",
            )
        except Exception as _exc:
            logger.debug("heartbeat JSON write failed: %s", _exc)

    def close(self) -> None:
        pass  # No persistent connection to close
