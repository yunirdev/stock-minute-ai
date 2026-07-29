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
from .models import OrderIntent, RiskVerdict, Signal, TradePlan

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
            CREATE TABLE IF NOT EXISTS plan_risk_events (
                ts          TIMESTAMPTZ,
                plan_id     TEXT,
                decision_id TEXT,
                symbol      TEXT,
                side        TEXT,
                action      TEXT,
                phase       TEXT,
                verdict     TEXT,
                reason      TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ai_safety_events (
                ts TIMESTAMPTZ, symbol TEXT, plan_id TEXT, score DOUBLE,
                run_id TEXT, created_at TIMESTAMPTZ, age_seconds DOUBLE,
                reason_code TEXT, message TEXT, broker_type TEXT, auto_trade_paper BOOLEAN
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reconciliation_reports (
                ts TIMESTAMPTZ, ok BOOLEAN, open_orders INTEGER, positions INTEGER,
                fills_seen INTEGER, unexplained_orders TEXT, unexplained_positions TEXT, errors TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS strategy_decisions (
                decision_id TEXT PRIMARY KEY, universe_version TEXT, symbol TEXT,
                strategy TEXT, strategy_version TEXT, side TEXT, confidence DOUBLE,
                market_regime TEXT, candidate_source TEXT, evidence TEXT,
                ai_advisory_run_id TEXT, strategy_statistics_id TEXT,
                data_version TEXT, reason_codes TEXT, rejected_alternatives TEXT,
                created_at TIMESTAMPTZ, valid_until TIMESTAMPTZ
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS decision_plan_links (
                decision_id TEXT, plan_id TEXT UNIQUE, created_at TIMESTAMPTZ
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS heartbeat (
                ts          TIMESTAMP,
                tick_count  INTEGER,
                equity      DOUBLE
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trade_plans (
                plan_id     TEXT PRIMARY KEY,
                symbol      TEXT,
                action      TEXT,
                side        TEXT,
                qty         DOUBLE,
                entry_price DOUBLE,
                stop_loss   DOUBLE,
                take_profit DOUBLE,
                confidence  DOUBLE,
                rationale   TEXT,
                status      TEXT,
                created_at  TIMESTAMPTZ,
                updated_at  TIMESTAMPTZ
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

    def log_plan_risk_event(
        self,
        plan: TradePlan,
        verdict: RiskVerdict,
        *,
        phase: str,
    ) -> None:
        """Persist a plan-linked deterministic risk verdict for traceability."""
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO plan_risk_events VALUES (?,?,?,?,?,?,?,?,?)",
                [
                    datetime.now(timezone.utc),
                    plan.plan_id,
                    plan.metadata.get("decision_id", ""),
                    plan.symbol,
                    plan.side.value,
                    plan.action,
                    phase,
                    "APPROVED" if verdict.approved else "BLOCKED",
                    verdict.reason,
                ],
            )
            conn.commit()
        finally:
            conn.close()

    def log_ai_safety_event(self, plan: TradePlan, result, config: TradingConfig) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO ai_safety_events VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                [datetime.now(timezone.utc), plan.symbol, plan.plan_id, result.score,
                 result.run_id, None, result.age_seconds, result.reason_code,
                 result.message, config.broker_type, config.auto_trade_paper],
            )
            conn.commit()
        finally:
            conn.close()
    def log_reconciliation(self, report) -> None:
        conn = self._connect()
        try:
            conn.execute("INSERT INTO reconciliation_reports VALUES (?,?,?,?,?,?,?,?)", [
                report.created_at, report.ok, report.open_orders, report.positions, report.fills_seen,
                json.dumps(report.unexplained_orders), json.dumps(report.unexplained_positions), json.dumps(report.errors)])
            conn.commit()
        finally:
            conn.close()

    def log_strategy_decision(self, decision) -> None:
        conn = self._connect()
        try:
            conn.execute("""
                INSERT INTO strategy_decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT (decision_id) DO NOTHING
            """, [decision.decision_id, decision.universe_version, decision.symbol,
                   decision.strategy, decision.strategy_version, decision.side.value,
                   decision.confidence, decision.market_regime, decision.candidate_source,
                   json.dumps(decision.evidence, default=str), decision.ai_advisory_run_id,
                   decision.strategy_statistics_id, decision.data_version,
                   json.dumps(decision.reason_codes), json.dumps(decision.rejected_alternatives),
                   decision.created_at, decision.valid_until])
            conn.commit()
        finally:
            conn.close()
    def link_decision_plan(self, decision_id: str, plan_id: str) -> None:
        conn = self._connect()
        try:
            conn.execute("INSERT INTO decision_plan_links VALUES (?,?,?) ON CONFLICT (plan_id) DO NOTHING",
                         [decision_id, plan_id, datetime.now(timezone.utc)])
            conn.commit()
        finally:
            conn.close()
    def log_heartbeat(self, tick_count: int, equity: float) -> None:
        """Overwrite the single-row heartbeat table on every tick."""
        now = datetime.now(timezone.utc)
        # Write to DuckDB (best-effort — may fail if another process holds the lock)
        try:
            conn = self._connect()
            try:
                conn.execute("DELETE FROM heartbeat")
                conn.execute(
                    "INSERT INTO heartbeat VALUES (?,?,?)",
                    [now, tick_count, equity],
                )
                conn.commit()
            finally:
                conn.close()
        except Exception as _exc:
            # WARNING, not debug: HeartbeatWatchdog reads this table to detect a
            # hung engine. A silent debug-level swallow here previously let the
            # DuckDB write fail every tick for a full day undetected, while the
            # watchdog kept firing CRITICAL staleness alerts against a table
            # nobody could see was never being updated.
            logger.warning("heartbeat DuckDB write skipped: %s", _exc)
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

    def log_trade_plan(self, plan: TradePlan) -> None:
        """写入（或更新）trade_plans 表。plan_id 已存在时只更新 status/updated_at。"""
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO trade_plans VALUES (?,?,?,?,?,?,?,?,?,?,?,NOW(),NOW())
                ON CONFLICT (plan_id) DO UPDATE SET status=excluded.status, updated_at=NOW()
                """,
                [plan.plan_id, plan.symbol, plan.action, plan.side.value,
                 plan.qty, plan.entry_price, plan.stop_loss, plan.take_profit,
                 plan.confidence, plan.rationale, plan.status],
            )
            conn.commit()
        finally:
            conn.close()

    def close(self) -> None:
        pass  # No persistent connection to close
