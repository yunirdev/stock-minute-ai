"""Immutable performance attribution snapshots for one PositionPlan episode."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb


class TradeEpisodeStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._migrate()

    def _connect(self, *, read_only: bool = False):
        return duckdb.connect(self.db_path, read_only=read_only)

    def _migrate(self) -> None:
        connection = self._connect()
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS trade_episode_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    episode_id TEXT,
                    position_plan_id TEXT,
                    snapshot_version INTEGER,
                    status TEXT,
                    symbol TEXT,
                    entry_quantity DOUBLE,
                    exit_quantity DOUBLE,
                    open_quantity DOUBLE,
                    average_entry_price DOUBLE,
                    realized_pnl DOUBLE,
                    adverse_slippage DOUBLE,
                    invalidation_event_count INTEGER,
                    adjustment_count INTEGER,
                    first_fill_at TIMESTAMPTZ,
                    last_fill_at TIMESTAMPTZ,
                    attribution_json TEXT,
                    created_at TIMESTAMPTZ,
                    UNIQUE(episode_id, snapshot_version)
                )
                """
            )
            connection.commit()
        finally:
            connection.close()

    def sync(self, position_plan_id: str) -> dict[str, Any]:
        connection = self._connect()
        try:
            head = connection.execute(
                """
                SELECT symbol, status, initial_quantity,
                       current_version
                FROM position_plan_heads
                WHERE position_plan_id=?
                """,
                [position_plan_id],
            ).fetchone()
            if head is None:
                raise KeyError(position_plan_id)
            has_orders = connection.execute(
                """
                SELECT 1 FROM information_schema.tables
                WHERE table_name='order_intents'
                """
            ).fetchone()
            if has_orders:
                fills = connection.execute(
                    """
                    SELECT f.order_id, f.side, f.applied_delta,
                           f.avg_price, f.created_at,
                           coalesce(o.limit_price, f.avg_price)
                    FROM position_plan_fill_events f
                    LEFT JOIN order_intents o
                      ON o.broker_order_id=f.order_id
                    WHERE f.position_plan_id=?
                    ORDER BY f.created_at, f.fill_event_id
                    """,
                    [position_plan_id],
                ).fetchall()
            else:
                fills = connection.execute(
                    """
                    SELECT order_id, side, applied_delta,
                           avg_price, created_at, avg_price
                    FROM position_plan_fill_events
                    WHERE position_plan_id=?
                    ORDER BY created_at, fill_event_id
                    """,
                    [position_plan_id],
                ).fetchall()
            current = connection.execute(
                """
                SELECT open_quantity, average_entry_price
                FROM position_plan_versions
                WHERE position_plan_id=? AND version=?
                """,
                [position_plan_id, int(head[3])],
            ).fetchone()
            event_count = self._count_if_table(
                connection,
                "invalidation_events",
                "position_plan_id",
                position_plan_id,
            )
            adjustment_count = self._count_if_table(
                connection,
                "position_adjustments",
                "position_plan_id",
                position_plan_id,
            )
            prior = connection.execute(
                """
                SELECT coalesce(max(snapshot_version), 0)
                FROM trade_episode_snapshots
                WHERE episode_id=?
                """,
                [position_plan_id],
            ).fetchone()[0]
        finally:
            connection.close()
        buys = [row for row in fills if str(row[1]) == "BUY"]
        sells = [row for row in fills if str(row[1]) == "SELL"]
        entry_quantity = sum(float(row[2]) for row in buys)
        exit_quantity = sum(float(row[2]) for row in sells)
        entry_value = sum(
            float(row[2]) * float(row[3]) for row in buys
        )
        average_entry = (
            entry_value / entry_quantity if entry_quantity else 0.0
        )
        realized = sum(
            (float(row[3]) - average_entry) * float(row[2])
            for row in sells
        )
        slippage = 0.0
        for row in fills:
            side = str(row[1])
            quantity = float(row[2])
            actual = float(row[3])
            planned = float(row[5])
            slippage += (
                (actual - planned) * quantity
                if side == "BUY"
                else (planned - actual) * quantity
            )
        attribution = {
            "fill_order_ids": [str(row[0]) for row in fills],
            "fill_count": len(fills),
            "position_plan_version": int(head[3]),
            "cross_day": (
                bool(fills)
                and fills[0][4].date() != fills[-1][4].date()
            ),
        }
        payload = {
            "episode_id": position_plan_id,
            "status": str(head[1]),
            "symbol": str(head[0]),
            "entry_quantity": entry_quantity,
            "exit_quantity": exit_quantity,
            "open_quantity": float(current[0]),
            "average_entry_price": (
                average_entry or float(current[1])
            ),
            "realized_pnl": realized,
            "adverse_slippage": slippage,
            "invalidation_event_count": event_count,
            "adjustment_count": adjustment_count,
            "first_fill_at": fills[0][4] if fills else None,
            "last_fill_at": fills[-1][4] if fills else None,
            "attribution": attribution,
        }
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        snapshot_id = "episode-snapshot-" + hashlib.sha256(
            canonical.encode()
        ).hexdigest()[:24]
        connection = self._connect()
        try:
            existing = connection.execute(
                """
                SELECT * FROM trade_episode_snapshots
                WHERE snapshot_id=?
                """,
                [snapshot_id],
            ).fetchone()
            if existing is None:
                version = int(prior) + 1
                connection.execute(
                    """
                    INSERT INTO trade_episode_snapshots VALUES
                    (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    [
                        snapshot_id,
                        position_plan_id,
                        position_plan_id,
                        version,
                        payload["status"],
                        payload["symbol"],
                        entry_quantity,
                        exit_quantity,
                        payload["open_quantity"],
                        payload["average_entry_price"],
                        realized,
                        slippage,
                        event_count,
                        adjustment_count,
                        payload["first_fill_at"],
                        payload["last_fill_at"],
                        json.dumps(attribution, separators=(",", ":")),
                        datetime.now(timezone.utc),
                    ],
                )
                connection.commit()
                payload["snapshot_version"] = version
            else:
                payload["snapshot_version"] = int(existing[3])
        finally:
            connection.close()
        payload["snapshot_id"] = snapshot_id
        return payload

    def latest(self, episode_id: str) -> dict[str, Any] | None:
        connection = self._connect(read_only=True)
        try:
            row = connection.execute(
                """
                SELECT * FROM trade_episode_snapshots
                WHERE episode_id=?
                ORDER BY snapshot_version DESC
                LIMIT 1
                """,
                [episode_id],
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        return {
            "snapshot_id": str(row[0]),
            "episode_id": str(row[1]),
            "snapshot_version": int(row[3]),
            "status": str(row[4]),
            "symbol": str(row[5]),
            "entry_quantity": float(row[6]),
            "exit_quantity": float(row[7]),
            "open_quantity": float(row[8]),
            "average_entry_price": float(row[9]),
            "realized_pnl": float(row[10]),
            "adverse_slippage": float(row[11]),
            "invalidation_event_count": int(row[12]),
            "adjustment_count": int(row[13]),
            "first_fill_at": row[14],
            "last_fill_at": row[15],
            "attribution": json.loads(row[16] or "{}"),
        }

    @staticmethod
    def _count_if_table(
        connection,
        table: str,
        column: str,
        value: str,
    ) -> int:
        exists = connection.execute(
            """
            SELECT 1 FROM information_schema.tables
            WHERE table_name=?
            """,
            [table],
        ).fetchone()
        if exists is None:
            return 0
        return int(
            connection.execute(
                f"SELECT count(*) FROM {table} WHERE {column}=?",
                [value],
            ).fetchone()[0]
        )
