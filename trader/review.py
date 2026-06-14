"""
review.py
复盘归因模块：从账本读取当日 equity/fills，生成 ReviewReport。
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import duckdb

from .models import ReviewReport, new_id, utc_now

logger = logging.getLogger(__name__)


class SimpleReviewer:
    """实现 Reviewer Protocol —— 从 DuckDB 账本生成简单盘后复盘报告。"""

    def __init__(self, db_path: str = "trade.duckdb") -> None:
        self._db_path = db_path

    def review(self, period: str = "daily", as_of: datetime | None = None) -> ReviewReport:
        as_of = as_of or utc_now()
        try:
            conn = duckdb.connect(self._db_path, read_only=True)
            pnl, trades, equity_start, equity_end = self._query(conn, period, as_of)
            conn.close()
        except Exception as exc:
            logger.warning("review 读取账本失败: %s", exc)
            return self._empty_report(period, as_of)

        market_summary = (
            f"期间权益: {equity_start:.2f} → {equity_end:.2f}，"
            f"净损益: {pnl:+.2f}"
        )
        attribution: Dict[str, Any] = {
            "equity_start": equity_start,
            "equity_end": equity_end,
            "realized_pnl": pnl,
            "trade_count": len(trades),
        }
        logger.info("复盘 [%s] pnl=%.2f trades=%d", period, pnl, len(trades))
        return ReviewReport(
            report_id=new_id(),
            period=period,
            market_summary=market_summary,
            portfolio_pnl=pnl,
            attribution=attribution,
            trades=trades,
            created_at=utc_now(),
        )

    def _query(self, conn, period: str, as_of: datetime):
        if period == "daily":
            since = as_of.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == "weekly":
            since = as_of - timedelta(days=7)
        else:
            since = as_of - timedelta(days=1)

        rows = conn.execute(
            "SELECT symbol, side, filled_qty, avg_price, fill_time "
            "FROM fills WHERE fill_time >= ? ORDER BY fill_time",
            [since],
        ).fetchall()
        trades = [
            {"symbol": r[0], "side": r[1], "qty": r[2], "price": r[3], "time": str(r[4])}
            for r in rows
        ]

        eq_rows = conn.execute(
            "SELECT total_equity FROM equity_snapshots WHERE ts >= ? ORDER BY ts",
            [since],
        ).fetchall()
        equity_start = float(eq_rows[0][0]) if eq_rows else 0.0
        equity_end = float(eq_rows[-1][0]) if eq_rows else 0.0
        pnl = equity_end - equity_start
        return pnl, trades, equity_start, equity_end

    @staticmethod
    def _empty_report(period: str, as_of: datetime) -> ReviewReport:
        return ReviewReport(
            report_id=new_id(),
            period=period,
            market_summary="无数据",
            portfolio_pnl=0.0,
            attribution={},
            trades=[],
            created_at=utc_now(),
        )
