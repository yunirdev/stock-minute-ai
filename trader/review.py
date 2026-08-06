"""
review.py
复盘归因模块：从账本读取当日 equity/fills，生成 ReviewReport。
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

import duckdb

from .models import ReviewReport, new_id, utc_now

logger = logging.getLogger(__name__)


class SimpleReviewer:
    """实现 Reviewer —— 从 DuckDB 账本生成简单盘后复盘报告。"""

    def __init__(self, db_path: str = "trade.duckdb") -> None:
        self._db_path = db_path

    def review(self, period: str = "daily", as_of: datetime | None = None) -> ReviewReport:
        as_of = as_of or utc_now()
        # 账本文件压根不存在（全新部署、还从没写过一笔账）跟"文件存在但打不开/
        # 损坏/被锁"是两码事，不能用同一个 except 兜底成同一句"读取失败"——前者
        # 是合法的"还没有数据"状态，如实报个空复盘就行；后者才是需要读者知道
        # 的真故障。这里先把前者单独摘出来，read_only 打开之后的失败仍然原样
        # 抛出，不在这里吞——A DB read failure must not become a fabricated
        # "flat day, zero trades" report; let the caller's own except-and-skip
        # handler (runtime._maybe_daily_review) decide what to do with a real
        # failed review instead of silently faking one here.
        if not Path(self._db_path).exists():
            return self._empty_report(period)
        conn = duckdb.connect(self._db_path, read_only=True)
        try:
            pnl, trades, equity_start, equity_end = self._query(conn, period, as_of)
        finally:
            conn.close()

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

    @staticmethod
    def _empty_report(period: str) -> ReviewReport:
        # market_summary 特意写"尚无账本数据"而不是"期间权益 0→0，净损益
        # +0.00"——后者看起来跟一个真实的、验证过的零成交平淡日没法区分，
        # 这里要让读者一眼看出这是"还没有数据"，不是"算出来是零"。
        return ReviewReport(
            report_id=new_id(),
            period=period,
            market_summary="尚无账本数据（账本文件还未创建）",
            portfolio_pnl=0.0,
            attribution={
                "equity_start": 0.0,
                "equity_end": 0.0,
                "realized_pnl": 0.0,
                "trade_count": 0,
            },
            trades=[],
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
