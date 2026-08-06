"""Read-only monitoring view for daily research and Runtime status."""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import duckdb

from .runtime_status import DEFAULT_STATUS_PATH, read_runtime_status

_EMPTY_RESEARCH: dict[str, Any] = {"run": None, "items": []}


def daily_research_monitor(db_path: str) -> dict[str, Any]:
    """Polled every few seconds by the dashboard — see db_query() in
    monitor_data.py for why this retries on IOException instead of holding
    a connection open across calls."""
    source = Path(db_path)
    if not source.exists():
        return dict(_EMPTY_RESEARCH)
    for attempt in range(3):
        try:
            return _read_daily_research(source)
        except duckdb.IOException:
            if attempt == 2:
                return dict(_EMPTY_RESEARCH)
            time.sleep(0.15)
        except Exception:
            return dict(_EMPTY_RESEARCH)
    return dict(_EMPTY_RESEARCH)


def _read_daily_research(source: Path) -> dict[str, Any]:
    con = duckdb.connect(str(source), read_only=True)
    try:
        tables = {
            str(row[0])
            for row in con.execute("SHOW TABLES").fetchall()
        }
        if "daily_research_runs" not in tables:
            return dict(_EMPTY_RESEARCH)
        run_row = con.execute(
            """
            SELECT run_id, trading_date, status, provider, model,
                   total_symbols, completed_symbols, failed_symbols,
                   started_at, completed_at, error_code, data_cutoff
            FROM daily_research_runs
            ORDER BY started_at DESC LIMIT 1
            """
        ).fetchone()
        if run_row is None:
            return dict(_EMPTY_RESEARCH)
        item_rows = con.execute(
            """
            SELECT symbol, rank, screening_score, screening_status, status,
                   recommendation, ai_score, confidence, thesis, risks_json,
                   error_code, completed_at, complexity
            FROM daily_research_items
            WHERE run_id=? ORDER BY rank, symbol
            """,
            [run_row[0]],
        ).fetchall()
    finally:
        con.close()
    run = {
        "run_id": run_row[0],
        "trading_date": run_row[1],
        "status": run_row[2],
        "provider": run_row[3],
        "model": run_row[4],
        "total_symbols": run_row[5],
        "completed_symbols": run_row[6],
        "failed_symbols": run_row[7],
        "started_at": _iso(run_row[8]),
        "completed_at": _iso(run_row[9]),
        "error_code": run_row[10] or "",
        "data_cutoff": _iso(run_row[11]),
    }
    items = [
        {
            "symbol": row[0],
            "rank": row[1],
            "screening_score": row[2],
            "screening_status": row[3],
            "status": row[4],
            "recommendation": row[5],
            "ai_score": row[6],
            "confidence": row[7],
            "thesis": row[8] or "",
            "risks": _loads(row[9], []),
            "error_code": row[10] or "",
            "completed_at": _iso(row[11]),
            "complexity": row[12] or "",
        }
        for row in item_rows
    ]
    return {"run": run, "items": items}


def list_debate_records(
    db_path: str,
    *,
    symbol: str = "",
    days: int | None = None,
    limit: int = 300,
) -> list[dict[str, Any]]:
    """研究档案——跨全部批次查历史记录，不像 daily_research_monitor() 只看
    最新一批。给"点标的名看完整多空辩论"这个入口用。"""
    source = Path(db_path)
    if not source.exists():
        return []
    for attempt in range(3):
        try:
            return _read_debate_records(source, symbol=symbol, days=days, limit=limit)
        except duckdb.IOException:
            if attempt == 2:
                return []
            time.sleep(0.15)
        except Exception:
            return []
    return []


def _read_debate_records(
    source: Path, *, symbol: str, days: int | None, limit: int
) -> list[dict[str, Any]]:
    con = duckdb.connect(str(source), read_only=True)
    try:
        clauses = ["status = 'COMPLETED'"]
        params: list[Any] = []
        clean_symbol = symbol.strip().upper()
        if clean_symbol:
            clauses.append("symbol = ?")
            params.append(clean_symbol)
        where = " AND ".join(clauses)
        rows = con.execute(
            f"""
            SELECT trading_date, symbol, recommendation, ai_score, thesis,
                   raw_json, completed_at, run_id
            FROM daily_research_items
            WHERE {where}
            ORDER BY completed_at DESC
            LIMIT ?
            """,
            [*params, max(1, int(limit))],
        ).fetchall()
    finally:
        con.close()
    cutoff = None
    if days is not None and days > 0:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    records = []
    for row in rows:
        completed_at = row[6]
        if cutoff is not None and completed_at is not None:
            aware = completed_at if completed_at.tzinfo else completed_at.replace(tzinfo=timezone.utc)
            if aware < cutoff:
                continue
        state: dict[str, Any] = {}
        try:
            raw = json.loads(row[5] or "{}")
            state = raw.get("state") if isinstance(raw, dict) else {}
        except Exception:
            state = {}
        debate = str((state or {}).get("investment_plan") or "").strip()
        records.append(
            {
                "trading_date": row[0],
                "symbol": row[1],
                "recommendation": row[2],
                "ai_score": row[3],
                "thesis": row[4] or "",
                "debate": debate,
                "completed_at": _iso(completed_at),
                "run_id": row[7],
            }
        )
    return records


def live_monitor_snapshot(
    ai_db_path: str,
    runtime_status_path: Path | str = DEFAULT_STATUS_PATH,
) -> dict[str, Any]:
    return {
        "read_at": datetime.now(timezone.utc).isoformat(),
        "research": daily_research_monitor(ai_db_path),
        "runtime": read_runtime_status(runtime_status_path),
    }


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _loads(value: Any, fallback: Any) -> Any:
    if not isinstance(value, str):
        return value if value is not None else fallback
    try:
        return json.loads(value)
    except Exception:
        return fallback
