"""Read-only monitoring view for daily research and Runtime status."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

from .runtime_status import DEFAULT_STATUS_PATH, read_runtime_status


def daily_research_monitor(db_path: str) -> dict[str, Any]:
    source = Path(db_path)
    if not source.exists():
        return {"run": None, "items": []}
    try:
        con = duckdb.connect(str(source), read_only=True)
        tables = {
            str(row[0])
            for row in con.execute("SHOW TABLES").fetchall()
        }
        if "daily_research_runs" not in tables:
            con.close()
            return {"run": None, "items": []}
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
            con.close()
            return {"run": None, "items": []}
        item_rows = con.execute(
            """
            SELECT symbol, rank, screening_score, screening_status, status,
                   recommendation, ai_score, confidence, thesis, risks_json,
                   error_code, completed_at
            FROM daily_research_items
            WHERE run_id=? ORDER BY rank, symbol
            """,
            [run_row[0]],
        ).fetchall()
        con.close()
    except Exception:
        return {"run": None, "items": []}
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
        }
        for row in item_rows
    ]
    return {"run": run, "items": items}


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
