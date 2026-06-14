"""
trader/monitor_data.py
Pure data-access layer for the monitoring dashboard.

Reads the audit DuckDB (trade.duckdb) and the heartbeat sidecar file. No Streamlit,
no UI — so it can be reused by any front-end (Streamlit monitor, a future Marimo
notebook, or an API) and unit-tested directly.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = str(_ROOT / "trade.duckdb")
HEARTBEAT_JSON = _ROOT / "logs" / "heartbeat.json"


def db_query(sql: str, params: list | None = None, db_path: str = DB_PATH) -> pd.DataFrame:
    """Run a read-only query against the audit DB. Returns empty frame on any error."""
    if not Path(db_path).exists():
        return pd.DataFrame()
    try:
        conn = duckdb.connect(db_path, read_only=True)
        df = conn.execute(sql, params or []).df()
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def heartbeat(db_path: str = DB_PATH, heartbeat_json: Path = HEARTBEAT_JSON) -> Optional[datetime]:
    """Latest engine heartbeat — JSON sidecar first (no DuckDB lock), then DB."""
    if heartbeat_json.exists():
        try:
            data = json.loads(heartbeat_json.read_text(encoding="utf-8"))
            return datetime.fromisoformat(data["ts"]).replace(tzinfo=timezone.utc)
        except Exception:
            pass
    df = db_query("SELECT ts FROM heartbeat LIMIT 1", db_path=db_path)
    if df.empty:
        return None
    return pd.to_datetime(df["ts"].iloc[0]).to_pydatetime().replace(tzinfo=timezone.utc)


def _since(hours: int) -> datetime:
    return datetime.now(timezone.utc) - timedelta(hours=hours)


def equity_df(hours: int, db_path: str = DB_PATH) -> pd.DataFrame:
    return db_query("SELECT * FROM equity_snapshots WHERE ts >= ? ORDER BY ts",
                    [_since(hours)], db_path)


def signals_df(hours: int, db_path: str = DB_PATH) -> pd.DataFrame:
    return db_query("SELECT * FROM signals WHERE signal_time >= ? ORDER BY signal_time DESC",
                    [_since(hours)], db_path)


def orders_df(hours: int, db_path: str = DB_PATH) -> pd.DataFrame:
    return db_query("SELECT * FROM orders WHERE created_at >= ? ORDER BY created_at DESC",
                    [_since(hours)], db_path)


def fills_df(hours: int, db_path: str = DB_PATH) -> pd.DataFrame:
    return db_query("SELECT * FROM fills WHERE fill_time >= ? ORDER BY fill_time DESC",
                    [_since(hours)], db_path)


def risk_events_df(hours: int, db_path: str = DB_PATH) -> pd.DataFrame:
    return db_query("SELECT * FROM risk_events WHERE ts >= ? ORDER BY ts DESC",
                    [_since(hours)], db_path)
