"""
trader/monitor_data.py
Pure data-access layer for the monitoring dashboard.

Reads the audit DuckDB (trade.duckdb) and the heartbeat sidecar file. No Streamlit,
no UI — so it can be reused by any front-end (Streamlit monitor, a future Marimo
notebook, or an API) and unit-tested directly.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_ROOT / ".env")

DB_PATH = str(_ROOT / os.getenv("TRADE_DB_PATH", "trade.duckdb"))
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


# ── Alpaca 实时账户权益（绕过 DuckDB，用于 monitor 总览）────────────────────────

_ALPACA_CACHE: dict = {"ts": None, "data": None}
_ALPACA_CACHE_TTL = 30   # seconds


def live_alpaca_equity() -> Optional[dict]:
    """直接调 Alpaca REST API 获取账户权益，绕过 DuckDB 历史数据。

    返回 {"equity": float, "cash": float, "buying_power": float}，
    或 None（API Key 未配置 / 网络超时 / 非 Alpaca broker）。
    结果缓存 30 秒，避免每次 UI 刷新都发起 HTTP 请求。
    """
    import json as _json
    import urllib.request as _urllib

    now = datetime.now(timezone.utc)
    if (
        _ALPACA_CACHE["ts"] is not None
        and (now - _ALPACA_CACHE["ts"]).total_seconds() < _ALPACA_CACHE_TTL
    ):
        return _ALPACA_CACHE["data"]

    api_key = os.getenv("ALPACA_API_KEY", "")
    secret = os.getenv("ALPACA_API_SECRET", "") or os.getenv("ALPACA_SECRET_KEY", "")
    broker_type = os.getenv("BROKER_TYPE", "alpaca_paper")

    if not api_key or not secret:
        _ALPACA_CACHE.update(ts=now, data=None)
        return None

    base = (
        "https://api.alpaca.markets"
        if broker_type == "alpaca_live"
        else "https://paper-api.alpaca.markets"
    )
    try:
        req = _urllib.Request(
            f"{base}/v2/account",
            headers={
                "APCA-API-KEY-ID": api_key,
                "APCA-API-SECRET-KEY": secret,
            },
        )
        with _urllib.urlopen(req, timeout=5) as resp:
            data = _json.loads(resp.read())
        result: dict = {
            "equity": float(data.get("equity", 0)),
            "cash": float(data.get("cash", 0)),
            "buying_power": float(data.get("buying_power", 0)),
        }
        _ALPACA_CACHE.update(ts=now, data=result)
        return result
    except Exception:
        _ALPACA_CACHE.update(ts=now, data=None)
        return None
