"""
trader/data_cache.py
====================
全局行情数据缓存 — 本地 Parquet 文件持久化 + 内存缓存 + 按需增量更新。

设计原则:
  - 启动时：优先从本地 Parquet 文件加载（毫秒级，零网络请求）
  - 本地文件存在且未过期（<24h）→ 直接用，不访问网络
  - 本地文件不存在或已过期 → 从 yfinance 拉取并保存到本地
  - 运行时增量更新：只追加最新几根 bar，不重拉全量

本地文件位置: data/bars/{symbol}_{timeframe}.parquet
  例: data/bars/AAPL_30m.parquet

使用方式:
    from trader.data_cache import get_bars, warm_up

    warm_up(symbols=["AAPL", "SPY"], timeframes=["5m", "30m", "1d"])
    df = get_bars("AAPL", "30m")   # 立即返回，不阻塞
"""
from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 本地存储路径
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[1]
_BARS_DIR = _ROOT / "data" / "bars"
_BARS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 内存缓存
# ---------------------------------------------------------------------------
_CACHE: Dict[Tuple[str, str], pd.DataFrame] = {}
_CACHE_LOCK = threading.Lock()
_LAST_NETWORK_FETCH: Dict[Tuple[str, str], datetime] = {}

_WARM_THREAD: Optional[threading.Thread] = None
_WARM_DONE: bool = False

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
# yfinance 全量拉取周期（仅当本地文件不存在或过期时使用）
_YF_PERIOD: Dict[str, str] = {
    "1m":  "7d",
    "5m":  "60d",
    "15m": "60d",
    "30m": "60d",
    "1h":  "730d",
    "1d":  "max",
}

# 增量更新只拉最近这些天（UI 刷新时调用）
_INCR_PERIOD: Dict[str, str] = {
    "1m":  "1d",
    "5m":  "3d",
    "15m": "5d",
    "30m": "5d",
    "1h":  "10d",
    "1d":  "30d",
}

# 本地文件超过这个时间才重新从网络拉取（秒）
_FILE_MAX_AGE: Dict[str, int] = {
    "1m":  8 * 3600,    # 8 小时
    "5m":  24 * 3600,   # 24 小时
    "15m": 24 * 3600,
    "30m": 24 * 3600,
    "1h":  24 * 3600,
    "1d":  24 * 3600,
}

# 内存中增量更新的最小间隔（秒），避免同一 UI 刷新重复请求
_INCR_MIN_INTERVAL = 60


# ---------------------------------------------------------------------------
# 本地 Parquet 文件 I/O
# ---------------------------------------------------------------------------
def _parquet_path(symbol: str, timeframe: str) -> Path:
    return _BARS_DIR / f"{symbol}_{timeframe}.parquet"


def _save_to_disk(symbol: str, timeframe: str, df: pd.DataFrame) -> None:
    """保存 DataFrame 到本地 Parquet 文件。"""
    if df.empty:
        return
    path = _parquet_path(symbol, timeframe)
    try:
        out = df.copy()
        out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True)
        out["timestamp"] = out["timestamp_utc"]
        out.to_parquet(path, index=False, engine="pyarrow")
        logger.info("data_cache saved %s %s → %s (%d rows)", symbol, timeframe, path.name, len(out))
    except Exception as exc:
        logger.warning("data_cache save_disk %s %s: %s", symbol, timeframe, exc)


def _load_from_disk(symbol: str, timeframe: str) -> pd.DataFrame:
    """从本地 Parquet 文件加载，返回 DataFrame 或空 DataFrame。"""
    path = _parquet_path(symbol, timeframe)
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(path, engine="pyarrow")
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        df["timestamp"] = df["timestamp_utc"]
        logger.info("data_cache load_disk %s %s ← %s (%d rows)", symbol, timeframe, path.name, len(df))
        return df
    except Exception as exc:
        logger.warning("data_cache load_disk %s %s: %s", symbol, timeframe, exc)
        return pd.DataFrame()


def _file_age_seconds(symbol: str, timeframe: str) -> float:
    """返回本地文件距今秒数，不存在返回 inf。"""
    path = _parquet_path(symbol, timeframe)
    if not path.exists():
        return float("inf")
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return (datetime.now(timezone.utc) - mtime).total_seconds()


# ---------------------------------------------------------------------------
# yfinance 拉取（仅在本地文件缺失/过期时调用）
# ---------------------------------------------------------------------------
def _yf_fetch(symbol: str, timeframe: str, period: str) -> pd.DataFrame:
    """从 yfinance 拉取行情，返回标准格式 DataFrame。"""
    try:
        import yfinance as yf
        df = yf.Ticker(symbol).history(
            period=period,
            interval=timeframe,
            auto_adjust=True,
            prepost=False,
        )
        if df.empty:
            return pd.DataFrame()
        rows = []
        for ts, row in df.iterrows():
            dt = ts.to_pydatetime()
            ts_utc = (pd.Timestamp(dt).tz_convert("UTC")
                      if dt.tzinfo else pd.Timestamp(dt, tz="UTC"))
            rows.append({
                "symbol":        symbol,
                "timestamp_utc": ts_utc,
                "timestamp":     ts_utc,
                "open":          float(row["Open"]),
                "high":          float(row["High"]),
                "low":           float(row["Low"]),
                "close":         float(row["Close"]),
                "volume":        float(row["Volume"]),
            })
        return pd.DataFrame(rows)
    except Exception as exc:
        logger.warning("data_cache yf_fetch %s %s: %s", symbol, timeframe, exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# 合并工具
# ---------------------------------------------------------------------------
def _merge_into_cache(key: Tuple[str, str], new_df: pd.DataFrame) -> None:
    if new_df.empty:
        return
    with _CACHE_LOCK:
        existing = _CACHE.get(key)
        if existing is None or existing.empty:
            _CACHE[key] = new_df
        else:
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined["timestamp_utc"] = pd.to_datetime(combined["timestamp_utc"], utc=True)
            combined = (combined
                        .sort_values("timestamp_utc")
                        .drop_duplicates(subset=["timestamp_utc"])
                        .reset_index(drop=True))
            combined["timestamp"] = combined["timestamp_utc"]
            _CACHE[key] = combined


# ---------------------------------------------------------------------------
# 加载策略
# ---------------------------------------------------------------------------
def _ensure_loaded(symbol: str, timeframe: str) -> None:
    """
    从本地加载数据到内存缓存。严格本地优先，绝不自动联网。
      1. 内存缓存已有 → 立即返回
      2. 本地 Parquet 文件存在 → 读取到内存
      3. 本地文件不存在 → 什么都不做（调用方会得到空 DataFrame）
    如需下载数据，请调用 fetch_and_save() 或在 UI 点击"📥 下载/更新数据"。
    """
    key = (symbol, timeframe)

    # 1. 内存已有
    with _CACHE_LOCK:
        if key in _CACHE and not _CACHE[key].empty:
            return

    # 2. 本地文件存在（不管是否过期，都读取 — 过期由用户手动刷新）
    df_disk = _load_from_disk(symbol, timeframe)
    if not df_disk.empty:
        with _CACHE_LOCK:
            _CACHE[key] = df_disk
        logger.debug("_ensure_loaded %s %s from disk (%d rows)", symbol, timeframe, len(df_disk))
    # 3. 本地文件不存在 → 不联网，直接返回空


def _incremental_update(symbol: str, timeframe: str) -> None:
    """
    增量更新：只拉最近几天的 bar 合并进缓存，并更新本地文件。
    仅在本地文件已存在时才执行（保证有基础数据）。
    有最小间隔限制，避免 UI 每次刷新都发请求。
    """
    key = (symbol, timeframe)

    # 本地文件不存在 → 不执行增量（用户需先点击下载）
    if not _parquet_path(symbol, timeframe).exists():
        return

    # 检查最小间隔
    last = _LAST_NETWORK_FETCH.get(key)
    if last and (datetime.now(timezone.utc) - last).total_seconds() < _INCR_MIN_INTERVAL:
        return  # 距上次拉取不足 60s，跳过

    period = _INCR_PERIOD.get(timeframe, "5d")
    # Alpaca primary (reliable for scanning many symbols); yfinance fallback.
    df_new = pd.DataFrame()
    if timeframe in _ALPACA_TF:
        days = int(period[:-1]) if period.endswith("d") and period[:-1].isdigit() else 5
        start_str = (datetime.now(timezone.utc) - timedelta(days=days + 1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        end_str   = (datetime.now(timezone.utc) - timedelta(minutes=20)).strftime("%Y-%m-%dT%H:%M:%SZ")
        df_new = _alpaca_fetch_bars(symbol, timeframe, start_str, end_str)
    if df_new.empty:
        df_new = _yf_fetch(symbol, timeframe, period)
    if df_new.empty:
        return

    _merge_into_cache(key, df_new)
    _LAST_NETWORK_FETCH[key] = datetime.now(timezone.utc)

    # 同步更新本地文件
    with _CACHE_LOCK:
        merged = _CACHE.get(key)
    if merged is not None and not merged.empty:
        _save_to_disk(symbol, timeframe, merged)

    logger.debug("data_cache incr %s %s +%d rows", symbol, timeframe, len(df_new))


# ---------------------------------------------------------------------------
# 公共 API
# ---------------------------------------------------------------------------
def get_bars(symbol: str, timeframe: str) -> pd.DataFrame:
    """
    返回 (symbol, timeframe) 的行情 DataFrame。
    严格本地优先：内存 → 本地 Parquet 文件 → 返回空 DataFrame。
    绝不自动联网。如需下载，请调用 fetch_and_save() 或点击 UI 按钮。
    """
    _ensure_loaded(symbol, timeframe)
    with _CACHE_LOCK:
        df = _CACHE.get((symbol, timeframe), pd.DataFrame())
    return df.copy() if df is not None else pd.DataFrame()


def refresh_bars(symbol: str, timeframe: str) -> pd.DataFrame:
    """
    增量更新并返回最新 DataFrame。
    - 本地文件存在 → 读取本地 + 追加最近几根 bar（yfinance 增量）
    - 本地文件不存在 → 直接返回空 DataFrame，不联网
    """
    _ensure_loaded(symbol, timeframe)          # 本地文件 → 内存
    _incremental_update(symbol, timeframe)     # 仅在本地文件存在时才追加新 bar
    return get_bars(symbol, timeframe)


def is_warm(symbol: str, timeframe: str) -> bool:
    """True 表示该 (symbol, timeframe) 已有数据（内存或本地文件均算）。"""
    key = (symbol, timeframe)
    with _CACHE_LOCK:
        df = _CACHE.get(key)
    if df is not None and not df.empty:
        return True
    return _parquet_path(symbol, timeframe).exists()


def list_cached_files() -> list:
    """列出所有本地缓存文件的信息（供 UI 展示）。"""
    result = []
    for f in sorted(_BARS_DIR.glob("*.parquet")):
        try:
            df = pd.read_parquet(f, engine="pyarrow", columns=["timestamp_utc"])
            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
            mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc)
            result.append({
                "文件":   f.name,
                "行数":   len(df),
                "起始":   str(df["timestamp_utc"].min())[:10],
                "截止":   str(df["timestamp_utc"].max())[:10],
                "更新时间": mtime.strftime("%Y-%m-%d %H:%M"),
                "大小(KB)": round(f.stat().st_size / 1024, 1),
            })
        except Exception:
            result.append({"文件": f.name, "行数": "?", "起始": "?",
                           "截止": "?", "更新时间": "?", "大小(KB)": "?"})
    return result


# Timeframes Alpaca supports for bar history (1m excluded — free plan too shallow).
_ALPACA_TF = {"5m": "5Min", "15m": "15Min", "30m": "30Min", "1h": "1Hour", "1d": "1Day"}


def _alpaca_creds() -> tuple[str, str, str]:
    """(key, secret, feed) from the typed Settings; empty key means 'not configured'."""
    try:
        from .config import settings
        return settings.alpaca_api_key, settings.alpaca_secret_key, settings.alpaca_feed
    except Exception:
        import os as _os
        return (_os.getenv("ALPACA_API_KEY", ""),
                _os.getenv("ALPACA_API_SECRET", "") or _os.getenv("ALPACA_SECRET_KEY", ""),
                _os.getenv("ALPACA_DATA_FEED", "") or _os.getenv("ALPACA_FEED", "sip"))


def _alpaca_fetch_bars(symbol: str, timeframe: str, start_str: str, end_str: str) -> pd.DataFrame:
    """Single Alpaca bars fetcher (paginated) — the one place we talk to Alpaca.
    Used by both full-history download and incremental refresh. Returns empty
    frame if creds are missing or the timeframe is unsupported."""
    import requests as _req
    api_key, api_secret, feed = _alpaca_creds()
    alp_tf = _ALPACA_TF.get(timeframe)
    if not api_key or not api_secret or not alp_tf:
        return pd.DataFrame()

    url     = f"https://data.alpaca.markets/v2/stocks/{symbol}/bars"
    headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": api_secret}
    params: dict = {"timeframe": alp_tf, "start": start_str, "end": end_str,
                    "limit": 10000, "feed": feed, "sort": "asc"}
    all_bars: list = []
    page = 0
    while True:
        try:
            resp = _req.get(url, headers=headers, params=params, timeout=(10, 60))
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("alpaca_fetch %s %s page=%d: %s", symbol, timeframe, page, exc)
            break
        bars = data.get("bars") or []
        all_bars.extend(bars)
        nxt = data.get("next_page_token")
        if not nxt or not bars:
            break
        params["page_token"] = nxt
        page += 1

    if not all_bars:
        return pd.DataFrame()
    return pd.DataFrame([{
        "symbol":        symbol,
        "timestamp_utc": pd.Timestamp(b["t"], tz="UTC"),
        "timestamp":     pd.Timestamp(b["t"], tz="UTC"),
        "open":   float(b["o"]), "high":   float(b["h"]),
        "low":    float(b["l"]), "close":  float(b["c"]),
        "volume": float(b["v"]),
    } for b in all_bars])


def _alpaca_fetch_full(symbol: str, timeframe: str) -> pd.DataFrame:
    """Full history from 2016 to now (free plan delay → end = now-20min)."""
    start_str = "2016-01-01T00:00:00Z"
    end_str   = (datetime.now(timezone.utc) - timedelta(minutes=20)).strftime("%Y-%m-%dT%H:%M:%SZ")
    df = _alpaca_fetch_bars(symbol, timeframe, start_str, end_str)
    if not df.empty:
        logger.info("alpaca_fetch_full %s %s → %d rows", symbol, timeframe, len(df))
    return df


def upsert_bars(symbol: str, timeframe: str, df: pd.DataFrame) -> None:
    """Merge a fresh DataFrame from the live data feed into the in-memory cache
    and flush to the local Parquet file.  Called by the Scheduler after each tick
    so that exploration/backtest panels always see the latest live bars."""
    if df is None or df.empty:
        return
    df = df.copy()
    if "timestamp_utc" not in df.columns and "timestamp" in df.columns:
        df["timestamp_utc"] = df["timestamp"]
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["timestamp"] = df["timestamp_utc"]
    if "symbol" not in df.columns:
        df["symbol"] = symbol
    key = (symbol, timeframe)
    _merge_into_cache(key, df)
    with _CACHE_LOCK:
        merged = _CACHE.get(key)
    if merged is not None and not merged.empty:
        _save_to_disk(symbol, timeframe, merged)


def fetch_and_save(symbol: str, timeframe: str) -> pd.DataFrame:
    """
    显式拉取全量数据并保存到本地 Parquet 文件。
    仅在用户点击"下载/更新数据"按钮时调用，不自动触发。

    策略：
      - 1m            → yfinance（Alpaca 不支持深度 1m）
      - 5m/15m/30m/1h/1d → Alpaca 优先（2016 至今），失败则 yfinance 兜底
    """
    key = (symbol, timeframe)
    df = pd.DataFrame()

    if timeframe == "1m":
        # Alpaca 免费计划 1m 历史很短，直接用 yfinance
        logger.info("fetch_and_save %s 1m — yfinance (7d)", symbol)
        df = _yf_fetch(symbol, "1m", "7d")
    else:
        # 优先 Alpaca（2016 至今）
        logger.info("fetch_and_save %s %s — Alpaca (2016-present)", symbol, timeframe)
        df = _alpaca_fetch_full(symbol, timeframe)
        if df.empty:
            # Alpaca 失败（无 Key 或网络问题）→ yfinance 兜底
            period = _YF_PERIOD.get(timeframe, "60d")
            logger.warning("fetch_and_save %s %s — Alpaca failed, fallback yfinance (%s)", symbol, timeframe, period)
            df = _yf_fetch(symbol, timeframe, period)

    if not df.empty:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        df = df.sort_values("timestamp_utc").drop_duplicates(subset=["timestamp_utc"]).reset_index(drop=True)
        with _CACHE_LOCK:
            _CACHE[key] = df
        _LAST_NETWORK_FETCH[key] = datetime.now(timezone.utc)
        _save_to_disk(symbol, timeframe, df)
    return df


def warm_up(
    symbols: list,
    timeframes: list | None = None,
    background: bool = True,
) -> None:
    """
    预热数据缓存。
    - 本地文件存在且未过期 → 从文件加载，不访问网络
    - 本地文件不存在或过期 → 从 yfinance 拉取并保存到本地
    - background=True（默认）→ 后台线程，不阻塞 UI
    """
    global _WARM_THREAD, _WARM_DONE
    if timeframes is None:
        timeframes = ["5m", "30m", "1d"]

    def _worker():
        global _WARM_DONE
        logger.info("data_cache warm_up START: %s × %s", symbols, timeframes)
        for sym in symbols:
            for tf in timeframes:
                try:
                    _ensure_loaded(sym, tf)
                except Exception as exc:
                    logger.warning("data_cache warm_up %s %s: %s", sym, tf, exc)
                time.sleep(0.1)   # 避免连续请求过快
        _WARM_DONE = True
        logger.info("data_cache warm_up DONE — files in %s", _BARS_DIR)

    if background:
        _WARM_THREAD = threading.Thread(target=_worker, daemon=True, name="data-cache-warmup")
        _WARM_THREAD.start()
    else:
        _worker()
