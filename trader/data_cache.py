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
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

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


def _write_parquet(path: Path, df: pd.DataFrame) -> None:
    try:
        df.to_parquet(path, index=False, engine="pyarrow")
        return
    except ImportError:
        pass

    import duckdb

    con = duckdb.connect(":memory:")
    try:
        con.register("bars_df", df)
        target = path.as_posix().replace("'", "''")
        con.execute(f"COPY bars_df TO '{target}' (FORMAT PARQUET)")
    finally:
        con.close()


def _read_parquet(path: Path, columns: Optional[list[str]] = None) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, engine="pyarrow", columns=columns)
    except ImportError:
        pass

    import duckdb

    col_sql = "*"
    if columns:
        col_sql = ", ".join(f'"{col}"' for col in columns)
    con = duckdb.connect(":memory:")
    try:
        return con.execute(
            f"SELECT {col_sql} FROM read_parquet(?)",
            [str(path)],
        ).df()
    finally:
        con.close()


def _save_to_disk(symbol: str, timeframe: str, df: pd.DataFrame) -> None:
    """保存 DataFrame 到本地 Parquet 文件。"""
    if df.empty:
        return
    path = _parquet_path(symbol, timeframe)
    try:
        out = df.copy()
        out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True)
        out["timestamp"] = out["timestamp_utc"]
        _write_parquet(path, out)
        logger.info("data_cache saved %s %s → %s (%d rows)", symbol, timeframe, path.name, len(out))
    except Exception as exc:
        logger.warning("data_cache save_disk %s %s: %s", symbol, timeframe, exc)


def _load_from_disk(symbol: str, timeframe: str) -> pd.DataFrame:
    """从本地 Parquet 文件加载，返回 DataFrame 或空 DataFrame。"""
    path = _parquet_path(symbol, timeframe)
    if not path.exists():
        return pd.DataFrame()
    try:
        df = _read_parquet(path)
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


def describe_cached_bars(
    symbol: str,
    timeframe: str,
    *,
    frame: pd.DataFrame | None = None,
    captured_at: datetime | None = None,
) -> dict:
    """Describe and serialize the exact cache frame consumed by research."""
    captured_at = (captured_at or datetime.now(timezone.utc)).astimezone(
        timezone.utc
    )
    bars = get_bars(symbol, timeframe) if frame is None else frame.copy()
    path = _parquet_path(symbol, timeframe)
    required = {
        "timestamp_utc",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }
    normalized = bars.copy()
    normalized.columns = [str(column).lower() for column in normalized.columns]
    missing_columns = sorted(required - set(normalized.columns))
    if missing_columns:
        rows: list[dict] = []
        data_start = None
        data_end = None
    else:
        normalized["timestamp_utc"] = pd.to_datetime(
            normalized["timestamp_utc"],
            utc=True,
        )
        columns = [
            column
            for column in (
                "symbol",
                "timestamp_utc",
                "open",
                "high",
                "low",
                "close",
                "volume",
            )
            if column in normalized.columns
        ]
        rows = []
        for raw in normalized[columns].to_dict(orient="records"):
            row = {}
            for key, value in raw.items():
                if key == "timestamp_utc":
                    row[key] = pd.Timestamp(value).isoformat()
                elif pd.isna(value):
                    row[key] = None
                elif hasattr(value, "item"):
                    row[key] = value.item()
                else:
                    row[key] = value
            rows.append(row)
        data_start = (
            normalized["timestamp_utc"].min().to_pydatetime()
            if not normalized.empty
            else None
        )
        data_end = (
            normalized["timestamp_utc"].max().to_pydatetime()
            if not normalized.empty
            else None
        )

    # 缓存文件里最近一次写入用的 feed 是否还和当前配置一致 —— data_cache 严格
    # 本地优先、绝不自动核实文件内容，所以这是唯一能发现"这份数据是用旧口径
    # 抓的"的地方。2026-08-06：一批 IEX 单一交易所口径的成交量（只有真实值
    # 2~7%）在磁盘上躺了几个月没被发现，起因正是完全没有这层核对。
    cached_feed = ""
    feed_mismatch = False
    if not normalized.empty:
        if "source_feed" in normalized.columns:
            last_feed = normalized["source_feed"].dropna()
            cached_feed = str(last_feed.iloc[-1]) if not last_feed.empty else ""
        # 没打过标签（老文件，早于本次修复）同样当可疑处理，不默认信任——
        # 不知道来源就是不知道，"当它是对的"正是这批脏数据躺了几个月的原因。
        if cached_feed != "yfinance":
            try:
                configured_feed = _alpaca_creds()[2]
            except Exception:
                configured_feed = ""
            feed_mismatch = bool(configured_feed) and cached_feed != configured_feed

    if normalized.empty:
        status = "MISSING"
        quality_score = 0.0
        failure_code = "BAR_CACHE_EMPTY"
    elif missing_columns:
        status = "FAILED"
        quality_score = 0.0
        failure_code = "BAR_COLUMNS_MISSING"
    elif not rows:
        status = "MISSING"
        quality_score = 0.0
        failure_code = "BAR_CACHE_EMPTY"
    elif len(rows) < 40:
        status = "DEGRADED"
        quality_score = min(len(rows) / 40.0, 1.0)
        failure_code = "BAR_HISTORY_SHORT"
    elif feed_mismatch:
        # 文件没被数据本身的问题拖累（够长、列齐全），但抓取时用的 feed 和
        # 现在配置的不一样 —— 典型场景就是成交量口径不一致。数据可能仍然可
        # 用（价格类字段通常影响很小），所以降级而不是判 MISSING。
        status = "DEGRADED"
        quality_score = 0.5
        failure_code = "BAR_FEED_STALE"
    else:
        status = "OK"
        quality_score = 1.0
        failure_code = ""
    file_updated_at = (
        datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        if path.exists()
        else None
    )
    return {
        "source": "local_bar_cache",
        "status": status,
        "as_of": data_end or captured_at,
        "fetched_at": captured_at,
        "quality_score": quality_score,
        "coverage": ("ohlcv",),
        "payload_version": "local-bars:v1",
        "failure_code": failure_code,
        "metadata": {
            "symbol": symbol,
            "timeframe": timeframe,
            "row_count": len(rows),
            "missing_columns": missing_columns,
            "cache_file": str(path),
            "file_updated_at": (
                file_updated_at.isoformat()
                if file_updated_at is not None
                else None
            ),
            "data_start": (
                data_start.isoformat() if data_start is not None else None
            ),
            "data_end": (
                data_end.isoformat() if data_end is not None else None
            ),
            "cached_feed": cached_feed,
            "feed_mismatch": feed_mismatch,
        },
        "payload": {
            "symbol": symbol,
            "timeframe": timeframe,
            "columns": sorted(required),
            "rows": rows,
        },
    }






def list_cached_files() -> list:
    """列出所有本地缓存文件的信息（供 UI 展示）。"""
    result = []
    for f in sorted(_BARS_DIR.glob("*.parquet")):
        try:
            df = _read_parquet(f, columns=["timestamp_utc"])
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


def list_cached_names() -> list[str]:
    """只列缓存文件名，不读取 Parquet 内容。"""
    return sorted(path.name for path in _BARS_DIR.glob("*.parquet"))


# ---------------------------------------------------------------------------
# 观察列表清理 — 被踢出观察池的 symbol 删除其本地 bar 缓存
#
# data/bars/ 是多个互不相干的消费方共用的：选股池、Runtime 的 --symbols、
# daily_research、全市场扫描、UI 手动下载。调用方（选股池重建）只知道自己那
# 一份名单，所以"其余消费方还要什么"必须在这里统一兜住，否则会删掉 Runtime
# 正在交易的标的——而 data_cache 是严格本地优先、绝不自动联网，删了不会自
# 动补回来，后果是 get_bars() 返回空、快照标记 BAR_CACHE_EMPTY、决策门禁静
# 默拦下所有交易。
# ---------------------------------------------------------------------------
def _configured_universe() -> set[str]:
    """.env 的 SYMBOLS —— 数据下载/预热脚本的默认标的池。

    Settings 刻意不读这个键（引擎 universe 由 CLI 控制），但它仍是用户声明
    "我关心这些标的"的唯一持久化信号，所以清理时必须尊重它。
    """
    import os
    try:
        from . import config as _config  # noqa: F401  (imported for load_dotenv)
    except Exception:
        pass
    raw = os.getenv("SYMBOLS", "") or ""
    return {s.strip().upper() for s in raw.split(",") if s.strip()}


def _recent_research_symbols(
    ai_db_path: str | Path | None = None,
    *,
    days: int = 30,
) -> set[str]:
    """最近做过研究的标的 —— Runtime 的 --symbols universe 的可观测代理。

    Runtime 的标的来自命令行、不落盘，这里无法直接读到；但 Runtime 跑过什么，
    daily_research 就会给什么留下快照，所以用近期快照反推交易 universe。
    """
    symbols: set[str] = set()
    path = Path(ai_db_path) if ai_db_path else _ROOT / "ai_states.duckdb"
    if not path.exists():
        return symbols
    try:
        import duckdb
        con = duckdb.connect(str(path), read_only=True)
        try:
            # trading_date is stored as an ISO date string, so compare as text
            # (ISO dates sort lexicographically) rather than binding a date.
            cutoff = (
                datetime.now(timezone.utc) - timedelta(days=days)
            ).date().isoformat()
            rows = con.execute(
                "SELECT DISTINCT symbol FROM research_snapshots "
                "WHERE CAST(trading_date AS VARCHAR) >= ?",
                [cutoff],
            ).fetchall()
            symbols.update(str(r[0]).upper() for r in rows if r[0])
        finally:
            con.close()
    except Exception as exc:
        logger.warning("data_cache _recent_research_symbols: %s", exc)
    return symbols


def _protected_symbols(db_path: str | Path | None = None) -> set[str]:
    """有实仓持仓或未终结挂单的 symbol，永远不清理，无论是否还在观察池里。
    只读、尽力而为：数据库/表不存在时静默返回空集合，不影响调用方自己的
    keep 列表继续生效。"""
    protected: set[str] = set()
    path = Path(db_path) if db_path else _ROOT / "trade.duckdb"
    if not path.exists():
        return protected
    try:
        import duckdb
        con = duckdb.connect(str(path), read_only=True)
        try:
            try:
                rows = con.execute(
                    """
                    SELECT symbol
                    FROM fills
                    GROUP BY symbol
                    HAVING SUM(CASE WHEN side = 'BUY' THEN filled_qty ELSE -filled_qty END) <> 0
                    """
                ).fetchall()
                protected.update(str(r[0]).upper() for r in rows if r[0])
            except Exception:
                pass
            try:
                rows = con.execute(
                    """
                    SELECT DISTINCT symbol FROM order_intents
                    WHERE state NOT IN ('FILLED', 'CANCELED', 'REJECTED', 'EXPIRED')
                    """
                ).fetchall()
                protected.update(str(r[0]).upper() for r in rows if r[0])
            except Exception:
                pass
        finally:
            con.close()
    except Exception as exc:
        logger.warning("data_cache _protected_symbols: %s", exc)
    return protected


def prune_bar_cache(
    keep_symbols: Iterable[str],
    *,
    db_path: str | Path | None = None,
    ai_db_path: str | Path | None = None,
    grace_days: int = 7,
) -> dict:
    """删除不在 keep_symbols 里的本地 bar 缓存文件（跨所有 timeframe）。

    调用方只需要传自己那份名单。其余消费方的需求在这里统一兜住，一律豁免：
      - 有实仓持仓、或有未终结挂单的标的
      - .env SYMBOLS 声明的标的池
      - 最近 30 天做过研究的标的（Runtime --symbols universe 的代理）
      - 最近 grace_days 天内下载过的文件（刚下好还没来得及被任何名单收录）

    grace_days=0 关闭时间宽限，只按名单删。
    """
    keep = {str(s).strip().upper() for s in keep_symbols if str(s).strip()}
    keep |= _protected_symbols(db_path)
    keep |= _configured_universe()
    keep |= _recent_research_symbols(ai_db_path)

    cutoff = (
        datetime.now(timezone.utc) - timedelta(days=grace_days)
        if grace_days > 0
        else None
    )

    removed: list[str] = []
    removed_bytes = 0
    kept_by_grace: list[str] = []
    if _BARS_DIR.exists():
        for path in sorted(_BARS_DIR.glob("*.parquet")):
            stem = path.stem
            if "_" not in stem:
                continue
            symbol = stem.rsplit("_", 1)[0].upper()
            if symbol in keep:
                continue
            try:
                stat = path.stat()
                if cutoff is not None:
                    mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
                    if mtime >= cutoff:
                        kept_by_grace.append(path.name)
                        continue
                size = stat.st_size
                path.unlink()
                removed.append(path.name)
                removed_bytes += size
                with _CACHE_LOCK:
                    for cache_key in [k for k in _CACHE if k[0] == symbol]:
                        _CACHE.pop(cache_key, None)
            except Exception as exc:
                logger.warning("prune_bar_cache unlink %s: %s", path.name, exc)

    if removed:
        logger.info(
            "prune_bar_cache removed %d file(s), %.1f MB reclaimed: %s",
            len(removed), removed_bytes / (1024 * 1024), ", ".join(removed),
        )
    if kept_by_grace:
        logger.info(
            "prune_bar_cache kept %d recently-downloaded file(s) within the "
            "%d-day grace window: %s",
            len(kept_by_grace), grace_days, ", ".join(kept_by_grace),
        )
    return {
        "removed_files": removed,
        "removed_bytes": removed_bytes,
        "kept_symbols": sorted(keep),
        "kept_by_grace": kept_by_grace,
    }


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


def fetch_alpaca_bars_window(
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Read one explicit Alpaca history window without mutating local cache."""
    if start.tzinfo is None or start.utcoffset() is None:
        raise ValueError("ALPACA_HISTORY_START_TIMEZONE_REQUIRED")
    if end.tzinfo is None or end.utcoffset() is None:
        raise ValueError("ALPACA_HISTORY_END_TIMEZONE_REQUIRED")
    start_utc = start.astimezone(timezone.utc)
    end_utc = end.astimezone(timezone.utc)
    if start_utc >= end_utc:
        raise ValueError("ALPACA_HISTORY_WINDOW_INVALID")
    normalized_symbol = symbol.strip().upper()
    if not normalized_symbol:
        raise ValueError("ALPACA_HISTORY_SYMBOL_REQUIRED")
    if timeframe not in _ALPACA_TF:
        raise ValueError("ALPACA_HISTORY_TIMEFRAME_UNSUPPORTED")
    return _alpaca_fetch_bars(
        normalized_symbol,
        timeframe,
        start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        end_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


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
    and flush to the local Parquet file.  Called by the Runtime after each tick
    so that exploration/backtest panels always see the latest live bars, and by
    ensure_bars() to top up thin caches. Both callers source almost exclusively
    from Alpaca (data_feed.AlpacaDataFeed / fetch_alpaca_bars_window), so the
    currently configured feed is an accurate default tag -- but a caller that
    already knows better (ensure_bars' yfinance fallback for 1m) can stamp its
    own source_feed before calling in, and that tag is respected here rather
    than overwritten."""
    if df is None or df.empty:
        return
    df = df.copy()
    if "timestamp_utc" not in df.columns and "timestamp" in df.columns:
        df["timestamp_utc"] = df["timestamp"]
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["timestamp"] = df["timestamp_utc"]
    if "symbol" not in df.columns:
        df["symbol"] = symbol
    if "source_feed" not in df.columns:
        df["source_feed"] = _alpaca_creds()[2]
    key = (symbol, timeframe)
    _merge_into_cache(key, df)
    with _CACHE_LOCK:
        merged = _CACHE.get(key)
    if merged is not None and not merged.empty:
        _save_to_disk(symbol, timeframe, merged)


# 每个周期补多长的窗口。必须让 min_rows 在该窗口内可达，否则每次调用都会
# 对同一批标的重复下载：日线 30 天只有 21 根，永远到不了阈值。
_ENSURE_LOOKBACK_DAYS = {
    "1m": 7,      # Alpaca 免费计划 1m 历史本来就浅
    "5m": 30,
    "15m": 60,
    "30m": 60,
    "1h": 120,
    "1d": 400,
}


def ensure_bars(
    symbols: Iterable[str],
    timeframe: str,
    *,
    lookback_days: int | None = None,
    min_rows: int = 200,
) -> dict:
    """补齐 symbols 在该周期上缺失/过短的本地缓存，best-effort。

    有界拉取（按周期取窗口，见 _ENSURE_LOOKBACK_DAYS），不是 fetch_and_save
    那种 2016 年至今的全量下载 —— 缓存里实际存的就是几天到几周的近期窗口，
    全量拉一次 5m 要几十万根。

    min_rows 刻意高于 describe_cached_bars 判 DEGRADED 的 40 根：40 根只够
    "不被标记为降级"，但远不够跑筛选和生成策略统计。实测 120 根 5m（约 1.5
    个交易日）就会让 strategy_statistics 完全生不成该标的的记录。

    行数够了不代表数据是新的：_FILE_MAX_AGE 之前定义了但没接到任何地方——
    这正是那批 IEX 口径脏数据能在磁盘上躺几个月不被发现的原因，文件行数一
    直够、只是内容早就不对了。这里把它接上：文件超过对应周期的最大寿命，
    即使行数达标也重新拉一次。

    任何一个标的失败都不抛异常：补数据是尽力而为的前置动作，不该让整个研究
    批次因为一次网络抖动而失败。
    """
    if lookback_days is None:
        lookback_days = _ENSURE_LOOKBACK_DAYS.get(timeframe, 30)
    max_age = _FILE_MAX_AGE.get(timeframe)
    filled: list[str] = []
    sufficient: list[str] = []
    stale: list[str] = []
    failed: list[str] = []

    for raw in symbols:
        symbol = str(raw).strip().upper()
        if not symbol:
            continue
        has_enough_rows = len(get_bars(symbol, timeframe)) >= min_rows
        is_stale = max_age is not None and _file_age_seconds(symbol, timeframe) > max_age
        if has_enough_rows and not is_stale:
            sufficient.append(symbol)
            continue
        if is_stale and has_enough_rows:
            stale.append(symbol)
        try:
            if timeframe == "1m":
                # fetch_alpaca_bars_window rejects "1m" outright (Alpaca's free
                # plan doesn't carry deep 1m history -- see _ALPACA_TF); every
                # symbol would land in `failed` without this. Same yfinance
                # fallback fetch_and_save() already uses for this timeframe.
                df = _yf_fetch(symbol, "1m", "7d")
                if not df.empty:
                    df["source_feed"] = "yfinance"
            else:
                end = datetime.now(timezone.utc) - timedelta(minutes=20)
                start = end - timedelta(days=lookback_days)
                df = fetch_alpaca_bars_window(symbol, timeframe, start, end)
            if df.empty:
                failed.append(symbol)
                continue
            upsert_bars(symbol, timeframe, df)
            filled.append(symbol)
        except Exception as exc:
            logger.warning("ensure_bars %s %s: %s", symbol, timeframe, exc)
            failed.append(symbol)

    if filled or failed:
        logger.info(
            "ensure_bars %s — filled=%d (of which stale=%d) sufficient=%d failed=%d%s",
            timeframe, len(filled), len(stale), len(sufficient), len(failed),
            f" (failed: {', '.join(failed)})" if failed else "",
        )
    return {
        "filled": filled,
        "sufficient": sufficient,
        "stale": stale,
        "failed": failed,
    }


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
    source_feed = ""

    if timeframe == "1m":
        # Alpaca 免费计划 1m 历史很短，直接用 yfinance
        logger.info("fetch_and_save %s 1m — yfinance (7d)", symbol)
        df = _yf_fetch(symbol, "1m", "7d")
        source_feed = "yfinance"
    else:
        # 优先 Alpaca（2016 至今）
        logger.info("fetch_and_save %s %s — Alpaca (2016-present)", symbol, timeframe)
        df = _alpaca_fetch_full(symbol, timeframe)
        source_feed = _alpaca_creds()[2]
        if df.empty:
            # Alpaca 失败（无 Key 或网络问题）→ yfinance 兜底
            period = _YF_PERIOD.get(timeframe, "60d")
            logger.warning("fetch_and_save %s %s — Alpaca failed, fallback yfinance (%s)", symbol, timeframe, period)
            df = _yf_fetch(symbol, timeframe, period)
            source_feed = "yfinance"

    if not df.empty:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        df = df.sort_values("timestamp_utc").drop_duplicates(subset=["timestamp_utc"]).reset_index(drop=True)
        # 记录这批数据是用哪个 feed 抓的，供 describe_cached_bars 核对是否和
        # 当前配置一致 —— 这是 2026-08-06 发现的成交量口径问题的补救：那批
        # 脏数据（IEX 单一交易所成交量，只有真实值的 2~7%）没有任何标记能
        # 让系统自己发现它和当前 sip 配置不一致，只能一直悄悄躺在缓存里。
        df["source_feed"] = source_feed
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
