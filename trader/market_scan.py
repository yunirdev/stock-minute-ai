from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

import pandas as pd

from .hot_universe import build_hot_universe, hot_symbols, load_hot_universe
from .index_universe import load_index_universe, update_index_universe
from .symbol_master import (
    SourceStatus,
    common_equity_symbols,
    load_symbol_master,
    parse_symbol_text,
    update_symbol_master,
)


_ROOT = Path(__file__).resolve().parents[1]
_STORE = _ROOT / "conf" / "market_scan_report.json"

KEEP = "KEEP"
WATCH = "WATCH"
HOT = "HOT"
REJECT = "REJECT"


@dataclass
class MarketScanItem:
    symbol: str
    rank: int
    score: float
    status: str
    setup: str = ""
    component_scores: dict[str, float | None] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    reject_reasons: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    metrics: dict[str, float | None] = field(default_factory=dict)


@dataclass
class MarketScanReport:
    updated_at: str
    universe_size: int
    scanned_size: int
    selected_size: int
    rejected_size: int
    items: list[MarketScanItem]
    source_status: list[SourceStatus] = field(default_factory=list)
    reject_summary: dict[str, int] = field(default_factory=dict)


def run_market_scan(
    *,
    source: Iterable[str] | str | None = None,
    refresh_universe: bool = True,
    include_broad_market: bool = True,
    max_symbols: int = 10000,
    max_downloads: int = 10000,
    download_missing: bool = True,
    require_fresh_bars: bool = True,
    timeframe: str = "1d",
    selected_limit: int = 500,
    min_price: float = 5.0,
    min_dollar_volume: float = 10_000_000.0,
    save: bool = True,
    path: Path | str = _STORE,
    progress_callback: Optional[Callable[[dict], None]] = None,
) -> MarketScanReport:
    universe, statuses, tags = build_first_round_universe(
        source=source,
        refresh_universe=refresh_universe,
        include_broad_market=include_broad_market,
        max_symbols=max_symbols,
    )
    total = len(universe[:max_symbols])
    _progress(progress_callback, "market_scan", 0, total, "scanning fresh bars")
    download_budget = max(0, int(max_downloads))
    items: list[MarketScanItem] = []
    reject_summary: dict[str, int] = {}
    price_attempts = 0
    price_success = 0
    price_fail = 0
    price_budget_exhausted = 0

    for idx, symbol in enumerate(universe[:max_symbols], start=1):
        downloaded = False
        budget_exhausted = False
        if require_fresh_bars:
            df = pd.DataFrame()
            if download_budget > 0:
                price_attempts += 1
                df = _download_bars(symbol, timeframe)
                downloaded = df is not None and not df.empty
                download_budget -= 1
                if downloaded:
                    price_success += 1
                else:
                    price_fail += 1
            else:
                budget_exhausted = True
                price_budget_exhausted += 1
        else:
            df = _load_bars(symbol, timeframe)
            if (df is None or df.empty) and download_missing and download_budget > 0:
                price_attempts += 1
                df = _download_bars(symbol, timeframe)
                downloaded = df is not None and not df.empty
                download_budget -= 1
                if downloaded:
                    price_success += 1
                else:
                    price_fail += 1

        item = _score_symbol(
            symbol,
            df,
            tags=tags.get(symbol, []),
            min_price=min_price,
            min_dollar_volume=min_dollar_volume,
        )
        if downloaded:
            item.reasons.append("downloaded fresh bars")
        elif require_fresh_bars and budget_exhausted:
            item.reject_reasons = ["fresh_bar_budget_exhausted"]
            item.status = REJECT
            item.setup = "NO_DATA"
        elif require_fresh_bars and item.status == REJECT and "missing_bars" in item.reject_reasons:
            item.reject_reasons = ["fresh_bar_unavailable"]
            item.setup = "NO_DATA"
        items.append(item)
        if item.status == REJECT:
            for reason in item.reject_reasons:
                reject_summary[reason] = reject_summary.get(reason, 0) + 1
        if idx == total or idx % 10 == 0:
            _progress(
                progress_callback,
                "market_scan",
                idx,
                total,
                f"{symbol}: {item.status}",
            )

    items.sort(key=lambda row: (_status_order(row.status), -row.score, row.symbol))
    for rank, item in enumerate(items, start=1):
        item.rank = rank
    selected = [item for item in items if item.status != REJECT][:selected_limit]
    rejected_size = sum(1 for item in items if item.status == REJECT)
    statuses.append(SourceStatus(
        source="price_bars",
        ok=price_success > 0 and price_budget_exhausted == 0 and price_fail == 0,
        count=price_success,
        message=(
            f"fresh attempts={price_attempts}, failed={price_fail}, "
            f"budget_exhausted={price_budget_exhausted}"
        ),
        updated_at=_now_s(),
    ))
    report = MarketScanReport(
        updated_at=_now_s(),
        universe_size=len(universe),
        scanned_size=len(items),
        selected_size=len(selected),
        rejected_size=rejected_size,
        items=items,
        source_status=statuses,
        reject_summary=dict(sorted(reject_summary.items(), key=lambda kv: (-kv[1], kv[0]))),
    )
    if save:
        save_market_scan_report(report, path)
    _progress(progress_callback, "market_scan", total, total, "saved scan report")
    return report


def build_first_round_universe(
    *,
    source: Iterable[str] | str | None = None,
    refresh_universe: bool = False,
    include_broad_market: bool = True,
    max_symbols: int = 10000,
) -> tuple[list[str], list[SourceStatus], dict[str, list[str]]]:
    explicit = parse_symbol_text(source)
    statuses: list[SourceStatus] = []
    tags: dict[str, list[str]] = {}
    symbols: list[str] = []

    if refresh_universe:
        master = update_symbol_master()
        index_snapshot = update_index_universe()
        hot_snapshot = build_hot_universe(base_symbols=common_equity_symbols(master, limit=10000))
    else:
        master = load_symbol_master()
        if not master.symbols:
            master = update_symbol_master()
        index_snapshot = load_index_universe()
        if not index_snapshot.core_symbols:
            index_snapshot = update_index_universe()
        hot_snapshot = load_hot_universe()
        if not hot_snapshot.symbols:
            hot_snapshot = build_hot_universe(base_symbols=common_equity_symbols(master, limit=10000))

    statuses.extend(master.source_status)
    statuses.extend(index_snapshot.source_status)
    statuses.extend(hot_snapshot.source_status)

    for symbol in index_snapshot.core_symbols:
        _add_symbol(symbols, tags, symbol, "core_index")
    for symbol in explicit:
        _add_symbol(symbols, tags, symbol, "manual")
    for symbol in hot_symbols(hot_snapshot, limit=120):
        _add_symbol(symbols, tags, symbol, "hot")
    if include_broad_market:
        for symbol in common_equity_symbols(master, limit=max_symbols * 2):
            _add_symbol(symbols, tags, symbol, "broad_market")

    return symbols[:max_symbols], statuses, tags


def load_market_scan_report(path: Path | str = _STORE) -> MarketScanReport:
    src = Path(path)
    if not src.exists():
        return MarketScanReport(
            updated_at="",
            universe_size=0,
            scanned_size=0,
            selected_size=0,
            rejected_size=0,
            items=[],
        )
    try:
        payload = json.loads(src.read_text(encoding="utf-8"))
        return MarketScanReport(
            updated_at=payload.get("updated_at", ""),
            universe_size=int(payload.get("universe_size", 0) or 0),
            scanned_size=int(payload.get("scanned_size", 0) or 0),
            selected_size=int(payload.get("selected_size", 0) or 0),
            rejected_size=int(payload.get("rejected_size", 0) or 0),
            items=[MarketScanItem(**item) for item in payload.get("items", [])],
            source_status=[SourceStatus(**item) for item in payload.get("source_status", [])],
            reject_summary=dict(payload.get("reject_summary", {})),
        )
    except Exception:
        return MarketScanReport(
            updated_at="",
            universe_size=0,
            scanned_size=0,
            selected_size=0,
            rejected_size=0,
            items=[],
        )


def save_market_scan_report(report: MarketScanReport, path: Path | str = _STORE) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8")


def market_scan_symbols(
    *,
    statuses: Optional[set[str]] = None,
    limit: int = 500,
    path: Path | str = _STORE,
) -> list[str]:
    report = load_market_scan_report(path)
    allowed = statuses or {KEEP, WATCH, HOT}
    rows = [item for item in report.items if item.status in allowed]
    rows.sort(key=lambda row: row.rank or 999)
    return [row.symbol for row in rows[:limit]]


def _score_symbol(
    symbol: str,
    df: Optional[pd.DataFrame],
    *,
    tags: list[str],
    min_price: float,
    min_dollar_volume: float,
) -> MarketScanItem:
    if df is None or df.empty:
        return MarketScanItem(
            symbol=symbol,
            rank=0,
            score=0.0,
            status=REJECT,
            setup="NO_DATA",
            reject_reasons=["missing_bars"],
            tags=tags,
        )

    clean = _clean_bars(df)
    if len(clean) < 80:
        return MarketScanItem(
            symbol=symbol,
            rank=0,
            score=0.0,
            status=REJECT,
            setup="NO_DATA",
            reject_reasons=["insufficient_history"],
            tags=tags,
        )

    close = clean["close"]
    volume = clean["volume"]
    last = float(close.iloc[-1])
    adv20 = float((close * volume).tail(20).mean())
    ret20 = _pct(last, float(close.iloc[-21]))
    ret60 = _pct(last, float(close.iloc[-61]))
    ret120 = _pct(last, float(close.iloc[-121])) if len(close) >= 121 else ret60
    ma50 = float(close.tail(50).mean())
    ma200 = float(close.tail(200).mean()) if len(close) >= 200 else ma50
    high120 = float(close.tail(min(120, len(close))).max())
    drawdown = _pct(last, high120)
    vol20 = float(close.pct_change().tail(20).std() * 100)

    reject_reasons: list[str] = []
    if last < min_price and adv20 < min_dollar_volume * 3:
        reject_reasons.append("price_below_min")
    if adv20 < min_dollar_volume:
        reject_reasons.append("dollar_volume_below_min")
    if vol20 > 8.0:
        reject_reasons.append("extreme_volatility")

    liquidity = _clip(35 + _scaled(adv20, 10_000_000, 500_000_000, 0, 40), 0, 100)
    long_trend = 50.0
    short_trend = 50.0
    reversal = 45.0
    reasons: list[str] = []
    if last > ma50:
        long_trend += 12
        short_trend -= 8
        reasons.append("above 50d average")
    else:
        short_trend += 10
    if last > ma200:
        long_trend += 12
        short_trend -= 8
        reasons.append("above 200d average")
    else:
        short_trend += 10
    if ma50 > ma200:
        long_trend += 8
        short_trend -= 6
        reasons.append("50d above 200d")
    else:
        short_trend += 7
    if ret60 > 8:
        long_trend += min(12, ret60 * 0.45)
        short_trend -= min(10, ret60 * 0.35)
        reasons.append(f"60d momentum {ret60:+.1f}%")
    elif ret60 < -8:
        long_trend += max(-14, ret60 * 0.45)
        short_trend += min(14, abs(ret60) * 0.45)
        reasons.append(f"60d downside momentum {ret60:+.1f}%")
    if ret20 < -12 and ret60 < -8:
        short_trend += min(10, abs(ret20) * 0.35)
        reasons.append(f"20d weakness {ret20:+.1f}%")
    if drawdown < -30:
        long_trend -= 8
        short_trend += 6
        reversal += min(25, abs(drawdown + 20) * 0.9)
        reasons.append(f"deep drawdown {drawdown:.1f}%")
    if ret20 < -18 and vol20 <= 6:
        reversal += 10
        reasons.append("oversold reversal candidate")
    if last > ma50 and ret20 > 12:
        reversal -= 8
    if last < min_price:
        reversal -= 5
        reasons.append("low price requires extra caution")
    long_trend = _clip(long_trend, 0, 100)
    short_trend = _clip(short_trend, 0, 100)
    reversal = _clip(reversal, 0, 100)

    risk = 75.0
    if vol20 > 4:
        risk -= min(25, (vol20 - 4) * 5)
    if drawdown < -20:
        risk -= min(25, abs(drawdown + 20) * 0.8)
    risk = _clip(risk, 0, 100)

    hot_bonus = 0.0
    index_bonus = 0.0
    if "hot" in tags:
        hot_bonus = 8.0
        reasons.append("hot universe supplement")
    if "core_index" in tags:
        index_bonus = 6.0
        reasons.append("Dow/S&P/Nasdaq-100 coverage")

    setup_scores = {
        "LONG_TREND": long_trend,
        "SHORT_TREND": short_trend,
        "REVERSAL": reversal,
    }
    setup = max(setup_scores, key=setup_scores.get)
    directional = setup_scores[setup]
    score = liquidity * 0.30 + directional * 0.45 + risk * 0.20 + hot_bonus + index_bonus
    status = REJECT
    if not reject_reasons:
        if score >= 72 or "hot" in tags:
            status = KEEP if score >= 72 else HOT
        elif score >= 58 or "core_index" in tags:
            status = WATCH

    return MarketScanItem(
        symbol=symbol,
        rank=0,
        score=round(_clip(score, 0, 100), 1),
        status=status,
        setup=setup,
        component_scores={
            "liquidity": round(liquidity, 1),
            "trend": round(long_trend, 1),
            "long": round(long_trend, 1),
            "short": round(short_trend, 1),
            "reversal": round(reversal, 1),
            "risk": round(risk, 1),
            "hot_bonus": round(hot_bonus, 1),
            "index_bonus": round(index_bonus, 1),
        },
        reasons=reasons[:6],
        reject_reasons=reject_reasons,
        tags=tags,
        metrics={
            "price": round(last, 2),
            "adv20": round(adv20, 0),
            "ret20_pct": round(ret20, 2),
            "ret60_pct": round(ret60, 2),
            "ret120_pct": round(ret120, 2),
            "drawdown_pct": round(drawdown, 2),
            "vol20_pct": round(vol20, 2),
        },
    )


def _load_bars(symbol: str, timeframe: str) -> pd.DataFrame:
    try:
        from .data_cache import get_bars
        return get_bars(symbol, timeframe)
    except Exception:
        return pd.DataFrame()


def _download_bars(symbol: str, timeframe: str) -> pd.DataFrame:
    try:
        from .data_cache import fetch_and_save
        return fetch_and_save(symbol, timeframe)
    except Exception:
        return pd.DataFrame()


def _clean_bars(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    clean.columns = [str(col).lower() for col in clean.columns]
    if not {"close", "volume"}.issubset(clean.columns):
        return pd.DataFrame()
    clean["close"] = pd.to_numeric(clean["close"], errors="coerce")
    clean["volume"] = pd.to_numeric(clean["volume"], errors="coerce")
    return clean.dropna(subset=["close", "volume"]).reset_index(drop=True)


def _add_symbol(symbols: list[str], tags: dict[str, list[str]], symbol: str, tag: str) -> None:
    symbol = symbol.upper().replace("/", ".")
    if not symbol:
        return
    if symbol not in symbols:
        symbols.append(symbol)
    tags.setdefault(symbol, [])
    if tag not in tags[symbol]:
        tags[symbol].append(tag)


def _status_order(status: str) -> int:
    return {KEEP: 0, HOT: 1, WATCH: 2, REJECT: 3}.get(status, 9)


def _pct(value: float, base: float) -> float:
    return (value - base) / base * 100 if base else 0.0


def _scaled(value: float, low: float, high: float, out_low: float, out_high: float) -> float:
    if value <= low:
        return out_low
    if value >= high:
        return out_high
    ratio = (value - low) / (high - low)
    return out_low + ratio * (out_high - out_low)


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _now_s() -> str:
    return datetime.now(timezone.utc).isoformat()


def _progress(callback: Optional[Callable[[dict], None]], stage: str, current: int, total: int, message: str) -> None:
    if not callback:
        return
    try:
        callback({
            "stage": stage,
            "current": int(current),
            "total": int(total),
            "message": message,
            "updated_at": _now_s(),
        })
    except Exception:
        pass
