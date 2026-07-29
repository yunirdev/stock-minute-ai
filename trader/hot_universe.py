from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from .symbol_master import SourceStatus, parse_symbol_text


_ROOT = Path(__file__).resolve().parents[1]
_STORE = _ROOT / "conf" / "hot_universe.json"

_DEFAULT_RSS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EGSPC&region=US&lang=en-US",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
]
_SYMBOL_RE = re.compile(r"\b[A-Z]{1,5}(?:\.[A-Z])?\b")
_COMMON_WORDS = {
    "A", "I", "AI", "CEO", "CFO", "USA", "US", "NYSE", "ETF", "SEC", "Fed",
    "THE", "AND", "FOR", "ARE", "NEW", "IPO", "GDP", "CPI", "FOMC",
}


@dataclass
class HotSymbol:
    symbol: str
    score: float
    sources: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


@dataclass
class HotUniverse:
    updated_at: str
    symbols: list[HotSymbol]
    source_status: list[SourceStatus] = field(default_factory=list)


def build_hot_universe(
    *,
    base_symbols: Optional[Iterable[str]] = None,
    rss_urls: Optional[list[str]] = None,
    price_mover_symbols: Optional[Iterable[str]] = None,
    limit: int = 80,
    save: bool = True,
    path: Path | str = _STORE,
) -> HotUniverse:
    scores: dict[str, HotSymbol] = {}
    statuses: list[SourceStatus] = []
    base_set = set(parse_symbol_text(base_symbols)) if base_symbols else set()

    rss_symbols = _rss_hot_symbols(rss_urls or _DEFAULT_RSS, base_set, statuses)
    for symbol, count in rss_symbols.items():
        _add(scores, symbol, min(30.0, 8.0 * count), "rss_news", f"mentioned in {count} headlines")

    mover_symbols = parse_symbol_text(price_mover_symbols)
    if mover_symbols:
        statuses.append(SourceStatus(
            source="price_move",
            ok=False,
            count=0,
            message="disabled in strict mode; price momentum is scored by market_scan fresh bars",
            updated_at=_now_s(),
        ))

    rows = sorted(scores.values(), key=lambda row: (-row.score, row.symbol))[:limit]
    snapshot = HotUniverse(updated_at=_now_s(), symbols=rows, source_status=statuses)
    if save:
        save_hot_universe(snapshot, path)
    return snapshot


def load_hot_universe(path: Path | str = _STORE) -> HotUniverse:
    src = Path(path)
    if not src.exists():
        return HotUniverse(updated_at="", symbols=[], source_status=[])
    try:
        payload = json.loads(src.read_text(encoding="utf-8"))
        return HotUniverse(
            updated_at=payload.get("updated_at", ""),
            symbols=[HotSymbol(**item) for item in payload.get("symbols", [])],
            source_status=[SourceStatus(**item) for item in payload.get("source_status", [])],
        )
    except Exception:
        return HotUniverse(updated_at="", symbols=[], source_status=[])


def save_hot_universe(snapshot: HotUniverse, path: Path | str = _STORE) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(snapshot), ensure_ascii=False, indent=2), encoding="utf-8")


def hot_symbols(snapshot: Optional[HotUniverse] = None, *, limit: int = 80) -> list[str]:
    snapshot = snapshot or load_hot_universe()
    return [row.symbol for row in sorted(snapshot.symbols, key=lambda row: (-row.score, row.symbol))[:limit]]


def _rss_hot_symbols(urls: list[str], allowed: set[str], statuses: list[SourceStatus]) -> dict[str, int]:
    counts: dict[str, int] = {}
    try:
        import feedparser
    except Exception as exc:
        statuses.append(SourceStatus(source="rss_news", ok=False, message=str(exc), updated_at=_now_s()))
        return counts

    ok_count = 0
    for url in urls:
        try:
            feed = feedparser.parse(url)
            entries = getattr(feed, "entries", []) or []
            ok_count += len(entries)
            for entry in entries:
                title = f"{getattr(entry, 'title', '')} {getattr(entry, 'summary', '')}"
                for symbol in _extract_symbols(title, allowed):
                    counts[symbol] = counts.get(symbol, 0) + 1
        except Exception:
            continue
    statuses.append(SourceStatus(source="rss_news", ok=True, count=ok_count, updated_at=_now_s()))
    return counts


def _extract_symbols(text: str, allowed: set[str]) -> list[str]:
    out: list[str] = []
    for raw in _SYMBOL_RE.findall(text.upper()):
        symbol = raw.strip().upper()
        if symbol in _COMMON_WORDS:
            continue
        if allowed and symbol not in allowed:
            continue
        if symbol and symbol not in out:
            out.append(symbol)
    return out


def _add(scores: dict[str, HotSymbol], symbol: str, score: float, source: str, reason: str) -> None:
    symbol = symbol.upper()
    row = scores.setdefault(symbol, HotSymbol(symbol=symbol, score=0.0))
    row.score += score
    if source not in row.sources:
        row.sources.append(source)
    if reason not in row.reasons:
        row.reasons.append(reason)


def _now_s() -> str:
    return datetime.now(timezone.utc).isoformat()
