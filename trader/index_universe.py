from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from .symbol_master import SourceStatus, parse_symbol_text


_ROOT = Path(__file__).resolve().parents[1]
_STORE = _ROOT / "conf" / "index_universe.json"

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
DOW_URL = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
NASDAQ100_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"

_UA = "stock-minute-ai/1.0 contact@example.com"


@dataclass
class IndexUniverse:
    updated_at: str
    dow: list[str] = field(default_factory=list)
    sp500: list[str] = field(default_factory=list)
    nasdaq100: list[str] = field(default_factory=list)
    source_status: list[SourceStatus] = field(default_factory=list)

    @property
    def core_symbols(self) -> list[str]:
        out: list[str] = []
        for symbol in [*self.dow, *self.sp500, *self.nasdaq100]:
            if symbol and symbol not in out:
                out.append(symbol)
        return out


def update_index_universe(
    *,
    fetcher: Optional[Callable[[str], str]] = None,
    path: Path | str = _STORE,
) -> IndexUniverse:
    fetch = fetcher or _fetch_text
    statuses: list[SourceStatus] = []

    sp500 = _fetch_index("sp500", SP500_URL, fetch, _parse_sp500, statuses)
    dow = _fetch_index("dow", DOW_URL, fetch, _parse_dow, statuses)
    nasdaq100 = _fetch_index("nasdaq100", NASDAQ100_URL, fetch, _parse_nasdaq100, statuses)

    snapshot = IndexUniverse(
        updated_at=_now_s(),
        dow=parse_symbol_text(dow),
        sp500=parse_symbol_text(sp500),
        nasdaq100=parse_symbol_text(nasdaq100),
        source_status=statuses,
    )
    save_index_universe(snapshot, path)
    return snapshot


def load_index_universe(path: Path | str = _STORE) -> IndexUniverse:
    src = Path(path)
    if not src.exists():
        return IndexUniverse(updated_at="", dow=[], sp500=[], nasdaq100=[], source_status=[])
    try:
        payload = json.loads(src.read_text(encoding="utf-8"))
        return IndexUniverse(
            updated_at=payload.get("updated_at", ""),
            dow=list(payload.get("dow", [])),
            sp500=list(payload.get("sp500", [])),
            nasdaq100=list(payload.get("nasdaq100", [])),
            source_status=[SourceStatus(**item) for item in payload.get("source_status", [])],
        )
    except Exception:
        return IndexUniverse(updated_at="", dow=[], sp500=[], nasdaq100=[], source_status=[])


def save_index_universe(snapshot: IndexUniverse, path: Path | str = _STORE) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(snapshot), ensure_ascii=False, indent=2), encoding="utf-8")




def _fetch_index(name: str, url: str, fetch: Callable[[str], str], parser, statuses: list[SourceStatus]) -> list[str]:
    try:
        symbols = parser(fetch(url))
        statuses.append(SourceStatus(source=name, ok=True, count=len(symbols), updated_at=_now_s()))
        return symbols
    except Exception as exc:
        statuses.append(SourceStatus(source=name, ok=False, message=str(exc), updated_at=_now_s()))
        return []


def _parse_sp500(html: str) -> list[str]:
    tables = pd.read_html(StringIO(html))
    for table in tables:
        if "Symbol" in table.columns:
            return _symbols_from_column(table["Symbol"])
    return []


def _parse_dow(html: str) -> list[str]:
    tables = pd.read_html(StringIO(html))
    for table in tables:
        for col in table.columns:
            col_s = str(col).lower()
            if "symbol" in col_s:
                symbols = _symbols_from_column(table[col])
                if len(symbols) >= 20:
                    return symbols[:30]
    return []


def _parse_nasdaq100(html: str) -> list[str]:
    tables = pd.read_html(StringIO(html))
    for table in tables:
        for col in table.columns:
            col_s = str(col).lower()
            if "ticker" in col_s or "symbol" in col_s:
                symbols = _symbols_from_column(table[col])
                if len(symbols) >= 50:
                    return symbols
    return []


def _symbols_from_column(values) -> list[str]:
    out: list[str] = []
    for raw in values:
        symbol = str(raw).strip().upper()
        symbol = re.sub(r"\s.*$", "", symbol).replace("-", ".")
        if symbol and symbol != "NAN" and symbol not in out:
            out.append(symbol)
    return out


def _fetch_text(url: str, timeout: float = 20.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _now_s() -> str:
    return datetime.now(timezone.utc).isoformat()
