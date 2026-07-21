from __future__ import annotations

import csv
import json
import re
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Callable, Iterable, Optional


_ROOT = Path(__file__).resolve().parents[1]
_STORE = _ROOT / "conf" / "symbol_master.json"

NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

_UA = "stock-minute-ai/1.0 contact@example.com"
_BAD_NAME_TOKENS = (
    " Warrant", " Warrants", " Unit", " Units", " Right", " Rights",
    " Preferred", " Preference", " Depositary Share", " Note", " Notes",
    " Debenture", " Bond", " Trust Preferred", "Baby Bond",
)
_GOOD_NAME_TOKENS = (
    "Common Stock", "Class A Common", "Class B Common", "Class C Common",
    "Ordinary Share", "Ordinary Shares", "American Depositary",
    "ADS", "ADR", "Common Shares",
)


@dataclass
class SourceStatus:
    source: str
    ok: bool
    count: int = 0
    message: str = ""
    updated_at: str = ""


@dataclass
class SymbolRecord:
    symbol: str
    name: str = ""
    exchange: str = ""
    source: str = ""
    is_etf: bool = False
    is_test: bool = False
    is_common: bool = True
    cik: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class SymbolMaster:
    updated_at: str
    symbols: list[SymbolRecord]
    source_status: list[SourceStatus] = field(default_factory=list)


def update_symbol_master(
    *,
    fetcher: Optional[Callable[[str], str]] = None,
    path: Path | str = _STORE,
) -> SymbolMaster:
    fetch = fetcher or _fetch_text
    statuses: list[SourceStatus] = []
    records: dict[str, SymbolRecord] = {}

    for source, url, parser in (
        ("nasdaq_listed", NASDAQ_LISTED_URL, _parse_nasdaq_listed),
        ("other_listed", OTHER_LISTED_URL, _parse_other_listed),
    ):
        try:
            parsed = parser(fetch(url))
            for row in parsed:
                records.setdefault(row.symbol, row)
            statuses.append(SourceStatus(source=source, ok=True, count=len(parsed), updated_at=_now_s()))
        except Exception as exc:
            statuses.append(SourceStatus(source=source, ok=False, message=str(exc), updated_at=_now_s()))

    sec_map: dict[str, str] = {}
    try:
        sec_map = _parse_sec_company_tickers(fetch(SEC_COMPANY_TICKERS_URL))
        statuses.append(SourceStatus(source="sec_company_tickers", ok=True, count=len(sec_map), updated_at=_now_s()))
    except Exception as exc:
        statuses.append(SourceStatus(source="sec_company_tickers", ok=False, message=str(exc), updated_at=_now_s()))

    for symbol, cik in sec_map.items():
        if symbol in records:
            records[symbol].cik = cik

    master = SymbolMaster(
        updated_at=_now_s(),
        symbols=sorted(records.values(), key=lambda item: item.symbol),
        source_status=statuses,
    )
    save_symbol_master(master, path)
    return master


def load_symbol_master(path: Path | str = _STORE) -> SymbolMaster:
    src = Path(path)
    if not src.exists():
        return SymbolMaster(updated_at="", symbols=[], source_status=[])
    try:
        payload = json.loads(src.read_text(encoding="utf-8"))
        return SymbolMaster(
            updated_at=payload.get("updated_at", ""),
            symbols=[SymbolRecord(**item) for item in payload.get("symbols", [])],
            source_status=[SourceStatus(**item) for item in payload.get("source_status", [])],
        )
    except Exception:
        return SymbolMaster(updated_at="", symbols=[], source_status=[])


def save_symbol_master(master: SymbolMaster, path: Path | str = _STORE) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(master), ensure_ascii=False, indent=2), encoding="utf-8")


def common_equity_symbols(master: Optional[SymbolMaster] = None, *, limit: int = 10000) -> list[str]:
    master = master or load_symbol_master()
    symbols = [
        item.symbol for item in master.symbols
        if item.is_common and not item.is_etf and not item.is_test and _valid_symbol(item.symbol)
    ]
    return symbols[:limit]


def parse_symbol_text(text: str | Iterable[str] | None) -> list[str]:
    if text is None:
        return []
    raw = text if not isinstance(text, str) else re.split(r"[\s,;，]+", text)
    out: list[str] = []
    for item in raw:
        symbol = str(item or "").strip().upper().replace("/", ".")
        if symbol and symbol not in out:
            out.append(symbol)
    return out


def _parse_nasdaq_listed(text: str) -> list[SymbolRecord]:
    rows: list[SymbolRecord] = []
    reader = csv.DictReader(_clean_lines(text), delimiter="|")
    for row in reader:
        symbol = (row.get("Symbol") or "").strip().upper()
        if not symbol or symbol == "FILE CREATION TIME":
            continue
        name = (row.get("Security Name") or "").strip()
        is_etf = (row.get("ETF") or "N").upper() == "Y"
        is_test = (row.get("Test Issue") or "N").upper() == "Y"
        rows.append(SymbolRecord(
            symbol=symbol,
            name=name,
            exchange="NASDAQ",
            source="nasdaq_listed",
            is_etf=is_etf,
            is_test=is_test,
            is_common=_is_common_equity(symbol, name, is_etf, is_test),
            metadata={
                "market_category": row.get("Market Category", ""),
                "financial_status": row.get("Financial Status", ""),
            },
        ))
    return rows


def _parse_other_listed(text: str) -> list[SymbolRecord]:
    rows: list[SymbolRecord] = []
    reader = csv.DictReader(_clean_lines(text), delimiter="|")
    for row in reader:
        symbol = (row.get("ACT Symbol") or "").strip().upper()
        if not symbol or symbol == "FILE CREATION TIME":
            continue
        name = (row.get("Security Name") or "").strip()
        is_etf = (row.get("ETF") or "N").upper() == "Y"
        is_test = (row.get("Test Issue") or "N").upper() == "Y"
        rows.append(SymbolRecord(
            symbol=symbol,
            name=name,
            exchange=(row.get("Exchange") or "").strip(),
            source="other_listed",
            is_etf=is_etf,
            is_test=is_test,
            is_common=_is_common_equity(symbol, name, is_etf, is_test),
            metadata={"cqs_symbol": row.get("CQS Symbol", "")},
        ))
    return rows


def _parse_sec_company_tickers(text: str) -> dict[str, str]:
    payload = json.loads(text)
    out: dict[str, str] = {}
    items = payload.values() if isinstance(payload, dict) else payload
    for item in items:
        ticker = str(item.get("ticker", "")).upper().replace("-", ".")
        cik = item.get("cik_str")
        if ticker and cik is not None:
            out[ticker] = str(cik).zfill(10)
    return out


def _is_common_equity(symbol: str, name: str, is_etf: bool, is_test: bool) -> bool:
    if is_etf or is_test or not _valid_symbol(symbol):
        return False
    upper_name = f" {name.upper()} "
    if any(token.upper() in upper_name for token in _BAD_NAME_TOKENS):
        return False
    if any(token.upper() in upper_name for token in _GOOD_NAME_TOKENS):
        return True
    return True


def _valid_symbol(symbol: str) -> bool:
    if not symbol or len(symbol) > 8:
        return False
    if any(ch in symbol for ch in ("$", "^", "+", "=")):
        return False
    suffix = symbol.split(".")[-1] if "." in symbol else ""
    if suffix in {"W", "WS", "WT", "U", "R", "P", "PR"}:
        return False
    return True


def _clean_lines(text: str) -> StringIO:
    lines = [line for line in text.splitlines() if line and not line.startswith("File Creation Time")]
    return StringIO("\n".join(lines))


def _fetch_text(url: str, timeout: float = 20.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _now_s() -> str:
    return datetime.now(timezone.utc).isoformat()
