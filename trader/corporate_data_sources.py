"""SEC EDGAR corporate, financial, disclosure, and insider fact adapters."""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
from collections import deque
from datetime import date, datetime, timezone
from threading import Lock
from typing import Any, Callable, Mapping

from .data_hub import (
    AdapterResult,
    DataDomain,
    SourceRegistry,
    SourceSpec,
)

_SEC_DATA_BASE = "https://data.sec.gov"
_SEC_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"
_INSIDER_FORMS = frozenset({"3", "3/A", "4", "4/A", "5", "5/A"})

CorporateLoader = Callable[[str, str], Mapping[str, Any]]
CikResolver = Callable[[str], str | None]


class SlidingWindowRateLimiter:
    """Fail-fast fixed-window guard for upstream request quotas."""

    def __init__(
        self,
        *,
        max_requests: int,
        window_seconds: float,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if max_requests <= 0:
            raise ValueError("DATA_RATE_LIMIT_MAX_REQUESTS_INVALID")
        if window_seconds <= 0:
            raise ValueError("DATA_RATE_LIMIT_WINDOW_INVALID")
        self._max_requests = max_requests
        self._window_seconds = float(window_seconds)
        self._clock = clock or time.monotonic
        self._calls: deque[float] = deque()
        self._lock = Lock()

    def acquire(self) -> None:
        now = float(self._clock())
        with self._lock:
            cutoff = now - self._window_seconds
            while self._calls and self._calls[0] <= cutoff:
                self._calls.popleft()
            if len(self._calls) >= self._max_requests:
                raise ValueError("DATA_SOURCE_RATE_LIMITED")
            self._calls.append(now)


class SecEdgarClient:
    """Small EDGAR JSON client; adapters own normalization and quality rules."""

    def __init__(
        self,
        *,
        user_agent: str | None = None,
        timeout_seconds: float = 10.0,
        rate_limiter: SlidingWindowRateLimiter | None = None,
        json_loader: Callable[[str, Mapping[str, str], float], Any] | None = None,
    ) -> None:
        self._user_agent = (
            user_agent
            or os.getenv("SEC_USER_AGENT", "")
        ).strip()
        if not self._user_agent:
            raise ValueError("DATA_SEC_USER_AGENT_REQUIRED")
        if timeout_seconds <= 0:
            raise ValueError("DATA_SEC_TIMEOUT_INVALID")
        self._timeout_seconds = float(timeout_seconds)
        # SEC permits at most 10 requests/second. Keep headroom by default.
        self._rate_limiter = rate_limiter or SlidingWindowRateLimiter(
            max_requests=8,
            window_seconds=1.0,
        )
        self._json_loader = json_loader or self._load_json

    @staticmethod
    def _load_json(
        url: str,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> Any:
        request = urllib.request.Request(url, headers=dict(headers))
        with urllib.request.urlopen(
            request,
            timeout=timeout_seconds,
        ) as response:
            return json.loads(response.read())

    def _get(self, url: str) -> Any:
        self._rate_limiter.acquire()
        return self._json_loader(
            url,
            {
                "User-Agent": self._user_agent,
                "Accept": "application/json",
                "Accept-Encoding": "gzip, deflate",
            },
            self._timeout_seconds,
        )

    def fetch_bundle(self, symbol: str, cik: str) -> Mapping[str, Any]:
        normalized_cik = _normalize_cik(cik)
        return {
            "symbol": symbol.upper(),
            "cik": normalized_cik,
            "companyfacts": self._get(
                f"{_SEC_DATA_BASE}/api/xbrl/companyfacts/"
                f"CIK{normalized_cik}.json"
            ),
            "submissions": self._get(
                f"{_SEC_DATA_BASE}/submissions/CIK{normalized_cik}.json"
            ),
        }


def _normalize_cik(value: Any) -> str:
    text = str(value or "").strip()
    if not text.isdigit():
        raise ValueError("DATA_SEC_CIK_INVALID")
    return text.zfill(10)


def _utc_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        if len(text) == 14 and text.isdigit():
            parsed = datetime.strptime(text, "%Y%m%d%H%M%S")
        elif len(text) == 8 and text.isdigit():
            parsed = datetime.strptime(text, "%Y%m%d")
        elif len(text) == 10:
            parsed = datetime.combine(date.fromisoformat(text), datetime.min.time())
        else:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _fact_identity(
    taxonomy: str,
    concept: str,
    unit: str,
    item: Mapping[str, Any],
) -> str:
    parts = (
        taxonomy,
        concept,
        unit,
        str(item.get("start") or ""),
        str(item.get("end") or ""),
        str(item.get("fy") or ""),
        str(item.get("fp") or ""),
    )
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:24]


def _normalize_financial_facts(
    raw: Mapping[str, Any],
    *,
    concepts: set[str] | None,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for taxonomy, taxonomy_facts in (raw.get("facts") or {}).items():
        if not isinstance(taxonomy_facts, Mapping):
            continue
        for concept, definition in taxonomy_facts.items():
            if concepts is not None and concept not in concepts:
                continue
            if not isinstance(definition, Mapping):
                continue
            for unit, entries in (definition.get("units") or {}).items():
                for item in entries or ():
                    if not isinstance(item, Mapping):
                        continue
                    filed_at = _utc_datetime(item.get("filed"))
                    accession = str(item.get("accn") or "").strip()
                    if filed_at is None or not accession or "val" not in item:
                        continue
                    fact_id = _fact_identity(
                        str(taxonomy),
                        str(concept),
                        str(unit),
                        item,
                    )
                    grouped.setdefault(fact_id, []).append(
                        {
                            "value": item["val"],
                            "unit": str(unit),
                            "period_start": item.get("start"),
                            "period_end": item.get("end"),
                            "fiscal_year": item.get("fy"),
                            "fiscal_period": item.get("fp"),
                            "form": str(item.get("form") or ""),
                            "frame": item.get("frame"),
                            "accession_number": accession,
                            "filed_at": filed_at.isoformat(),
                            "as_of": filed_at.isoformat(),
                            "is_amendment": str(item.get("form") or "").endswith("/A"),
                            "source": "sec_edgar_companyfacts",
                        }
                    )

            for fact_id, revisions in tuple(grouped.items()):
                if revisions and "taxonomy" not in revisions[0]:
                    for revision in revisions:
                        revision.update(
                            {
                                "taxonomy": str(taxonomy),
                                "concept": str(concept),
                                "label": str(definition.get("label") or ""),
                                "description": str(
                                    definition.get("description") or ""
                                ),
                            }
                        )

    current_facts: list[dict[str, Any]] = []
    for fact_id, revisions in grouped.items():
        revisions.sort(
            key=lambda item: (
                item["filed_at"],
                item["accession_number"],
            )
        )
        current = dict(revisions[-1])
        current.update(
            {
                "fact_id": fact_id,
                "revision": len(revisions) - 1,
                "revision_count": len(revisions),
                "revisions": [dict(item) for item in revisions],
            }
        )
        current_facts.append(current)
    current_facts.sort(
        key=lambda item: (
            item["taxonomy"],
            item["concept"],
            item["unit"],
            str(item.get("period_end") or ""),
            item["fact_id"],
        )
    )
    return current_facts


def _parallel_value(
    recent: Mapping[str, Any],
    name: str,
    index: int,
) -> Any:
    values = recent.get(name)
    if not isinstance(values, list) or index >= len(values):
        return None
    return values[index]


def _normalize_filings(
    raw: Mapping[str, Any],
    cik: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    recent = ((raw.get("filings") or {}).get("recent") or {})
    if not isinstance(recent, Mapping):
        return [], []
    accessions = recent.get("accessionNumber")
    if not isinstance(accessions, list):
        return [], []

    disclosures: list[dict[str, Any]] = []
    insider_filings: list[dict[str, Any]] = []
    seen: set[str] = set()
    archive_cik = str(int(cik))
    for index, accession_value in enumerate(accessions):
        accession = str(accession_value or "").strip()
        form = str(_parallel_value(recent, "form", index) or "").strip().upper()
        if not accession or not form or accession in seen:
            continue
        accepted_at = (
            _utc_datetime(_parallel_value(recent, "acceptanceDateTime", index))
            or _utc_datetime(_parallel_value(recent, "filingDate", index))
        )
        if accepted_at is None:
            continue
        primary_document = str(
            _parallel_value(recent, "primaryDocument", index) or ""
        ).strip()
        accession_path = accession.replace("-", "")
        document_url = (
            f"{_SEC_ARCHIVES_BASE}/{archive_cik}/{accession_path}/"
            f"{primary_document}"
            if primary_document
            else f"{_SEC_ARCHIVES_BASE}/{archive_cik}/{accession_path}/"
        )
        record = {
            "accession_number": accession,
            "form": form,
            "filed_at": accepted_at.isoformat(),
            "as_of": accepted_at.isoformat(),
            "filing_date": _parallel_value(recent, "filingDate", index),
            "report_date": _parallel_value(recent, "reportDate", index),
            "items": _parallel_value(recent, "items", index),
            "primary_document": primary_document,
            "document_url": document_url,
            "is_amendment": form.endswith("/A"),
            "source": "sec_edgar_submissions",
        }
        seen.add(accession)
        if form in _INSIDER_FORMS:
            insider_filings.append(record)
        else:
            disclosures.append(record)

    def sort_key(item: Mapping[str, Any]) -> tuple[str, str]:
        return item["filed_at"], item["accession_number"]

    disclosures.sort(key=sort_key, reverse=True)
    insider_filings.sort(key=sort_key, reverse=True)
    return disclosures, insider_filings


def sec_edgar_corporate_adapter(
    loader: CorporateLoader,
    *,
    cik_resolver: CikResolver | None = None,
):
    """Normalize SEC facts without inference or LLM-created replacements."""

    def fetch(request) -> AdapterResult:
        cik_value = request.params.get("cik")
        if cik_value is None and cik_resolver is not None:
            cik_value = cik_resolver(request.key)
        if cik_value is None:
            raise ValueError("DATA_SEC_CIK_REQUIRED")
        cik = _normalize_cik(cik_value)
        raw = loader(request.key, cik)
        if not isinstance(raw, Mapping):
            raise ValueError("DATA_SEC_PAYLOAD_INVALID")

        concepts_param = request.params.get("concepts")
        concepts = None
        if concepts_param is not None:
            if isinstance(concepts_param, str):
                concepts = {
                    item.strip()
                    for item in concepts_param.split(",")
                    if item.strip()
                }
            else:
                concepts = {
                    str(item).strip()
                    for item in concepts_param
                    if str(item).strip()
                }
        companyfacts = raw.get("companyfacts") or {}
        submissions = raw.get("submissions") or {}
        if not isinstance(companyfacts, Mapping) or not isinstance(
            submissions,
            Mapping,
        ):
            raise ValueError("DATA_SEC_PAYLOAD_INVALID")

        financial_facts = _normalize_financial_facts(
            companyfacts,
            concepts=concepts,
        )
        disclosures, insider_filings = _normalize_filings(
            submissions,
            cik,
        )
        sections = {
            "financial_facts": financial_facts,
            "disclosures": disclosures,
            "insider_filings": insider_filings,
        }
        missing_sections = [
            name for name, records in sections.items() if not records
        ]
        if len(missing_sections) == len(sections):
            raise ValueError("DATA_CORPORATE_FACTS_EMPTY")

        available_times = [
            _utc_datetime(item["as_of"])
            for records in sections.values()
            for item in records
        ]
        as_of = max(
            item for item in available_times if item is not None
        )
        entity_name = str(
            companyfacts.get("entityName")
            or submissions.get("name")
            or ""
        )
        quality_score = max(0.0, 1.0 - 0.2 * len(missing_sections))
        return AdapterResult(
            payload={
                "symbol": request.key,
                "cik": cik,
                "entity_name": entity_name,
                **sections,
                "missing_sections": missing_sections,
                "fact_generation": "SOURCE_ONLY",
                "execution_eligible": False,
            },
            as_of=as_of,
            quality_score=quality_score,
            metadata={
                "upstream": "sec_edgar",
                "authoritative": True,
                "revision_policy": "LATEST_FILED_WITH_FULL_HISTORY",
                "missing_sections": missing_sections,
                "llm_fact_fill": False,
            },
        )

    return fetch


def register_sec_edgar_corporate(
    registry: SourceRegistry,
    *,
    client: SecEdgarClient,
    cik_resolver: CikResolver | None = None,
) -> None:
    registry.register(
        SourceSpec(
            source_id="sec_edgar_corporate",
            domain=DataDomain.CORPORATE,
            adapter=sec_edgar_corporate_adapter(
                client.fetch_bundle,
                cik_resolver=cik_resolver,
            ),
            priority=0,
            timeout_seconds=25.0,
            ttl_seconds=900.0,
            max_stale_seconds=86_400.0,
            required_fields=(
                "symbol",
                "cik",
                "financial_facts",
                "disclosures",
                "insider_filings",
                "missing_sections",
                "fact_generation",
                "execution_eligible",
            ),
        )
    )
