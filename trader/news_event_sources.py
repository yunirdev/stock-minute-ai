"""Unified research-only news, event, and calendar Data Hub adapters."""
from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import urlsplit, urlunsplit
from zoneinfo import ZoneInfo

from .data_hub import (
    AdapterResult,
    DataDomain,
    DataRequest,
    SourceRegistry,
    SourceSpec,
)

NewsLoader = Callable[[DataRequest], Iterable[Any] | Mapping[str, Any] | None]

_EASTERN = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class NewsEventSource:
    source_id: str
    loader: NewsLoader
    priority: int
    quality_weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("DATA_NEWS_SOURCE_ID_REQUIRED")
        if self.priority < 0:
            raise ValueError("DATA_NEWS_SOURCE_PRIORITY_INVALID")
        if (
            not math.isfinite(self.quality_weight)
            or self.quality_weight <= 0
        ):
            raise ValueError("DATA_NEWS_SOURCE_WEIGHT_INVALID")


def _mapping_or_attrs(item: Any, *names: str) -> Any:
    if isinstance(item, Mapping):
        for name in names:
            if name in item:
                return item[name]
    for name in names:
        if hasattr(item, name):
            return getattr(item, name)
    return None


def _content_value(item: Any, *names: str) -> Any:
    value = _mapping_or_attrs(item, *names)
    if value is not None:
        return value
    content = _mapping_or_attrs(item, "content")
    if isinstance(content, Mapping):
        return _mapping_or_attrs(content, *names)
    return None


def _aware_datetime(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, (int, float)) and math.isfinite(float(value)):
        parsed = datetime.fromtimestamp(float(value), tz=timezone.utc)
    else:
        text = str(value).strip()
        if not text:
            return None
        if text.isdigit():
            parsed = datetime.fromtimestamp(float(text), tz=timezone.utc)
        else:
            try:
                parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            except ValueError:
                try:
                    parsed = parsedate_to_datetime(text)
                except (TypeError, ValueError):
                    return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


def _request_datetime(value: Any, code: str) -> datetime:
    parsed = _aware_datetime(value)
    if parsed is None:
        raise ValueError(code)
    return parsed


def request_window(request: DataRequest) -> tuple[datetime, datetime]:
    since = (
        _request_datetime(
            request.params["since"],
            "DATA_NEWS_SINCE_TIMEZONE_REQUIRED",
        )
        if "since" in request.params
        else request.requested_at - timedelta(hours=24)
    )
    until = (
        _request_datetime(
            request.params["until"],
            "DATA_NEWS_UNTIL_TIMEZONE_REQUIRED",
        )
        if "until" in request.params
        else request.requested_at + timedelta(days=7)
    )
    if since > until:
        raise ValueError("DATA_NEWS_WINDOW_INVALID")
    return since, until


def polling_news_loader(poller: Any) -> NewsLoader:
    """Adapt existing ``poll(since=...)`` sources without changing their path."""

    def load(request: DataRequest) -> Iterable[Any]:
        since, _ = request_window(request)
        return poller.poll(since=since)

    return load


def _calendar_datetime(item: Any) -> tuple[datetime | None, str]:
    direct = _aware_datetime(
        _mapping_or_attrs(item, "event_at", "event_time")
    )
    raw_date = _mapping_or_attrs(item, "event_date", "date")
    if direct is not None:
        event_date = str(raw_date or direct.astimezone(_EASTERN).date())
        return direct, event_date[:10]
    if raw_date is None:
        return None, ""
    try:
        event_date_value = date.fromisoformat(str(raw_date)[:10])
    except ValueError:
        return None, ""
    time_text = str(
        _mapping_or_attrs(item, "time_str", "time") or ""
    ).lower()
    match = re.search(r"(\d{1,2}):(\d{2})", time_text)
    if match:
        hour, minute = int(match.group(1)), int(match.group(2))
    elif any(token in time_text for token in ("盘前", "bmo")):
        hour, minute = 8, 0
    elif any(token in time_text for token in ("盘后", "amc")):
        hour, minute = 16, 30
    elif any(token in time_text for token in ("盘中", "dmh")):
        hour, minute = 12, 0
    else:
        hour, minute = 12, 0
    local = datetime.combine(
        event_date_value,
        time(hour=hour, minute=minute),
        tzinfo=_EASTERN,
    )
    return local.astimezone(timezone.utc), event_date_value.isoformat()


def _normalize_url(value: Any) -> str:
    if isinstance(value, Mapping):
        value = value.get("url") or value.get("href") or ""
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        parts = urlsplit(text)
    except ValueError:
        return text
    return urlunsplit(
        (
            parts.scheme.lower(),
            parts.netloc.lower(),
            parts.path.rstrip("/"),
            "",
            "",
        )
    )


def _canonical_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"[^\w]+", " ", text).strip()


def _normalize_item(
    item: Any,
    *,
    source_id: str,
    request: DataRequest,
) -> dict[str, Any]:
    title = str(
        _content_value(
            item,
            "title",
            "headline",
            "title_en",
            "event",
            "content_text",
        )
        or ""
    ).strip()
    if not title:
        raise ValueError("DATA_NEWS_ITEM_TITLE_REQUIRED")

    raw_kind = str(
        _mapping_or_attrs(item, "kind", "type") or ""
    ).strip().upper()
    category = str(
        _mapping_or_attrs(item, "category") or raw_kind or "news"
    ).strip().lower()
    has_calendar_date = _mapping_or_attrs(
        item,
        "event_at",
        "event_time",
        "event_date",
        "date",
    ) is not None
    if raw_kind in {"CALENDAR", "EVENT"} or has_calendar_date:
        kind = "CALENDAR" if category in {
            "earnings",
            "economic",
            "fomc",
            "holiday",
        } else "EVENT"
    else:
        kind = "NEWS"

    published_at = _aware_datetime(
        _content_value(
            item,
            "published_at",
            "ts",
            "datetime",
            "providerPublishTime",
            "display_time",
            "pubDate",
            "published",
            "updated",
        )
    )
    event_at, event_date = (
        _calendar_datetime(item)
        if kind != "NEWS"
        else (None, "")
    )
    if kind == "NEWS" and published_at is None:
        raise ValueError("DATA_NEWS_ITEM_PUBLISHED_AT_REQUIRED")
    if kind != "NEWS" and event_at is None:
        raise ValueError("DATA_EVENT_TIME_REQUIRED")

    explicit_as_of = _mapping_or_attrs(item, "as_of", "fetched_at")
    as_of = _aware_datetime(explicit_as_of)
    if explicit_as_of is not None and as_of is None:
        raise ValueError("DATA_NEWS_ITEM_AS_OF_TIMEZONE_REQUIRED")
    if as_of is None:
        as_of = published_at if kind == "NEWS" else request.requested_at
    if as_of is None:
        raise ValueError("DATA_NEWS_ITEM_AS_OF_REQUIRED")
    if as_of > request.requested_at:
        raise ValueError("DATA_NEWS_ITEM_AS_OF_FUTURE")

    boundary_time = published_at if kind == "NEWS" else event_at
    if boundary_time is None:
        raise ValueError("DATA_NEWS_ITEM_TIME_REQUIRED")

    symbol_value = _mapping_or_attrs(item, "symbol", "ticker")
    symbol = str(symbol_value or "").strip().upper() or None
    severity_value = _mapping_or_attrs(item, "severity", "impact_score")
    try:
        severity = float(severity_value or 0.0)
    except (TypeError, ValueError):
        severity = 0.0
    if not math.isfinite(severity):
        severity = 0.0

    upstream_id = str(
        _mapping_or_attrs(item, "upstream_id", "id", "event_id") or ""
    ).strip()
    return {
        "item_id": "",
        "upstream_id": upstream_id,
        "kind": kind,
        "category": category,
        "symbol": symbol,
        "title": title,
        "summary": str(
            _content_value(
                item,
                "summary",
                "description",
                "note",
                "content_text",
            )
            or ""
        ).strip(),
        "url": _normalize_url(
            _content_value(
                item,
                "url",
                "link",
                "document_url",
                "canonicalUrl",
                "clickThroughUrl",
            )
        ),
        "published_at": (
            published_at.isoformat() if published_at is not None else None
        ),
        "event_at": event_at.isoformat() if event_at is not None else None,
        "event_date": event_date or None,
        "as_of": as_of.isoformat(),
        "severity": max(0.0, min(severity, 1.0)),
        "source": source_id,
        "sources": [source_id],
        "execution_eligible": False,
        "_boundary_time": boundary_time,
    }


def _dedupe_key(item: Mapping[str, Any]) -> str:
    if item["kind"] != "NEWS":
        if item.get("symbol") and item.get("category") == "earnings":
            identity = (
                "calendar",
                "earnings",
                item["symbol"],
                item.get("event_date") or "",
            )
        else:
            identity = (
                item["kind"].lower(),
                item.get("category") or "",
                item.get("event_date") or "",
                _canonical_text(item["title"]),
            )
    else:
        identity = (
            "news",
            item.get("symbol") or "",
            str(item["published_at"])[:10],
            _canonical_text(item["title"]),
        )
    return "\x1f".join(identity)


def _material_conflicts(
    selected: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> list[dict[str, Any]]:
    differences = []
    for field_name in ("kind", "category", "symbol", "event_at"):
        left = selected.get(field_name)
        right = candidate.get(field_name)
        if left != right and (left is not None or right is not None):
            differences.append(
                {
                    "field": field_name,
                    "selected": left,
                    "candidate": right,
                }
            )
    return differences


def _coerce_items(raw: Any) -> list[Any]:
    if raw is None:
        return []
    if isinstance(raw, Mapping):
        nested = raw.get("items")
        if isinstance(nested, Iterable) and not isinstance(
            nested,
            (str, bytes, Mapping),
        ):
            return list(nested)
        return [raw]
    if isinstance(raw, (str, bytes)):
        raise ValueError("DATA_NEWS_SOURCE_PAYLOAD_INVALID")
    try:
        return list(raw)
    except TypeError as exc:
        raise ValueError("DATA_NEWS_SOURCE_PAYLOAD_INVALID") from exc


def multi_source_news_event_adapter(
    sources: Iterable[NewsEventSource],
):
    ordered_sources = tuple(
        sorted(sources, key=lambda source: (source.priority, source.source_id))
    )
    if not ordered_sources:
        raise ValueError("DATA_NEWS_SOURCES_REQUIRED")
    priority_by_source = {
        source.source_id: source.priority for source in ordered_sources
    }

    def fetch(request: DataRequest) -> AdapterResult:
        since, until = request_window(request)
        statuses: list[dict[str, Any]] = []
        collected: list[dict[str, Any]] = []
        successful_weight = 0.0
        total_weight = sum(
            source.quality_weight for source in ordered_sources
        )

        for source in ordered_sources:
            try:
                raw_items = _coerce_items(source.loader(request))
            except Exception as exc:
                code = (
                    str(exc)
                    if isinstance(exc, ValueError)
                    and str(exc).startswith("DATA_")
                    else type(exc).__name__
                )
                statuses.append(
                    {
                        "source_id": source.source_id,
                        "status": "FAILED",
                        "item_count": 0,
                        "dropped_count": 0,
                        "failure_code": code,
                    }
                )
                continue

            normalized: list[dict[str, Any]] = []
            dropped_codes: list[str] = []
            for raw_item in raw_items:
                try:
                    item = _normalize_item(
                        raw_item,
                        source_id=source.source_id,
                        request=request,
                    )
                except ValueError as exc:
                    dropped_codes.append(str(exc))
                    continue
                boundary_time = item["_boundary_time"]
                if since <= boundary_time <= until:
                    normalized.append(item)
            collected.extend(normalized)
            drop_ratio = (
                len(dropped_codes) / len(raw_items) if raw_items else 0.0
            )
            successful_weight += source.quality_weight * (1.0 - drop_ratio)
            statuses.append(
                {
                    "source_id": source.source_id,
                    "status": "DEGRADED" if dropped_codes else "OK",
                    "item_count": len(normalized),
                    "dropped_count": len(dropped_codes),
                    "failure_code": (
                        "DATA_NEWS_ITEMS_DROPPED" if dropped_codes else ""
                    ),
                    "dropped_codes": sorted(set(dropped_codes)),
                }
            )

        if not any(status["status"] != "FAILED" for status in statuses):
            raise ValueError("DATA_NEWS_ALL_SOURCES_FAILED")

        by_key: dict[str, dict[str, Any]] = {}
        conflicts: list[dict[str, Any]] = []
        for candidate in collected:
            key = _dedupe_key(candidate)
            existing = by_key.get(key)
            if existing is None:
                by_key[key] = candidate
                continue
            differences = _material_conflicts(existing, candidate)
            selected = existing
            if (
                priority_by_source[candidate["source"]]
                < priority_by_source[existing["source"]]
            ):
                selected = candidate
            selected = dict(selected)
            selected["sources"] = sorted(
                set(existing["sources"]) | set(candidate["sources"]),
                key=lambda source_id: (
                    priority_by_source[source_id],
                    source_id,
                ),
            )
            by_key[key] = selected
            if differences:
                conflicts.append(
                    {
                        "dedupe_key": key,
                        "selected_source": selected["source"],
                        "candidate_source": (
                            candidate["source"]
                            if selected["source"] == existing["source"]
                            else existing["source"]
                        ),
                        "differences": differences,
                    }
                )

        items = []
        for key, item in by_key.items():
            normalized = {
                field_name: value
                for field_name, value in item.items()
                if not field_name.startswith("_")
            }
            normalized["item_id"] = (
                "news-item-"
                + hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
            )
            items.append(normalized)
        items.sort(
            key=lambda item: (
                item["event_at"] or item["published_at"] or "",
                item["item_id"],
            )
        )

        quality_score = successful_weight / total_weight
        if conflicts:
            quality_score *= 0.9
        quality_score = max(0.0, min(quality_score, 1.0))
        item_times = [
            _aware_datetime(item["as_of"])
            for item in items
            if item.get("as_of")
        ]
        as_of = max(
            (value for value in item_times if value is not None),
            default=request.requested_at,
        )
        return AdapterResult(
            payload={
                "items": items,
                "source_statuses": statuses,
                "conflicts": conflicts,
                "window": {
                    "since": since.isoformat(),
                    "until": until.isoformat(),
                },
                "directive_capability": "RESEARCH_ONLY",
                "execution_eligible": False,
            },
            as_of=as_of,
            quality_score=quality_score,
            metadata={
                "source_count": len(ordered_sources),
                "deduplicated_count": len(collected) - len(items),
                "conflict_count": len(conflicts),
                "execution_eligible": False,
            },
        )

    return fetch


def register_news_event_sources(
    registry: SourceRegistry,
    *,
    finnhub_loader: NewsLoader,
    nasdaq_loader: NewsLoader,
    wallstreetcn_loader: NewsLoader,
    yahoo_loader: NewsLoader,
    rss_loader: NewsLoader,
) -> None:
    sources = (
        NewsEventSource("nasdaq_calendar", nasdaq_loader, priority=0),
        NewsEventSource("finnhub", finnhub_loader, priority=10),
        NewsEventSource("wallstreetcn", wallstreetcn_loader, priority=20),
        NewsEventSource("yahoo_news", yahoo_loader, priority=30),
        NewsEventSource("rss_news", rss_loader, priority=40),
    )
    registry.register(
        SourceSpec(
            source_id="multi_source_news_events",
            domain=DataDomain.NEWS,
            adapter=multi_source_news_event_adapter(sources),
            priority=0,
            timeout_seconds=30.0,
            ttl_seconds=300.0,
            max_stale_seconds=3_600.0,
            required_fields=(
                "items",
                "source_statuses",
                "conflicts",
                "window",
                "directive_capability",
                "execution_eligible",
            ),
        )
    )
