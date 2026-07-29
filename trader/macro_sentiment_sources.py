"""FRED macro facts and research-only social/expectation Data Hub adapters."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping

from .data_hub import (
    AdapterResult,
    DataDomain,
    DataRequest,
    SourceRegistry,
    SourceSpec,
)

SourceLoader = Callable[
    [DataRequest],
    Iterable[Any] | Mapping[str, Any] | None,
]


@dataclass(frozen=True)
class FredSeriesSpec:
    series_id: str
    max_age_seconds: float
    required: bool = True

    def __post_init__(self) -> None:
        if not self.series_id.strip():
            raise ValueError("DATA_FRED_SERIES_ID_REQUIRED")
        if (
            not math.isfinite(self.max_age_seconds)
            or self.max_age_seconds <= 0
        ):
            raise ValueError("DATA_FRED_MAX_AGE_INVALID")


DEFAULT_FRED_SERIES = (
    FredSeriesSpec("DGS10", 7 * 86_400),
    FredSeriesSpec("FEDFUNDS", 45 * 86_400),
    FredSeriesSpec("CPIAUCSL", 45 * 86_400),
    FredSeriesSpec("UNRATE", 45 * 86_400),
    FredSeriesSpec("WALCL", 14 * 86_400),
    FredSeriesSpec("M2SL", 45 * 86_400),
)


@dataclass(frozen=True)
class ResearchSignalSource:
    source_id: str
    dimension: str
    loader: SourceLoader
    max_age_seconds: float
    min_items: int = 1
    quality_floor: float = 0.5
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("DATA_SIGNAL_SOURCE_ID_REQUIRED")
        if self.dimension not in {"SOCIAL_SENTIMENT", "MARKET_EXPECTATION"}:
            raise ValueError("DATA_SIGNAL_DIMENSION_INVALID")
        if (
            not math.isfinite(self.max_age_seconds)
            or self.max_age_seconds <= 0
        ):
            raise ValueError("DATA_SIGNAL_MAX_AGE_INVALID")
        if self.min_items <= 0:
            raise ValueError("DATA_SIGNAL_MIN_ITEMS_INVALID")
        if (
            not math.isfinite(self.quality_floor)
            or not 0 <= self.quality_floor <= 1
        ):
            raise ValueError("DATA_SIGNAL_QUALITY_FLOOR_INVALID")
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("DATA_SIGNAL_WEIGHT_INVALID")


def _value(item: Any, *names: str) -> Any:
    if isinstance(item, Mapping):
        for name in names:
            if name in item:
                return item[name]
    for name in names:
        if hasattr(item, name):
            return getattr(item, name)
    return None


def _nested(item: Any, *path: str) -> Any:
    value = item
    for name in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(name)
    return value


def _source_datetime(value: Any) -> datetime | None:
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
                return None
            if (
                parsed.tzinfo is None
                and len(text) == 10
                and text[4] == "-"
                and text[7] == "-"
            ):
                parsed = parsed.replace(tzinfo=timezone.utc)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


def _finite_float(value: Any) -> float | None:
    if value in (None, "", "."):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _coerce_items(raw: Any) -> list[Any]:
    if raw is None:
        return []
    if isinstance(raw, Mapping):
        for field_name in ("items", "messages", "posts", "markets"):
            nested = raw.get(field_name)
            if isinstance(nested, Iterable) and not isinstance(
                nested,
                (str, bytes, Mapping),
            ):
                return list(nested)
        return [raw]
    if isinstance(raw, (str, bytes)):
        raise ValueError("DATA_SIGNAL_SOURCE_PAYLOAD_INVALID")
    try:
        return list(raw)
    except TypeError as exc:
        raise ValueError("DATA_SIGNAL_SOURCE_PAYLOAD_INVALID") from exc


def _fred_series_payload(
    raw: Any,
    request: DataRequest,
) -> dict[str, list[Any]]:
    if not isinstance(raw, Mapping):
        raise ValueError("DATA_FRED_PAYLOAD_INVALID")
    observations = raw.get("observations")
    if isinstance(observations, list):
        series_id = str(
            raw.get("series_id")
            or request.params.get("series_id")
            or ""
        ).strip()
        if not series_id:
            raise ValueError("DATA_FRED_SERIES_ID_REQUIRED")
        return {series_id: observations}
    result = {}
    for series_id, entries in raw.items():
        if isinstance(entries, Iterable) and not isinstance(
            entries,
            (str, bytes, Mapping),
        ):
            result[str(series_id)] = list(entries)
    return result


def fred_macro_adapter(
    loader: SourceLoader,
    *,
    series_specs: Iterable[FredSeriesSpec] = DEFAULT_FRED_SERIES,
):
    specs = tuple(series_specs)
    if not specs:
        raise ValueError("DATA_FRED_SERIES_REQUIRED")

    def fetch(request: DataRequest) -> AdapterResult:
        raw_by_series = _fred_series_payload(loader(request), request)
        observations: list[dict[str, Any]] = []
        missing_series = []
        stale_series = []
        required_specs = [spec for spec in specs if spec.required]

        for spec in specs:
            normalized = []
            for item in raw_by_series.get(spec.series_id, []):
                observed_at = _source_datetime(
                    _value(item, "observed_at", "date")
                )
                as_of = _source_datetime(
                    _value(
                        item,
                        "as_of",
                        "realtime_start",
                        "published_at",
                    )
                ) or observed_at
                value = _finite_float(_value(item, "value"))
                if (
                    observed_at is None
                    or as_of is None
                    or as_of > request.requested_at
                    or value is None
                ):
                    continue
                normalized.append((observed_at, as_of, value))
            if not normalized:
                if spec.required:
                    missing_series.append(spec.series_id)
                continue

            observed_at, as_of, value = max(
                normalized,
                key=lambda row: (row[0], row[1]),
            )
            age_seconds = max(
                0.0,
                (request.requested_at - as_of).total_seconds(),
            )
            is_stale = age_seconds > spec.max_age_seconds
            if is_stale:
                stale_series.append(spec.series_id)
            observations.append(
                {
                    "series_id": spec.series_id,
                    "value": value,
                    "observed_at": observed_at.isoformat(),
                    "as_of": as_of.isoformat(),
                    "age_seconds": age_seconds,
                    "freshness": "STALE" if is_stale else "FRESH",
                    "quality_score": 0.4 if is_stale else 1.0,
                    "source": "fred",
                }
            )

        if not observations:
            raise ValueError("DATA_FRED_OBSERVATIONS_EMPTY")
        required_count = len(required_specs)
        present_required = [
            item
            for item in observations
            if any(
                spec.series_id == item["series_id"] and spec.required
                for spec in specs
            )
        ]
        coverage = (
            len(present_required) / required_count
            if required_count
            else 1.0
        )
        average_quality = (
            sum(item["quality_score"] for item in observations)
            / len(observations)
        )
        quality_score = coverage * average_quality
        as_of = max(
            _source_datetime(item["as_of"])
            for item in observations
            if _source_datetime(item["as_of"]) is not None
        )
        observations.sort(key=lambda item: item["series_id"])
        return AdapterResult(
            payload={
                "observations": observations,
                "coverage": coverage,
                "missing_series": missing_series,
                "stale_series": stale_series,
                "fact_role": "RESEARCH_MACRO",
                "broker_fact_eligible": False,
                "execution_eligible": False,
            },
            as_of=as_of,
            quality_score=quality_score,
            metadata={
                "upstream": "fred",
                "required_series": [
                    spec.series_id for spec in required_specs
                ],
                "low_quality": quality_score < 0.7,
                "execution_eligible": False,
            },
        )

    return fetch


def _stocktwits_signal(
    item: Any,
    source: ResearchSignalSource,
    request: DataRequest,
) -> dict[str, Any]:
    observed_at = _source_datetime(
        _value(item, "observed_at", "created_at", "timestamp")
    )
    if observed_at is None:
        raise ValueError("DATA_SIGNAL_TIME_REQUIRED")
    if observed_at > request.requested_at:
        raise ValueError("DATA_SIGNAL_TIME_FUTURE")
    label = str(
        _value(item, "sentiment")
        or _nested(item, "entities", "sentiment", "basic")
        or ""
    ).strip().upper()
    score = _finite_float(_value(item, "sentiment_score"))
    if score is None and label in {"BULLISH", "BEARISH"}:
        score = 1.0 if label == "BULLISH" else -1.0
    quality = 0.85 if score is not None else 0.45
    return _base_signal(
        item,
        source,
        request,
        observed_at=observed_at,
        quality_score=quality,
        fields={
            "sentiment_label": label or "UNLABELED",
            "sentiment_score": score,
            "engagement": _finite_float(
                _value(item, "likes", "like_count")
                or _nested(item, "likes", "total")
            ),
        },
    )


def _reddit_signal(
    item: Any,
    source: ResearchSignalSource,
    request: DataRequest,
) -> dict[str, Any]:
    observed_at = _source_datetime(
        _value(item, "observed_at", "created_utc", "created_at")
    )
    if observed_at is None:
        raise ValueError("DATA_SIGNAL_TIME_REQUIRED")
    if observed_at > request.requested_at:
        raise ValueError("DATA_SIGNAL_TIME_FUTURE")
    score = _finite_float(_value(item, "sentiment_score"))
    quality = 0.65 if score is not None else 0.3
    return _base_signal(
        item,
        source,
        request,
        observed_at=observed_at,
        quality_score=quality,
        fields={
            "sentiment_label": str(
                _value(item, "sentiment") or "UNLABELED"
            ).upper(),
            "sentiment_score": score,
            "engagement": _finite_float(_value(item, "score")),
            "comment_count": _finite_float(
                _value(item, "num_comments", "comment_count")
            ),
            "community": str(_value(item, "subreddit") or ""),
        },
    )


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _polymarket_probability(item: Any) -> float | None:
    direct = _finite_float(
        _value(item, "probability", "yes_price", "lastTradePrice")
    )
    if direct is not None:
        return direct if 0 <= direct <= 1 else None
    outcomes = [
        str(value).strip().upper()
        for value in _json_list(_value(item, "outcomes"))
    ]
    prices = [
        _finite_float(value)
        for value in _json_list(_value(item, "outcomePrices"))
    ]
    if "YES" in outcomes:
        index = outcomes.index("YES")
        if index < len(prices):
            return prices[index]
    return prices[0] if prices and prices[0] is not None else None


def _polymarket_signal(
    item: Any,
    source: ResearchSignalSource,
    request: DataRequest,
) -> dict[str, Any]:
    observed_at = _source_datetime(
        _value(
            item,
            "observed_at",
            "updatedAt",
            "updated_at",
            "createdAt",
        )
    )
    if observed_at is None:
        raise ValueError("DATA_SIGNAL_TIME_REQUIRED")
    if observed_at > request.requested_at:
        raise ValueError("DATA_SIGNAL_TIME_FUTURE")
    probability = _polymarket_probability(item)
    liquidity = _finite_float(_value(item, "liquidity", "liquidityNum"))
    volume = _finite_float(_value(item, "volume", "volumeNum"))
    if probability is None:
        quality = 0.25
    elif (liquidity or 0) > 0 or (volume or 0) > 0:
        quality = 0.85
    else:
        quality = 0.55
    return _base_signal(
        item,
        source,
        request,
        observed_at=observed_at,
        quality_score=quality,
        fields={
            "probability": probability,
            "liquidity": liquidity,
            "volume": volume,
            "market_end_at": (
                _source_datetime(_value(item, "endDate", "end_date")).isoformat()
                if _source_datetime(_value(item, "endDate", "end_date"))
                else None
            ),
        },
    )


def _base_signal(
    item: Any,
    source: ResearchSignalSource,
    request: DataRequest,
    *,
    observed_at: datetime,
    quality_score: float,
    fields: Mapping[str, Any],
) -> dict[str, Any]:
    age_seconds = max(
        0.0,
        (request.requested_at - observed_at).total_seconds(),
    )
    is_stale = age_seconds > source.max_age_seconds
    effective_quality = quality_score * (0.5 if is_stale else 1.0)
    return {
        "signal_id": str(
            _value(item, "signal_id", "id", "post_id", "market_id")
            or ""
        ),
        "dimension": source.dimension,
        "symbol": str(
            _value(item, "symbol", "ticker") or request.key
        ).upper(),
        "title": str(
            _value(item, "title", "body", "question") or ""
        ).strip(),
        "observed_at": observed_at.isoformat(),
        "as_of": observed_at.isoformat(),
        "age_seconds": age_seconds,
        "freshness": "STALE" if is_stale else "FRESH",
        "quality_score": effective_quality,
        "quality_label": (
            "LOW"
            if effective_quality < source.quality_floor
            else "ACCEPTABLE"
        ),
        "source": source.source_id,
        "broker_fact_eligible": False,
        "execution_eligible": False,
        **dict(fields),
    }


def sentiment_expectation_adapter(
    sources: Iterable[ResearchSignalSource],
):
    source_specs = tuple(sources)
    if not source_specs:
        raise ValueError("DATA_SIGNAL_SOURCES_REQUIRED")

    def fetch(request: DataRequest) -> AdapterResult:
        signals: list[dict[str, Any]] = []
        statuses: list[dict[str, Any]] = []
        total_weight = sum(source.weight for source in source_specs)
        weighted_quality = 0.0

        for source in source_specs:
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
                        "coverage": 0.0,
                        "item_count": 0,
                        "fresh_count": 0,
                        "low_quality_count": 0,
                        "failure_code": code,
                    }
                )
                continue

            normalized = []
            dropped_codes = []
            for item in raw_items:
                try:
                    if source.source_id == "stocktwits":
                        signal = _stocktwits_signal(item, source, request)
                    elif source.source_id == "reddit":
                        signal = _reddit_signal(item, source, request)
                    elif source.source_id == "polymarket":
                        signal = _polymarket_signal(item, source, request)
                    else:
                        raise ValueError("DATA_SIGNAL_SOURCE_UNSUPPORTED")
                except ValueError as exc:
                    dropped_codes.append(str(exc))
                    continue
                normalized.append(signal)
            signals.extend(normalized)

            coverage = min(len(normalized) / source.min_items, 1.0)
            fresh_count = sum(
                item["freshness"] == "FRESH" for item in normalized
            )
            low_quality_count = sum(
                item["quality_label"] == "LOW" for item in normalized
            )
            average_quality = (
                sum(item["quality_score"] for item in normalized)
                / len(normalized)
                if normalized
                else 0.0
            )
            freshness_ratio = (
                fresh_count / len(normalized) if normalized else 0.0
            )
            source_quality = (
                coverage * average_quality * freshness_ratio
            )
            weighted_quality += source.weight * source_quality

            if not normalized or coverage < 1.0:
                status = "LOW_COVERAGE"
            elif fresh_count == 0:
                status = "STALE"
            elif average_quality < source.quality_floor:
                status = "LOW_QUALITY"
            elif dropped_codes or fresh_count < len(normalized):
                status = "DEGRADED"
            else:
                status = "OK"
            statuses.append(
                {
                    "source_id": source.source_id,
                    "status": status,
                    "coverage": coverage,
                    "item_count": len(normalized),
                    "fresh_count": fresh_count,
                    "low_quality_count": low_quality_count,
                    "failure_code": (
                        "DATA_SIGNAL_ITEMS_DROPPED"
                        if dropped_codes
                        else ""
                    ),
                    "dropped_codes": sorted(set(dropped_codes)),
                }
            )

        if not any(status["status"] != "FAILED" for status in statuses):
            raise ValueError("DATA_SENTIMENT_ALL_SOURCES_FAILED")

        quality_score = weighted_quality / total_weight
        signals.sort(
            key=lambda item: (
                item["source"],
                item["observed_at"],
                item["signal_id"],
            )
        )
        low_quality_sources = [
            status["source_id"]
            for status in statuses
            if status["status"] not in {"OK"}
        ]
        signal_times = [
            _source_datetime(item["as_of"])
            for item in signals
            if _source_datetime(item["as_of"]) is not None
        ]
        return AdapterResult(
            payload={
                "signals": signals,
                "source_statuses": statuses,
                "coverage": {
                    status["source_id"]: status["coverage"]
                    for status in statuses
                },
                "low_quality_sources": low_quality_sources,
                "directive_capability": "RESEARCH_ONLY",
                "broker_fact_eligible": False,
                "execution_eligible": False,
            },
            as_of=max(signal_times, default=request.requested_at),
            quality_score=quality_score,
            metadata={
                "upstreams": [
                    source.source_id for source in source_specs
                ],
                "low_authority": True,
                "execution_eligible": False,
            },
        )

    return fetch


def register_macro_sentiment_sources(
    registry: SourceRegistry,
    *,
    fred_loader: SourceLoader,
    stocktwits_loader: SourceLoader,
    reddit_loader: SourceLoader,
    polymarket_loader: SourceLoader,
    fred_series_specs: Iterable[FredSeriesSpec] = DEFAULT_FRED_SERIES,
) -> None:
    registry.register(
        SourceSpec(
            source_id="fred_macro",
            domain=DataDomain.MACRO,
            adapter=fred_macro_adapter(
                fred_loader,
                series_specs=fred_series_specs,
            ),
            priority=0,
            timeout_seconds=20.0,
            ttl_seconds=3_600.0,
            max_stale_seconds=86_400.0,
            required_fields=(
                "observations",
                "coverage",
                "missing_series",
                "stale_series",
                "fact_role",
                "broker_fact_eligible",
                "execution_eligible",
            ),
        )
    )
    registry.register(
        SourceSpec(
            source_id="social_market_expectations",
            domain=DataDomain.SENTIMENT,
            adapter=sentiment_expectation_adapter(
                (
                    ResearchSignalSource(
                        "stocktwits",
                        "SOCIAL_SENTIMENT",
                        stocktwits_loader,
                        max_age_seconds=6 * 3_600,
                        quality_floor=0.5,
                    ),
                    ResearchSignalSource(
                        "reddit",
                        "SOCIAL_SENTIMENT",
                        reddit_loader,
                        max_age_seconds=12 * 3_600,
                        quality_floor=0.4,
                    ),
                    ResearchSignalSource(
                        "polymarket",
                        "MARKET_EXPECTATION",
                        polymarket_loader,
                        max_age_seconds=24 * 3_600,
                        quality_floor=0.5,
                    ),
                )
            ),
            priority=0,
            timeout_seconds=30.0,
            ttl_seconds=600.0,
            max_stale_seconds=3_600.0,
            required_fields=(
                "signals",
                "source_statuses",
                "coverage",
                "low_quality_sources",
                "directive_capability",
                "broker_fact_eligible",
                "execution_eligible",
            ),
            quality_cap=0.85,
        )
    )
