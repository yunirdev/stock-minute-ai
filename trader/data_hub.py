"""Unified, production-neutral data source registry and quality contract."""
from __future__ import annotations

import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from threading import Lock
from typing import Any, Callable, Mapping


class DataDomain(StrEnum):
    MARKET = "MARKET"
    BROKER = "BROKER"
    CORPORATE = "CORPORATE"
    NEWS = "NEWS"
    MACRO = "MACRO"
    SENTIMENT = "SENTIMENT"
    INTERNAL = "INTERNAL"


class DataStatus(StrEnum):
    OK = "OK"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class DataRequest:
    domain: DataDomain
    key: str
    requested_at: datetime
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AdapterResult:
    payload: Mapping[str, Any]
    as_of: datetime
    quality_score: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


DataAdapter = Callable[[DataRequest], AdapterResult]


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    domain: DataDomain
    adapter: DataAdapter
    priority: int
    timeout_seconds: float
    ttl_seconds: float
    max_stale_seconds: float = 0.0
    required_fields: tuple[str, ...] = ()
    quality_cap: float = 1.0

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("DATA_SOURCE_ID_REQUIRED")
        if self.priority < 0:
            raise ValueError("DATA_SOURCE_PRIORITY_INVALID")
        for field_name in (
            "timeout_seconds",
            "ttl_seconds",
            "max_stale_seconds",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"DATA_SOURCE_{field_name.upper()}_INVALID"
                )
        if self.timeout_seconds <= 0:
            raise ValueError("DATA_SOURCE_TIMEOUT_SECONDS_INVALID")
        if not math.isfinite(self.quality_cap) or not 0 <= self.quality_cap <= 1:
            raise ValueError("DATA_SOURCE_QUALITY_CAP_INVALID")


@dataclass(frozen=True)
class SourceFailure:
    source_id: str
    code: str


@dataclass(frozen=True)
class DataEnvelope:
    request_id: str
    domain: DataDomain
    key: str
    source_id: str
    status: DataStatus
    payload: Mapping[str, Any]
    as_of: datetime
    fetched_at: datetime
    expires_at: datetime
    quality_score: float
    cache_hit: bool = False
    fallback_from: str = ""
    failure_code: str = ""
    failures: tuple[SourceFailure, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


class SourceRegistry:
    def __init__(self) -> None:
        self._sources: dict[str, SourceSpec] = {}

    def register(self, spec: SourceSpec) -> None:
        if spec.source_id in self._sources:
            raise ValueError("DATA_SOURCE_DUPLICATE")
        self._sources[spec.source_id] = spec

    def sources_for(self, domain: DataDomain) -> tuple[SourceSpec, ...]:
        return tuple(
            sorted(
                (
                    source
                    for source in self._sources.values()
                    if source.domain == domain
                ),
                key=lambda source: (source.priority, source.source_id),
            )
        )

    def get(self, source_id: str) -> SourceSpec:
        try:
            return self._sources[source_id]
        except KeyError as exc:
            raise KeyError(f"DATA_SOURCE_UNKNOWN:{source_id}") from exc


class DataHub:
    """Resolve one domain request through cache, quality, and fallback rules."""

    def __init__(
        self,
        registry: SourceRegistry,
        *,
        clock: Callable[[], datetime] | None = None,
        max_workers: int = 4,
    ) -> None:
        self.registry = registry
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="data-hub",
        )
        self._cache: dict[tuple[str, str], DataEnvelope] = {}
        self._cache_lock = Lock()

    @staticmethod
    def _request_id(
        domain: DataDomain,
        key: str,
        params: Mapping[str, Any],
    ) -> str:
        raw = json.dumps(
            {
                "domain": domain.value,
                "key": key,
                "params": dict(params),
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return (
            "data-request-"
            + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]
        )

    @staticmethod
    def _aware(value: datetime, code: str) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(code)
        return value.astimezone(timezone.utc)

    def fetch(
        self,
        domain: DataDomain,
        key: str,
        *,
        params: Mapping[str, Any] | None = None,
    ) -> DataEnvelope:
        now = self._aware(self._clock(), "DATA_HUB_CLOCK_TIMEZONE_REQUIRED")
        normalized_key = key.strip().upper()
        if not normalized_key:
            raise ValueError("DATA_REQUEST_KEY_REQUIRED")
        request_params = dict(params or {})
        request_id = self._request_id(
            domain,
            normalized_key,
            request_params,
        )
        request = DataRequest(
            domain=domain,
            key=normalized_key,
            requested_at=now,
            params=request_params,
        )
        sources = self.registry.sources_for(domain)
        if not sources:
            return self._failed_envelope(
                request,
                request_id,
                now,
                (SourceFailure("", "NO_REGISTERED_SOURCE"),),
            )
        primary_id = sources[0].source_id
        failures: list[SourceFailure] = []
        stale_candidates: list[tuple[SourceSpec, DataEnvelope]] = []

        for source in sources:
            cache_key = (source.source_id, request_id)
            with self._cache_lock:
                cached = self._cache.get(cache_key)
            if cached is not None and now <= cached.expires_at:
                return replace(cached, cache_hit=True)
            if (
                cached is not None
                and source.max_stale_seconds > 0
                and now
                <= cached.expires_at
                + timedelta(seconds=source.max_stale_seconds)
            ):
                stale_candidates.append((source, cached))
            future = self._executor.submit(source.adapter, request)
            try:
                result = future.result(timeout=source.timeout_seconds)
                envelope = self._validate_result(
                    request,
                    request_id,
                    source,
                    result,
                    now,
                    primary_id,
                    tuple(failures),
                )
            except FutureTimeout:
                future.cancel()
                failures.append(
                    SourceFailure(source.source_id, "SOURCE_TIMEOUT")
                )
                continue
            except Exception as exc:
                code = (
                    str(exc)
                    if isinstance(exc, ValueError)
                    and str(exc).startswith("DATA_")
                    else type(exc).__name__
                )
                failures.append(SourceFailure(source.source_id, code))
                continue
            with self._cache_lock:
                self._cache[cache_key] = envelope
            return envelope

        if stale_candidates:
            source, cached = stale_candidates[0]
            return replace(
                cached,
                status=DataStatus.DEGRADED,
                quality_score=min(cached.quality_score, 0.5),
                cache_hit=True,
                fallback_from=primary_id,
                failure_code="STALE_CACHE_FALLBACK",
                failures=tuple(failures),
                metadata={
                    **dict(cached.metadata),
                    "stale_seconds": max(
                        0.0,
                        (now - cached.expires_at).total_seconds(),
                    ),
                    "stale_source": source.source_id,
                },
            )
        return self._failed_envelope(
            request,
            request_id,
            now,
            tuple(failures),
        )

    def _validate_result(
        self,
        request: DataRequest,
        request_id: str,
        source: SourceSpec,
        result: AdapterResult,
        now: datetime,
        primary_id: str,
        failures: tuple[SourceFailure, ...],
    ) -> DataEnvelope:
        if not isinstance(result, AdapterResult):
            raise ValueError("DATA_ADAPTER_RESULT_INVALID")
        as_of = self._aware(
            result.as_of,
            "DATA_SOURCE_AS_OF_TIMEZONE_REQUIRED",
        )
        if as_of > now:
            raise ValueError("DATA_SOURCE_AS_OF_FUTURE")
        if not isinstance(result.payload, Mapping):
            raise ValueError("DATA_SOURCE_PAYLOAD_INVALID")
        missing = [
            field_name
            for field_name in source.required_fields
            if field_name not in result.payload
        ]
        if missing:
            raise ValueError("DATA_SOURCE_REQUIRED_FIELDS_MISSING")
        quality_score = float(result.quality_score)
        if (
            not math.isfinite(quality_score)
            or not 0 <= quality_score <= 1
        ):
            raise ValueError("DATA_SOURCE_QUALITY_INVALID")
        effective_quality = min(quality_score, source.quality_cap)
        is_fallback = source.source_id != primary_id
        is_quality_degraded = effective_quality < 1.0
        status = (
            DataStatus.DEGRADED
            if is_fallback or is_quality_degraded
            else DataStatus.OK
        )
        if is_fallback:
            failure_code = "FALLBACK_SOURCE"
        elif is_quality_degraded:
            failure_code = "SOURCE_QUALITY_DEGRADED"
        else:
            failure_code = ""
        return DataEnvelope(
            request_id=request_id,
            domain=request.domain,
            key=request.key,
            source_id=source.source_id,
            status=status,
            payload=dict(result.payload),
            as_of=as_of,
            fetched_at=now,
            expires_at=now + timedelta(seconds=source.ttl_seconds),
            quality_score=effective_quality,
            fallback_from=primary_id if is_fallback else "",
            failure_code=failure_code,
            failures=failures,
            metadata=dict(result.metadata),
        )

    @staticmethod
    def _failed_envelope(
        request: DataRequest,
        request_id: str,
        now: datetime,
        failures: tuple[SourceFailure, ...],
    ) -> DataEnvelope:
        return DataEnvelope(
            request_id=request_id,
            domain=request.domain,
            key=request.key,
            source_id="",
            status=DataStatus.FAILED,
            payload={},
            as_of=now,
            fetched_at=now,
            expires_at=now,
            quality_score=0.0,
            failure_code="ALL_SOURCES_FAILED",
            failures=failures,
        )

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
