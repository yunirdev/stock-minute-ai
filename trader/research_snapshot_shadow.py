"""Build shadow ResearchSnapshots from the exact existing screening inputs."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from .data_cache import describe_cached_bars
from .models import (
    ResearchQuality,
    ResearchSnapshot,
    ResearchSourceManifestEntry,
    ResearchSourceStatus,
)
from .research_snapshot import CURRENT_SNAPSHOT_SCHEMA_VERSION
from .strategy_statistics import describe_strategy_statistics


def _candidate_as_of(candidate: Any, captured_at: datetime) -> datetime:
    raw = str(getattr(candidate, "as_of", "") or "").strip()
    if not raw:
        return captured_at
    try:
        value = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("CANDIDATE_AS_OF_INVALID") from exc
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("CANDIDATE_AS_OF_TIMEZONE_REQUIRED")
    return value.astimezone(timezone.utc)


def _entry(descriptor: dict[str, Any]) -> ResearchSourceManifestEntry:
    return ResearchSourceManifestEntry(
        source=str(descriptor["source"]),
        status=ResearchSourceStatus(str(descriptor["status"])),
        as_of=descriptor["as_of"],
        fetched_at=descriptor["fetched_at"],
        quality_score=float(descriptor["quality_score"]),
        coverage=tuple(descriptor["coverage"]),
        payload_version=str(descriptor["payload_version"]),
        failure_code=str(descriptor.get("failure_code", "")),
        metadata=dict(descriptor.get("metadata") or {}),
    )


def build_screening_shadow_snapshot(
    *,
    run_id: str,
    trading_date: str,
    timeframe: str,
    candidate: Any,
    bars: pd.DataFrame | None,
    strategy_statistics: tuple | list,
    strategy_statistics_path: str,
    captured_at: datetime,
) -> ResearchSnapshot:
    """Capture inputs after screening without changing any production reader."""
    captured_at = captured_at.astimezone(timezone.utc)
    symbol = str(candidate.symbol).strip().upper()
    bar_descriptor = describe_cached_bars(
        symbol,
        timeframe,
        frame=bars if bars is not None else pd.DataFrame(),
        captured_at=captured_at,
    )
    statistics_descriptor = describe_strategy_statistics(
        tuple(strategy_statistics),
        symbol=symbol,
        timeframe=timeframe,
        source_path=strategy_statistics_path,
        captured_at=captured_at,
    )
    confidence_score = {
        "高": 1.0,
        "中": 0.7,
        "低": 0.3,
    }.get(str(candidate.data_confidence), 0.0)
    screening_descriptor = {
        "source": "deterministic_screening",
        "status": "OK" if confidence_score >= 1.0 else "DEGRADED",
        "as_of": _candidate_as_of(candidate, captured_at),
        "fetched_at": captured_at,
        "quality_score": confidence_score,
        "coverage": ("candidate_fields",),
        "payload_version": "daily-candidate:v1",
        "failure_code": (
            "" if confidence_score >= 1.0 else "CANDIDATE_CONFIDENCE_REDUCED"
        ),
        "metadata": {
            "timeframe": timeframe,
            "rank": int(candidate.rank),
        },
    }
    manifest = tuple(
        _entry(descriptor)
        for descriptor in (
            bar_descriptor,
            statistics_descriptor,
            screening_descriptor,
        )
    )
    statuses = {entry.status for entry in manifest}
    if statuses & {
        ResearchSourceStatus.FAILED,
        ResearchSourceStatus.MISSING,
    }:
        quality = ResearchQuality.PARTIAL
    elif ResearchSourceStatus.DEGRADED in statuses:
        quality = ResearchQuality.DEGRADED
    else:
        quality = ResearchQuality.GOOD
    quality_score = sum(
        entry.quality_score for entry in manifest
    ) / len(manifest)
    candidate_payload = asdict(candidate)
    payload = {
        "candidate": candidate_payload,
        "bars": bar_descriptor["payload"],
        "strategy_statistics": statistics_descriptor["payload"],
        "context": {
            "timeframe": timeframe,
            "strategy_statistics_path": strategy_statistics_path,
        },
    }
    fingerprint = json.dumps(
        {
            "run_id": run_id,
            "symbol": symbol,
            "payload": payload,
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    snapshot_id = (
        "snapshot-"
        + hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:24]
    )
    source_as_of = max(entry.as_of for entry in manifest)
    return ResearchSnapshot(
        snapshot_id=snapshot_id,
        run_id=run_id,
        symbol=symbol,
        trading_date=trading_date,
        as_of=source_as_of,
        data_cutoff=source_as_of,
        captured_at=captured_at,
        source_manifest=manifest,
        quality=quality,
        quality_score=quality_score,
        payload_version="daily-screening-input:v1",
        payload=payload,
        schema_version=CURRENT_SNAPSHOT_SCHEMA_VERSION,
        created_at=captured_at,
    )


def compare_candidate_to_snapshot(
    candidate: Any,
    snapshot: ResearchSnapshot,
) -> list[dict[str, Any]]:
    """Compare normalized candidate fields to the frozen shadow payload."""
    actual = json.loads(
        json.dumps(
            asdict(candidate),
            sort_keys=True,
            default=str,
        )
    )
    captured = dict(snapshot.payload.get("candidate") or {})
    differences = []
    for field_name in sorted(set(actual) | set(captured)):
        if actual.get(field_name) == captured.get(field_name):
            continue
        differences.append(
            {
                "field": field_name,
                "actual": actual.get(field_name),
                "captured": captured.get(field_name),
                "classification": "UNCLASSIFIED",
            }
        )
    return differences
