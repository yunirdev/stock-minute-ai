"""ResearchSnapshot serialization and backward-compatible DuckDB storage."""
from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

from .models import (
    ResearchQuality,
    ResearchSnapshot,
    ResearchSourceManifestEntry,
    ResearchSourceStatus,
)

CURRENT_SNAPSHOT_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class SnapshotSaveResult:
    snapshot_id: str
    created: bool
    deduplicated: bool


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _datetime(value: Any, field_name: str) -> datetime:
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, str):
        try:
            result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(
                f"{field_name.upper()}_INVALID"
            ) from exc
    else:
        raise ValueError(f"{field_name.upper()}_INVALID")
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError(f"{field_name.upper()}_TIMEZONE_REQUIRED")
    return result


def source_manifest_to_dict(
    entry: ResearchSourceManifestEntry,
) -> dict[str, Any]:
    return {
        "source": entry.source,
        "status": entry.status.value,
        "as_of": _iso(entry.as_of),
        "fetched_at": _iso(entry.fetched_at),
        "quality_score": entry.quality_score,
        "coverage": list(entry.coverage),
        "payload_version": entry.payload_version,
        "failure_code": entry.failure_code,
        "metadata": entry.metadata,
    }


def source_manifest_from_dict(
    value: dict[str, Any],
) -> ResearchSourceManifestEntry:
    try:
        status = ResearchSourceStatus(str(value["status"]))
    except (KeyError, ValueError) as exc:
        raise ValueError("SOURCE_STATUS_INVALID") from exc
    return ResearchSourceManifestEntry(
        source=str(value.get("source", "")),
        status=status,
        as_of=_datetime(value.get("as_of"), "source_as_of"),
        fetched_at=_datetime(
            value.get("fetched_at"),
            "source_fetched_at",
        ),
        quality_score=float(value.get("quality_score", -1)),
        coverage=tuple(str(item) for item in value.get("coverage", [])),
        payload_version=str(value.get("payload_version", "")),
        failure_code=str(value.get("failure_code", "")),
        metadata=dict(value.get("metadata") or {}),
    )


def snapshot_to_dict(snapshot: ResearchSnapshot) -> dict[str, Any]:
    return {
        "snapshot_id": snapshot.snapshot_id,
        "symbol": snapshot.symbol,
        "trading_date": snapshot.trading_date,
        "as_of": _iso(snapshot.as_of),
        "data_cutoff": _iso(snapshot.data_cutoff),
        "captured_at": _iso(snapshot.captured_at),
        "source_manifest": [
            source_manifest_to_dict(entry)
            for entry in snapshot.source_manifest
        ],
        "quality": snapshot.quality.value,
        "quality_score": snapshot.quality_score,
        "payload_version": snapshot.payload_version,
        "payload": snapshot.payload,
        "run_id": snapshot.run_id,
        "schema_version": snapshot.schema_version,
        "created_at": _iso(snapshot.created_at),
    }


def snapshot_from_dict(value: dict[str, Any]) -> ResearchSnapshot:
    try:
        quality = ResearchQuality(str(value["quality"]))
    except (KeyError, ValueError) as exc:
        raise ValueError("SNAPSHOT_QUALITY_INVALID") from exc
    manifest = tuple(
        source_manifest_from_dict(dict(entry))
        for entry in value.get("source_manifest", [])
    )
    return ResearchSnapshot(
        snapshot_id=str(value.get("snapshot_id", "")),
        symbol=str(value.get("symbol", "")),
        trading_date=str(value.get("trading_date", "")),
        as_of=_datetime(value.get("as_of"), "snapshot_as_of"),
        data_cutoff=_datetime(
            value.get("data_cutoff"),
            "snapshot_data_cutoff",
        ),
        captured_at=_datetime(
            value.get("captured_at"),
            "snapshot_captured_at",
        ),
        source_manifest=manifest,
        quality=quality,
        quality_score=float(value.get("quality_score", -1)),
        payload_version=str(value.get("payload_version", "")),
        payload=dict(value.get("payload") or {}),
        run_id=str(value.get("run_id", "")),
        schema_version=int(value.get("schema_version", 1)),
        created_at=_datetime(
            value.get("created_at"),
            "snapshot_created_at",
        ),
    )


def snapshot_content_hash(snapshot: ResearchSnapshot) -> str:
    """Hash immutable content while excluding identity/storage timestamps."""
    value = snapshot_to_dict(snapshot)
    value.pop("snapshot_id", None)
    value.pop("created_at", None)
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _decode_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        decoded = json.loads(str(value or "{}"))
    except json.JSONDecodeError:
        return {"legacy_raw": str(value)}
    return decoded if isinstance(decoded, dict) else {"legacy_value": decoded}


def _decode_manifest(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, dict)]
    try:
        decoded = json.loads(str(value or "[]"))
    except json.JSONDecodeError:
        return []
    return [dict(item) for item in decoded if isinstance(item, dict)]


class ResearchSnapshotStore:
    """Append-only snapshot store; current readers also accept legacy rows."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._migrate()

    def _connect(self, *, read_only: bool = False):
        return duckdb.connect(self.db_path, read_only=read_only)

    def _migrate(self) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS research_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    symbol TEXT,
                    trading_date TEXT,
                    as_of TIMESTAMPTZ,
                    data_cutoff TIMESTAMPTZ,
                    captured_at TIMESTAMPTZ,
                    source_manifest TEXT,
                    quality TEXT,
                    quality_score DOUBLE,
                    payload_version TEXT,
                    payload TEXT,
                    run_id TEXT,
                    schema_version INTEGER,
                    created_at TIMESTAMPTZ,
                    content_hash TEXT,
                    frozen_at TIMESTAMPTZ
                )
                """
            )
            columns = {
                row[0]
                for row in conn.execute(
                    "DESCRIBE research_snapshots"
                ).fetchall()
            }
            additions = {
                "symbol": "TEXT",
                "trading_date": "TEXT",
                "as_of": "TIMESTAMPTZ",
                "data_cutoff": "TIMESTAMPTZ",
                "captured_at": "TIMESTAMPTZ",
                "source_manifest": "TEXT",
                "quality": "TEXT",
                "quality_score": "DOUBLE",
                "payload_version": "TEXT",
                "payload": "TEXT",
                "run_id": "TEXT",
                "schema_version": "INTEGER",
                "created_at": "TIMESTAMPTZ",
                "content_hash": "TEXT",
                "frozen_at": "TIMESTAMPTZ",
            }
            for name, sql_type in additions.items():
                if name not in columns:
                    conn.execute(
                        f"ALTER TABLE research_snapshots "
                        f"ADD COLUMN {name} {sql_type}"
                    )

            conn.execute(
                "UPDATE research_snapshots "
                "SET created_at=COALESCE(created_at, CURRENT_TIMESTAMP)"
            )
            conn.execute(
                "UPDATE research_snapshots "
                "SET captured_at=COALESCE(captured_at, created_at)"
            )
            conn.execute(
                "UPDATE research_snapshots "
                "SET data_cutoff=COALESCE(data_cutoff, captured_at)"
            )
            conn.execute(
                "UPDATE research_snapshots "
                "SET as_of=COALESCE(as_of, data_cutoff)"
            )
            conn.execute(
                "UPDATE research_snapshots "
                "SET trading_date=COALESCE("
                "NULLIF(trading_date, ''), "
                "CAST(CAST(as_of AS DATE) AS VARCHAR))"
            )
            conn.execute(
                "UPDATE research_snapshots SET source_manifest="
                "COALESCE(source_manifest, '[]')"
            )
            conn.execute(
                "UPDATE research_snapshots SET quality='UNKNOWN' "
                "WHERE quality IS NULL OR quality NOT IN "
                "('GOOD','DEGRADED','PARTIAL','FAILED','UNKNOWN')"
            )
            conn.execute(
                "UPDATE research_snapshots SET quality_score=0 "
                "WHERE quality_score IS NULL OR quality_score < 0 "
                "OR quality_score > 1"
            )
            conn.execute(
                "UPDATE research_snapshots SET payload_version="
                "COALESCE(NULLIF(payload_version, ''), 'legacy')"
            )
            conn.execute(
                "UPDATE research_snapshots SET payload="
                "COALESCE(payload, '{}')"
            )
            conn.execute(
                "UPDATE research_snapshots SET run_id=COALESCE(run_id, '')"
            )
            conn.execute(
                "UPDATE research_snapshots SET schema_version="
                "COALESCE(schema_version, 1)"
            )
            conn.execute(
                "UPDATE research_snapshots SET frozen_at="
                "COALESCE(frozen_at, created_at)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS research_snapshot_run_bindings (
                    run_id TEXT,
                    symbol TEXT,
                    trading_date TEXT,
                    snapshot_id TEXT,
                    bound_at TIMESTAMPTZ,
                    PRIMARY KEY(run_id, symbol)
                )
                """
            )
            conn.commit()
        finally:
            conn.close()
        self._backfill_content_hashes()

    def _backfill_content_hashes(self) -> None:
        conn = self._connect(read_only=True)
        try:
            ids = [
                row[0]
                for row in conn.execute(
                    "SELECT snapshot_id FROM research_snapshots "
                    "WHERE content_hash IS NULL OR content_hash=''"
                ).fetchall()
            ]
        finally:
            conn.close()
        for snapshot_id in ids:
            snapshot = self.get(snapshot_id)
            if snapshot is None:
                continue
            conn = self._connect()
            try:
                conn.execute(
                    "UPDATE research_snapshots SET content_hash=? "
                    "WHERE snapshot_id=?",
                    [snapshot_content_hash(snapshot), snapshot_id],
                )
                conn.commit()
            finally:
                conn.close()

    def save_or_get(
        self,
        snapshot: ResearchSnapshot,
    ) -> SnapshotSaveResult:
        value = snapshot_to_dict(snapshot)
        content_hash = snapshot_content_hash(snapshot)
        existing = self.get(snapshot.snapshot_id)
        if existing is not None:
            if snapshot_content_hash(existing) != content_hash:
                raise ValueError("SNAPSHOT_IMMUTABLE_CONFLICT")
            return SnapshotSaveResult(
                snapshot_id=snapshot.snapshot_id,
                created=False,
                deduplicated=True,
            )

        conn = self._connect()
        try:
            duplicate = conn.execute(
                "SELECT snapshot_id FROM research_snapshots "
                "WHERE content_hash=? ORDER BY created_at, snapshot_id LIMIT 1",
                [content_hash],
            ).fetchone()
            if duplicate:
                return SnapshotSaveResult(
                    snapshot_id=str(duplicate[0]),
                    created=False,
                    deduplicated=True,
                )
            conn.execute(
                """
                INSERT INTO research_snapshots
                (snapshot_id, symbol, trading_date, as_of, data_cutoff,
                 captured_at, source_manifest, quality, quality_score,
                 payload_version, payload, run_id, schema_version, created_at,
                 content_hash, frozen_at)
                VALUES
                (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                [
                    snapshot.snapshot_id,
                    snapshot.symbol,
                    snapshot.trading_date,
                    snapshot.as_of,
                    snapshot.data_cutoff,
                    snapshot.captured_at,
                    json.dumps(
                        value["source_manifest"],
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    snapshot.quality.value,
                    snapshot.quality_score,
                    snapshot.payload_version,
                    json.dumps(
                        snapshot.payload,
                        sort_keys=True,
                        separators=(",", ":"),
                        default=str,
                    ),
                    snapshot.run_id,
                    snapshot.schema_version,
                    snapshot.created_at,
                    content_hash,
                    snapshot.created_at,
                ],
            )
            conn.commit()
            return SnapshotSaveResult(
                snapshot_id=snapshot.snapshot_id,
                created=True,
                deduplicated=False,
            )
        finally:
            conn.close()

    def save(self, snapshot: ResearchSnapshot) -> bool:
        return self.save_or_get(snapshot).created

    def get(self, snapshot_id: str) -> ResearchSnapshot | None:
        conn = self._connect(read_only=True)
        try:
            cursor = conn.execute(
                "SELECT * FROM research_snapshots WHERE snapshot_id=?",
                [snapshot_id],
            )
            row = cursor.fetchone()
            if row is None:
                return None
            columns = [item[0] for item in cursor.description]
        finally:
            conn.close()
        value = dict(zip(columns, row))
        value["source_manifest"] = _decode_manifest(
            value.get("source_manifest")
        )
        value["payload"] = _decode_json_object(value.get("payload"))
        return snapshot_from_dict(value)

    def bind_to_run(
        self,
        *,
        run_id: str,
        symbol: str,
        trading_date: str,
        snapshot_id: str,
        bound_at: datetime | None = None,
    ) -> bool:
        snapshot = self.get(snapshot_id)
        if snapshot is None:
            raise ValueError("SNAPSHOT_BINDING_TARGET_MISSING")
        normalized_symbol = symbol.strip().upper()
        if snapshot.symbol != normalized_symbol:
            raise ValueError("SNAPSHOT_BINDING_SYMBOL_MISMATCH")
        if snapshot.trading_date != trading_date:
            raise ValueError("SNAPSHOT_BINDING_DATE_MISMATCH")
        if snapshot.run_id and snapshot.run_id != run_id:
            raise ValueError("SNAPSHOT_BINDING_RUN_MISMATCH")
        conn = self._connect()
        try:
            existing = conn.execute(
                "SELECT snapshot_id, trading_date "
                "FROM research_snapshot_run_bindings "
                "WHERE run_id=? AND symbol=?",
                [run_id, normalized_symbol],
            ).fetchone()
            if existing:
                if (
                    str(existing[0]) != snapshot_id
                    or str(existing[1]) != trading_date
                ):
                    raise ValueError(
                        "RUN_SNAPSHOT_BINDING_IMMUTABLE"
                    )
                return False
            conn.execute(
                "INSERT INTO research_snapshot_run_bindings "
                "VALUES (?,?,?,?,?)",
                [
                    run_id,
                    normalized_symbol,
                    trading_date,
                    snapshot_id,
                    bound_at or datetime.now(timezone.utc),
                ],
            )
            conn.commit()
            return True
        finally:
            conn.close()

    def replay(self, snapshot_id: str) -> ResearchSnapshot:
        snapshot = self.get(snapshot_id)
        if snapshot is None:
            raise KeyError(snapshot_id)
        conn = self._connect(read_only=True)
        try:
            stored_hash = conn.execute(
                "SELECT content_hash FROM research_snapshots "
                "WHERE snapshot_id=?",
                [snapshot_id],
            ).fetchone()[0]
        finally:
            conn.close()
        if stored_hash != snapshot_content_hash(snapshot):
            raise ValueError("SNAPSHOT_CONTENT_HASH_MISMATCH")
        return snapshot_from_dict(snapshot_to_dict(snapshot))

    def replay_for_run(
        self,
        run_id: str,
        symbol: str,
    ) -> ResearchSnapshot:
        conn = self._connect(read_only=True)
        try:
            row = conn.execute(
                "SELECT snapshot_id FROM research_snapshot_run_bindings "
                "WHERE run_id=? AND symbol=?",
                [run_id, symbol.strip().upper()],
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            raise KeyError(f"{run_id}:{symbol.strip().upper()}")
        return self.replay(str(row[0]))

    @staticmethod
    def retention_policy() -> dict[str, Any]:
        return {
            "mode": "KEEP_ALL",
            "automatic_delete": False,
            "minimum_days": None,
        }

    def list_for_symbol(
        self,
        symbol: str,
        *,
        trading_date: str | None = None,
    ) -> list[ResearchSnapshot]:
        conn = self._connect(read_only=True)
        try:
            query = (
                "SELECT snapshot_id FROM research_snapshots "
                "WHERE symbol=?"
            )
            params: list[Any] = [symbol.strip().upper()]
            if trading_date is not None:
                query += " AND trading_date=?"
                params.append(trading_date)
            query += " ORDER BY captured_at, snapshot_id"
            ids = [
                row[0]
                for row in conn.execute(query, params).fetchall()
            ]
        finally:
            conn.close()
        return [
            snapshot
            for snapshot_id in ids
            if (snapshot := self.get(snapshot_id)) is not None
        ]
