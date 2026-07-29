"""Append-only stock, ETF, and fund universe versions."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import duckdb

_ASSET_TYPES = {"STOCK", "ETF", "FUND"}
_ASSET_STATUSES = {"ACTIVE", "INACTIVE", "DELISTED"}


def _aware(value: datetime, code: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(code)


def _text(value: str, code: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(code)
    return normalized


@dataclass(frozen=True)
class UniverseAsset:
    symbol: str
    asset_type: str
    exchange: str
    status: str
    tradable: bool
    source: str
    as_of: datetime

    def normalized(self) -> UniverseAsset:
        _aware(self.as_of, "UNIVERSE_ASSET_AS_OF_TZ_REQUIRED")
        symbol = _text(self.symbol, "UNIVERSE_ASSET_SYMBOL_REQUIRED").upper()
        asset_type = _text(
            self.asset_type,
            "UNIVERSE_ASSET_TYPE_REQUIRED",
        ).upper()
        status = _text(
            self.status,
            "UNIVERSE_ASSET_STATUS_REQUIRED",
        ).upper()
        if asset_type not in _ASSET_TYPES:
            raise ValueError("UNIVERSE_ASSET_TYPE_INVALID")
        if status not in _ASSET_STATUSES:
            raise ValueError("UNIVERSE_ASSET_STATUS_INVALID")
        return UniverseAsset(
            symbol=symbol,
            asset_type=asset_type,
            exchange=_text(
                self.exchange,
                "UNIVERSE_ASSET_EXCHANGE_REQUIRED",
            ).upper(),
            status=status,
            tradable=bool(self.tradable),
            source=_text(self.source, "UNIVERSE_ASSET_SOURCE_REQUIRED"),
            as_of=self.as_of,
        )


class UniverseRegistryStore:
    """Content-addressed universe versions; historical versions never change."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._migrate()

    def _connect(self, *, read_only: bool = False):
        return duckdb.connect(self.db_path, read_only=read_only)

    def _migrate(self) -> None:
        connection = self._connect()
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS universe_versions (
                    version_id TEXT PRIMARY KEY,
                    universe_name TEXT,
                    source_version TEXT,
                    content_hash TEXT,
                    as_of TIMESTAMPTZ,
                    asset_count INTEGER,
                    eligible_count INTEGER,
                    created_at TIMESTAMPTZ
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS universe_version_assets (
                    version_id TEXT,
                    symbol TEXT,
                    asset_type TEXT,
                    exchange TEXT,
                    status TEXT,
                    tradable BOOLEAN,
                    source TEXT,
                    as_of TIMESTAMPTZ,
                    PRIMARY KEY (version_id, symbol)
                )
                """
            )
            connection.commit()
        finally:
            connection.close()

    def create_version(
        self,
        *,
        universe_name: str,
        source_version: str,
        assets: Iterable[UniverseAsset],
        as_of: datetime,
        created_at: datetime,
    ) -> dict[str, Any]:
        name = _text(universe_name, "UNIVERSE_NAME_REQUIRED")
        source = _text(source_version, "UNIVERSE_SOURCE_VERSION_REQUIRED")
        _aware(as_of, "UNIVERSE_AS_OF_TZ_REQUIRED")
        _aware(created_at, "UNIVERSE_CREATED_TZ_REQUIRED")
        if created_at < as_of:
            raise ValueError("UNIVERSE_CREATED_BEFORE_AS_OF")
        by_symbol: dict[str, UniverseAsset] = {}
        for raw in assets:
            if not isinstance(raw, UniverseAsset):
                raise ValueError("UNIVERSE_ASSET_INVALID")
            asset = raw.normalized()
            if asset.as_of > as_of:
                raise ValueError("UNIVERSE_ASSET_FROM_FUTURE")
            existing = by_symbol.get(asset.symbol)
            if existing is not None and existing != asset:
                raise ValueError("UNIVERSE_DUPLICATE_SYMBOL_CONFLICT")
            by_symbol[asset.symbol] = asset
        if not by_symbol:
            raise ValueError("UNIVERSE_EMPTY")
        ordered = [by_symbol[symbol] for symbol in sorted(by_symbol)]
        canonical_assets = [
            {
                **asdict(asset),
                "as_of": asset.as_of.isoformat(),
            }
            for asset in ordered
        ]
        payload = {
            "universe_name": name,
            "source_version": source,
            "as_of": as_of.isoformat(),
            "assets": canonical_assets,
        }
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        )
        content_hash = hashlib.sha256(canonical.encode()).hexdigest()
        version_id = "universe-" + content_hash[:24]
        eligible_count = sum(
            asset.status == "ACTIVE" and asset.tradable
            for asset in ordered
        )
        connection = self._connect()
        try:
            connection.execute("BEGIN TRANSACTION")
            connection.execute(
                """
                INSERT INTO universe_versions VALUES (?,?,?,?,?,?,?,?)
                ON CONFLICT (version_id) DO NOTHING
                """,
                [
                    version_id,
                    name,
                    source,
                    content_hash,
                    as_of,
                    len(ordered),
                    eligible_count,
                    created_at,
                ],
            )
            for asset in ordered:
                connection.execute(
                    """
                    INSERT INTO universe_version_assets VALUES
                    (?,?,?,?,?,?,?,?)
                    ON CONFLICT (version_id, symbol) DO NOTHING
                    """,
                    [
                        version_id,
                        asset.symbol,
                        asset.asset_type,
                        asset.exchange,
                        asset.status,
                        asset.tradable,
                        asset.source,
                        asset.as_of,
                    ],
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        version = self.get_version(version_id)
        if version is None:  # pragma: no cover
            raise RuntimeError("UNIVERSE_PERSIST_FAILED")
        return version

    def get_version(self, version_id: str) -> dict[str, Any] | None:
        connection = self._connect(read_only=True)
        try:
            row = connection.execute(
                "SELECT * FROM universe_versions WHERE version_id=?",
                [version_id],
            ).fetchone()
            assets = (
                connection.execute(
                    """
                    SELECT symbol, asset_type, exchange, status, tradable,
                           source, as_of
                    FROM universe_version_assets
                    WHERE version_id=?
                    ORDER BY symbol
                    """,
                    [version_id],
                ).fetchall()
                if row is not None
                else []
            )
        finally:
            connection.close()
        if row is None:
            return None
        return {
            "version_id": str(row[0]),
            "universe_name": str(row[1]),
            "source_version": str(row[2]),
            "content_hash": str(row[3]),
            "as_of": row[4],
            "asset_count": int(row[5]),
            "eligible_count": int(row[6]),
            "created_at": row[7],
            "assets": [
                UniverseAsset(
                    symbol=str(asset[0]),
                    asset_type=str(asset[1]),
                    exchange=str(asset[2]),
                    status=str(asset[3]),
                    tradable=bool(asset[4]),
                    source=str(asset[5]),
                    as_of=asset[6],
                )
                for asset in assets
            ],
        }

    def latest(self, universe_name: str) -> dict[str, Any] | None:
        connection = self._connect(read_only=True)
        try:
            row = connection.execute(
                """
                SELECT version_id FROM universe_versions
                WHERE universe_name=?
                ORDER BY as_of DESC, created_at DESC, version_id DESC
                LIMIT 1
                """,
                [universe_name],
            ).fetchone()
        finally:
            connection.close()
        return self.get_version(str(row[0])) if row is not None else None

    def eligible_assets(self, version_id: str) -> tuple[UniverseAsset, ...]:
        version = self.get_version(version_id)
        if version is None:
            raise KeyError(version_id)
        return tuple(
            asset
            for asset in version["assets"]
            if asset.status == "ACTIVE" and asset.tradable
        )
