"""Deterministic, auditable focus-pool construction."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import duckdb

from .universe_registry import UniverseRegistryStore


def _aware(value: datetime, code: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(code)


@dataclass(frozen=True)
class FocusPoolInput:
    symbol: str
    holdout_reliable: bool
    holdout_score: float
    average_dollar_volume: float
    data_quality: float

    def normalized(self) -> FocusPoolInput:
        values = (
            self.holdout_score,
            self.average_dollar_volume,
            self.data_quality,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("FOCUS_POOL_INPUT_NON_FINITE")
        if not 0 <= self.holdout_score <= 1:
            raise ValueError("FOCUS_POOL_HOLDOUT_SCORE_INVALID")
        if self.average_dollar_volume < 0:
            raise ValueError("FOCUS_POOL_LIQUIDITY_INVALID")
        if not 0 <= self.data_quality <= 1:
            raise ValueError("FOCUS_POOL_DATA_QUALITY_INVALID")
        symbol = self.symbol.strip().upper()
        if not symbol:
            raise ValueError("FOCUS_POOL_SYMBOL_REQUIRED")
        return FocusPoolInput(
            symbol=symbol,
            holdout_reliable=bool(self.holdout_reliable),
            holdout_score=float(self.holdout_score),
            average_dollar_volume=float(self.average_dollar_volume),
            data_quality=float(self.data_quality),
        )


@dataclass(frozen=True)
class FocusPoolPolicy:
    max_size: int = 50
    min_pool_size: int = 1
    min_average_dollar_volume: float = 1_000_000
    liquidity_target: float = 20_000_000
    min_data_quality: float = 0.8
    require_reliable_holdout: bool = True

    def __post_init__(self) -> None:
        if self.max_size < 1 or not 1 <= self.min_pool_size <= self.max_size:
            raise ValueError("FOCUS_POOL_SIZE_POLICY_INVALID")
        numeric = (
            self.min_average_dollar_volume,
            self.liquidity_target,
            self.min_data_quality,
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("FOCUS_POOL_POLICY_NON_FINITE")
        if (
            self.min_average_dollar_volume < 0
            or self.liquidity_target <= 0
            or not 0 <= self.min_data_quality <= 1
        ):
            raise ValueError("FOCUS_POOL_POLICY_INVALID")


class FocusPoolStore:
    """Completed pools are immutable; failed rebuilds preserve the prior pool."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self.universes = UniverseRegistryStore(db_path)
        self._migrate()

    def _connect(self, *, read_only: bool = False):
        return duckdb.connect(self.db_path, read_only=read_only)

    def _migrate(self) -> None:
        connection = self._connect()
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS focus_pool_versions (
                    pool_id TEXT PRIMARY KEY,
                    pool_name TEXT,
                    universe_version TEXT,
                    policy_json TEXT,
                    input_hash TEXT,
                    member_count INTEGER,
                    eligible_input_count INTEGER,
                    as_of TIMESTAMPTZ,
                    created_at TIMESTAMPTZ
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS focus_pool_decisions (
                    pool_id TEXT,
                    symbol TEXT,
                    included BOOLEAN,
                    rank INTEGER,
                    score DOUBLE,
                    reasons_json TEXT,
                    input_json TEXT,
                    PRIMARY KEY (pool_id, symbol)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS focus_pool_failures (
                    failure_id TEXT PRIMARY KEY,
                    pool_name TEXT,
                    universe_version TEXT,
                    previous_pool_id TEXT,
                    error_code TEXT,
                    as_of TIMESTAMPTZ,
                    created_at TIMESTAMPTZ
                )
                """
            )
            connection.commit()
        finally:
            connection.close()

    def attempt_build(
        self,
        *,
        pool_name: str,
        universe_version: str,
        inputs: Iterable[FocusPoolInput],
        policy: FocusPoolPolicy,
        as_of: datetime,
        created_at: datetime,
        source_complete: bool = True,
    ) -> dict[str, Any]:
        """Build, or return the previous valid pool with explicit failure evidence."""
        previous = self.latest(pool_name)
        try:
            return self._build(
                pool_name=pool_name,
                universe_version=universe_version,
                inputs=inputs,
                policy=policy,
                as_of=as_of,
                created_at=created_at,
                source_complete=source_complete,
            )
        except Exception as exc:
            error_code = str(exc)
            if not error_code or " " in error_code:
                error_code = type(exc).__name__.upper()
            failure_payload = {
                "pool_name": pool_name,
                "universe_version": universe_version,
                "previous_pool_id": (
                    previous["pool_id"] if previous is not None else ""
                ),
                "error_code": error_code,
                "as_of": as_of.isoformat(),
            }
            failure_id = "focus-failure-" + self._digest(failure_payload, 24)
            connection = self._connect()
            try:
                connection.execute(
                    """
                    INSERT INTO focus_pool_failures VALUES (?,?,?,?,?,?,?)
                    ON CONFLICT (failure_id) DO NOTHING
                    """,
                    [
                        failure_id,
                        pool_name,
                        universe_version,
                        failure_payload["previous_pool_id"],
                        error_code,
                        as_of,
                        created_at,
                    ],
                )
                connection.commit()
            finally:
                connection.close()
            if previous is None:
                raise
            return {
                **previous,
                "preserved_after_failure": True,
                "failure_id": failure_id,
                "failure_code": error_code,
            }

    def _build(
        self,
        *,
        pool_name: str,
        universe_version: str,
        inputs: Iterable[FocusPoolInput],
        policy: FocusPoolPolicy,
        as_of: datetime,
        created_at: datetime,
        source_complete: bool,
    ) -> dict[str, Any]:
        name = pool_name.strip()
        if not name:
            raise ValueError("FOCUS_POOL_NAME_REQUIRED")
        _aware(as_of, "FOCUS_POOL_AS_OF_TZ_REQUIRED")
        _aware(created_at, "FOCUS_POOL_CREATED_TZ_REQUIRED")
        if created_at < as_of:
            raise ValueError("FOCUS_POOL_CREATED_BEFORE_AS_OF")
        if not source_complete:
            raise ValueError("FOCUS_POOL_SOURCE_INCOMPLETE")
        universe = self.universes.get_version(universe_version)
        if universe is None:
            raise ValueError("FOCUS_POOL_UNIVERSE_NOT_FOUND")
        universe_symbols = {asset.symbol for asset in universe["assets"]}
        by_symbol: dict[str, FocusPoolInput] = {}
        for raw in inputs:
            if not isinstance(raw, FocusPoolInput):
                raise ValueError("FOCUS_POOL_INPUT_INVALID")
            item = raw.normalized()
            if item.symbol not in universe_symbols:
                raise ValueError("FOCUS_POOL_SYMBOL_OUTSIDE_UNIVERSE")
            existing = by_symbol.get(item.symbol)
            if existing is not None and existing != item:
                raise ValueError("FOCUS_POOL_DUPLICATE_INPUT_CONFLICT")
            by_symbol[item.symbol] = item
        eligible_symbols = {
            asset.symbol
            for asset in universe["assets"]
            if asset.status == "ACTIVE" and asset.tradable
        }
        if not eligible_symbols.issubset(by_symbol):
            raise ValueError("FOCUS_POOL_ELIGIBLE_INPUT_MISSING")

        decisions: list[dict[str, Any]] = []
        passing: list[dict[str, Any]] = []
        for asset in universe["assets"]:
            item = by_symbol.get(asset.symbol)
            reasons = []
            if asset.status != "ACTIVE":
                reasons.append(f"ASSET_{asset.status}")
            if not asset.tradable:
                reasons.append("ASSET_NOT_TRADABLE")
            if item is not None:
                if policy.require_reliable_holdout and not item.holdout_reliable:
                    reasons.append("HOLDOUT_UNRELIABLE")
                if (
                    item.average_dollar_volume
                    < policy.min_average_dollar_volume
                ):
                    reasons.append("LIQUIDITY_BELOW_MINIMUM")
                if item.data_quality < policy.min_data_quality:
                    reasons.append("DATA_QUALITY_BELOW_MINIMUM")
                score = self._score(item, policy)
                input_payload = asdict(item)
            else:
                score = 0.0
                input_payload = {}
            decision = {
                "symbol": asset.symbol,
                "included": False,
                "rank": 0,
                "score": score,
                "reasons": reasons,
                "input": input_payload,
            }
            decisions.append(decision)
            if not reasons:
                passing.append(decision)
        passing.sort(key=lambda row: (-row["score"], row["symbol"]))
        for rank, decision in enumerate(passing, start=1):
            if rank <= policy.max_size:
                decision["included"] = True
                decision["rank"] = rank
            else:
                decision["reasons"].append("BELOW_RANK_CUTOFF")
        included = [row for row in decisions if row["included"]]
        if len(included) < policy.min_pool_size:
            raise ValueError("FOCUS_POOL_BELOW_MINIMUM_SIZE")

        input_payload = {
            symbol: asdict(item)
            for symbol, item in sorted(by_symbol.items())
        }
        canonical = json.dumps(
            {
                "pool_name": name,
                "universe_version": universe_version,
                "policy": asdict(policy),
                "inputs": input_payload,
                "as_of": as_of.isoformat(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        input_hash = hashlib.sha256(canonical.encode()).hexdigest()
        pool_id = "focus-pool-" + input_hash[:24]
        connection = self._connect()
        try:
            connection.execute("BEGIN TRANSACTION")
            connection.execute(
                """
                INSERT INTO focus_pool_versions VALUES (?,?,?,?,?,?,?,?,?)
                ON CONFLICT (pool_id) DO NOTHING
                """,
                [
                    pool_id,
                    name,
                    universe_version,
                    json.dumps(asdict(policy), sort_keys=True),
                    input_hash,
                    len(included),
                    len(eligible_symbols),
                    as_of,
                    created_at,
                ],
            )
            for decision in decisions:
                connection.execute(
                    """
                    INSERT INTO focus_pool_decisions VALUES (?,?,?,?,?,?,?)
                    ON CONFLICT (pool_id, symbol) DO NOTHING
                    """,
                    [
                        pool_id,
                        decision["symbol"],
                        decision["included"],
                        decision["rank"],
                        decision["score"],
                        json.dumps(decision["reasons"]),
                        json.dumps(decision["input"], sort_keys=True),
                    ],
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        pool = self.get(pool_id)
        if pool is None:  # pragma: no cover
            raise RuntimeError("FOCUS_POOL_PERSIST_FAILED")
        return pool

    def get(self, pool_id: str) -> dict[str, Any] | None:
        connection = self._connect(read_only=True)
        try:
            row = connection.execute(
                "SELECT * FROM focus_pool_versions WHERE pool_id=?",
                [pool_id],
            ).fetchone()
            decisions = (
                connection.execute(
                    """
                    SELECT symbol, included, rank, score, reasons_json,
                           input_json
                    FROM focus_pool_decisions
                    WHERE pool_id=?
                    ORDER BY included DESC, rank, symbol
                    """,
                    [pool_id],
                ).fetchall()
                if row is not None
                else []
            )
        finally:
            connection.close()
        if row is None:
            return None
        return {
            "pool_id": str(row[0]),
            "pool_name": str(row[1]),
            "universe_version": str(row[2]),
            "policy": json.loads(row[3]),
            "input_hash": str(row[4]),
            "member_count": int(row[5]),
            "eligible_input_count": int(row[6]),
            "as_of": row[7],
            "created_at": row[8],
            "decisions": [
                {
                    "symbol": str(item[0]),
                    "included": bool(item[1]),
                    "rank": int(item[2]),
                    "score": float(item[3]),
                    "reasons": json.loads(item[4]),
                    "input": json.loads(item[5]),
                }
                for item in decisions
            ],
            "preserved_after_failure": False,
        }

    def latest(self, pool_name: str) -> dict[str, Any] | None:
        connection = self._connect(read_only=True)
        try:
            row = connection.execute(
                """
                SELECT pool_id FROM focus_pool_versions
                WHERE pool_name=?
                ORDER BY as_of DESC, created_at DESC, pool_id DESC
                LIMIT 1
                """,
                [pool_name],
            ).fetchone()
        finally:
            connection.close()
        return self.get(str(row[0])) if row is not None else None

    @staticmethod
    def _score(item: FocusPoolInput, policy: FocusPoolPolicy) -> float:
        liquidity = min(
            1.0,
            item.average_dollar_volume / policy.liquidity_target,
        )
        return round(
            item.holdout_score * 50
            + item.data_quality * 30
            + liquidity * 20,
            6,
        )

    @staticmethod
    def _digest(payload: dict[str, Any], length: int) -> str:
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode()).hexdigest()[:length]
