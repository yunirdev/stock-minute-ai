"""Concrete Data Hub adapters for market and Alpaca broker facts."""
from __future__ import annotations

import math
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

import pandas as pd

from .data_hub import (
    AdapterResult,
    DataDomain,
    DataEnvelope,
    DataStatus,
    SourceRegistry,
    SourceSpec,
)

MarketLoader = Callable[[str, str], Any]


def _value(item: Any, *names: str) -> Any:
    if isinstance(item, dict):
        for name in names:
            if name in item:
                return item[name]
    for name in names:
        if hasattr(item, name):
            return getattr(item, name)
    return None


def _canonical_bars(raw: Any, symbol: str) -> list[dict[str, Any]]:
    if isinstance(raw, pd.DataFrame):
        frame = raw.copy()
        frame.columns = [str(column).lower() for column in frame.columns]
        items: Iterable[Any] = frame.to_dict(orient="records")
    else:
        items = list(raw or [])
    rows = []
    for item in items:
        timestamp = _value(item, "timestamp_utc", "timestamp", "t")
        if timestamp is None:
            continue
        timestamp = pd.Timestamp(timestamp)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        row = {
            "symbol": str(_value(item, "symbol", "S") or symbol).upper(),
            "timestamp_utc": timestamp.isoformat(),
            "open": float(_value(item, "open", "o")),
            "high": float(_value(item, "high", "h")),
            "low": float(_value(item, "low", "l")),
            "close": float(_value(item, "close", "c")),
            "volume": float(_value(item, "volume", "v") or 0.0),
        }
        if all(
            math.isfinite(row[field_name])
            for field_name in ("open", "high", "low", "close", "volume")
        ):
            rows.append(row)
    rows.sort(key=lambda row: row["timestamp_utc"])
    return rows


def market_adapter(
    loader: MarketLoader,
    *,
    upstream: str,
    execution_eligible: bool,
    quality_score: float,
):
    def fetch(request) -> AdapterResult:
        timeframe = str(request.params.get("timeframe", "5m"))
        rows = _canonical_bars(
            loader(request.key, timeframe),
            request.key,
        )
        if not rows:
            raise ValueError("DATA_MARKET_BARS_EMPTY")
        as_of = datetime.fromisoformat(rows[-1]["timestamp_utc"])
        return AdapterResult(
            payload={
                "symbol": request.key,
                "timeframe": timeframe,
                "bars": rows,
                "last_price": rows[-1]["close"],
                "execution_eligible": execution_eligible,
            },
            as_of=as_of,
            quality_score=quality_score,
            metadata={
                "upstream": upstream,
                "execution_eligible": execution_eligible,
            },
        )

    return fetch


def register_market_sources(
    registry: SourceRegistry,
    *,
    alpaca_loader: MarketLoader,
    local_cache_loader: MarketLoader,
    yahoo_loader: MarketLoader,
) -> None:
    required = (
        "symbol",
        "timeframe",
        "bars",
        "last_price",
        "execution_eligible",
    )
    registry.register(
        SourceSpec(
            source_id="alpaca_market",
            domain=DataDomain.MARKET,
            adapter=market_adapter(
                alpaca_loader,
                upstream="alpaca",
                execution_eligible=True,
                quality_score=1.0,
            ),
            priority=0,
            timeout_seconds=10.0,
            ttl_seconds=30.0,
            max_stale_seconds=0.0,
            required_fields=required,
        )
    )
    registry.register(
        SourceSpec(
            source_id="local_bar_cache",
            domain=DataDomain.MARKET,
            adapter=market_adapter(
                local_cache_loader,
                upstream="local_bar_cache",
                execution_eligible=False,
                quality_score=0.8,
            ),
            priority=10,
            timeout_seconds=2.0,
            ttl_seconds=60.0,
            max_stale_seconds=300.0,
            required_fields=required,
            quality_cap=0.8,
        )
    )
    registry.register(
        SourceSpec(
            source_id="yahoo_market_fallback",
            domain=DataDomain.MARKET,
            adapter=market_adapter(
                yahoo_loader,
                upstream="yahoo",
                execution_eligible=False,
                quality_score=0.6,
            ),
            priority=20,
            timeout_seconds=10.0,
            ttl_seconds=60.0,
            max_stale_seconds=300.0,
            required_fields=required,
            quality_cap=0.6,
        )
    )


def _serialize(value: Any) -> Any:
    if is_dataclass(value):
        return {
            key: _serialize(item)
            for key, item in asdict(value).items()
        }
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if hasattr(value, "value"):
        return _serialize(value.value)
    return value


def alpaca_broker_facts_adapter(broker):
    def fetch(request) -> AdapterResult:
        return AdapterResult(
            payload={
                "equity": float(broker.get_account_equity()),
                "positions": _serialize(list(broker.get_positions())),
                "open_orders": _serialize(list(broker.get_open_orders())),
                "recent_fills": _serialize(list(broker.get_recent_fills())),
                "execution_eligible": True,
            },
            as_of=request.requested_at,
            quality_score=1.0,
            metadata={"upstream": "alpaca", "authoritative": True},
        )

    return fetch


def register_alpaca_broker_facts(
    registry: SourceRegistry,
    broker,
) -> None:
    registry.register(
        SourceSpec(
            source_id="alpaca_broker_facts",
            domain=DataDomain.BROKER,
            adapter=alpaca_broker_facts_adapter(broker),
            priority=0,
            timeout_seconds=10.0,
            ttl_seconds=5.0,
            required_fields=(
                "equity",
                "positions",
                "open_orders",
                "recent_fills",
                "execution_eligible",
            ),
        )
    )


def compare_market_envelopes(
    primary: DataEnvelope,
    shadow: DataEnvelope,
    *,
    price_tolerance_bps: float = 5.0,
) -> dict[str, Any]:
    if primary.status == DataStatus.FAILED or shadow.status == DataStatus.FAILED:
        return {
            "comparable": False,
            "classification": "SOURCE_UNAVAILABLE",
            "differences": [],
        }
    primary_price = float(primary.payload["last_price"])
    shadow_price = float(shadow.payload["last_price"])
    price_bps = (
        abs(primary_price - shadow_price) / primary_price * 10_000
        if primary_price
        else float("inf")
    )
    differences = []
    if price_bps > price_tolerance_bps:
        differences.append(
            {
                "field": "last_price",
                "primary": primary_price,
                "shadow": shadow_price,
                "difference_bps": price_bps,
                "classification": "PRICE_DIFFERENCE",
            }
        )
    primary_count = len(primary.payload.get("bars", []))
    shadow_count = len(shadow.payload.get("bars", []))
    if primary_count != shadow_count:
        differences.append(
            {
                "field": "bar_count",
                "primary": primary_count,
                "shadow": shadow_count,
                "classification": "COVERAGE_DIFFERENCE",
            }
        )
    return {
        "comparable": True,
        "classification": "MATCH" if not differences else "DIFFERENT",
        "differences": differences,
    }
