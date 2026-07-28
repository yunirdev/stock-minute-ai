"""Read-only Data Hub shadow cycles for production source validation.

This module deliberately has no Runtime, broker, or order dependency.  It
performs two independent reads from the configured Alpaca market-data feed,
compares the legacy feed envelope with the Data Hub envelope, and persists only
quality observations and reports.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence
from zoneinfo import ZoneInfo

from .config import TradingConfig, settings
from .data_feed import AlpacaDataFeed
from .data_hub import (
    DataDomain,
    DataEnvelope,
    DataHub,
    DataRequest,
    DataStatus,
    SourceFailure,
    SourceRegistry,
    SourceSpec,
)
from .data_hub_quality import (
    DataHubQualityStore,
    DoubleReadObservation,
    SourceReadMetrics,
    generate_data_hub_quality_report,
    observe_double_read,
)
from .data_hub_sources import market_adapter

_UTC = timezone.utc
_ET = ZoneInfo("America/New_York")


class MarketFeed(Protocol):
    def fetch_bars(self, symbol: str, n_bars: int = 120) -> list[Any]: ...


@dataclass(frozen=True)
class ShadowCycleResult:
    observed_at: datetime
    trading_date: str
    observations: tuple[DoubleReadObservation, ...]
    saved_observations: int
    quality_report: Mapping[str, Any]
    saved_report: bool

    @property
    def successful(self) -> bool:
        return all(
            item.comparable
            and item.unclassified_critical_differences == 0
            for item in self.observations
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": "READ_ONLY_SHADOW",
            "execution_input_switched": False,
            "observed_at": self.observed_at.isoformat(),
            "trading_date": self.trading_date,
            "successful": self.successful,
            "saved_observations": self.saved_observations,
            "saved_report": self.saved_report,
            "observations": [
                {
                    "observation_id": item.observation_id,
                    "symbol": item.key,
                    "primary_source": item.primary_source,
                    "shadow_source": item.shadow_source,
                    "primary_status": item.primary_status.value,
                    "shadow_status": item.shadow_status.value,
                    "comparable": item.comparable,
                    "differences": len(item.differences),
                    "unclassified_critical_differences": (
                        item.unclassified_critical_differences
                    ),
                    "primary_latency_ms": round(
                        item.primary_metrics.latency_ms,
                        3,
                    ),
                    "shadow_latency_ms": round(
                        item.shadow_metrics.latency_ms,
                        3,
                    ),
                }
                for item in self.observations
            ],
            "quality_report": dict(self.quality_report),
        }


def _request_id(source_id: str, symbol: str, timeframe: str) -> str:
    raw = f"{source_id}:{symbol}:{timeframe}".encode()
    return "shadow-read-" + hashlib.sha256(raw).hexdigest()[:20]


def _failed_envelope(
    *,
    source_id: str,
    symbol: str,
    timeframe: str,
    observed_at: datetime,
    failure_code: str,
) -> DataEnvelope:
    return DataEnvelope(
        request_id=_request_id(source_id, symbol, timeframe),
        domain=DataDomain.MARKET,
        key=symbol,
        source_id=source_id,
        status=DataStatus.FAILED,
        payload={},
        as_of=observed_at,
        fetched_at=observed_at,
        expires_at=observed_at,
        quality_score=0.0,
        failure_code=failure_code,
        failures=(SourceFailure(source_id, failure_code),),
    )


class ShadowDataHubRunner:
    """Compare legacy and Data Hub market reads without changing execution."""

    def __init__(
        self,
        *,
        feed: MarketFeed,
        store: DataHubQualityStore,
        timeframe: str = "5m",
        n_bars: int = 120,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if n_bars < 1:
            raise ValueError("DATA_HUB_SHADOW_BARS_INVALID")
        if not timeframe.strip():
            raise ValueError("DATA_HUB_SHADOW_TIMEFRAME_REQUIRED")
        self.feed = feed
        self.store = store
        self.timeframe = timeframe
        self.n_bars = n_bars
        self.clock = clock or (lambda: datetime.now(_UTC))

    @staticmethod
    def _aware(value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("DATA_HUB_SHADOW_CLOCK_TIMEZONE_REQUIRED")
        return value.astimezone(_UTC)

    def _loader(self, symbol: str, _timeframe: str) -> list[Any]:
        return self.feed.fetch_bars(symbol, n_bars=self.n_bars)

    def _legacy_read(
        self,
        symbol: str,
        observed_at: datetime,
    ) -> tuple[DataEnvelope, SourceReadMetrics]:
        source_id = "runtime_alpaca_feed"
        request = DataRequest(
            domain=DataDomain.MARKET,
            key=symbol,
            requested_at=observed_at,
            params={"timeframe": self.timeframe},
        )
        adapter = market_adapter(
            self._loader,
            upstream="alpaca_runtime_legacy",
            execution_eligible=True,
            quality_score=1.0,
        )
        started = time.perf_counter()
        try:
            result = adapter(request)
            envelope = DataEnvelope(
                request_id=_request_id(
                    source_id,
                    symbol,
                    self.timeframe,
                ),
                domain=DataDomain.MARKET,
                key=symbol,
                source_id=source_id,
                status=DataStatus.OK,
                payload=dict(result.payload),
                as_of=result.as_of,
                fetched_at=observed_at,
                expires_at=observed_at,
                quality_score=result.quality_score,
                metadata=dict(result.metadata),
            )
        except Exception as exc:
            failure_code = (
                str(exc)
                if isinstance(exc, ValueError)
                and str(exc).startswith("DATA_")
                else type(exc).__name__
            )
            envelope = _failed_envelope(
                source_id=source_id,
                symbol=symbol,
                timeframe=self.timeframe,
                observed_at=observed_at,
                failure_code=failure_code,
            )
        latency_ms = (time.perf_counter() - started) * 1_000
        return envelope, SourceReadMetrics(
            source_id=source_id,
            latency_ms=latency_ms,
            failure_count=int(envelope.status == DataStatus.FAILED),
        )

    def _build_hub(self, observed_at: datetime) -> DataHub:
        registry = SourceRegistry()
        registry.register(
            SourceSpec(
                source_id="alpaca_market",
                domain=DataDomain.MARKET,
                adapter=market_adapter(
                    self._loader,
                    upstream="alpaca",
                    execution_eligible=True,
                    quality_score=1.0,
                ),
                priority=0,
                timeout_seconds=20.0,
                ttl_seconds=0.0,
                required_fields=(
                    "symbol",
                    "timeframe",
                    "bars",
                    "last_price",
                    "execution_eligible",
                ),
            )
        )
        return DataHub(registry, clock=lambda: observed_at)

    def run(
        self,
        symbols: Iterable[str],
        *,
        trading_date: str | None = None,
    ) -> ShadowCycleResult:
        normalized = tuple(
            dict.fromkeys(
                symbol.strip().upper()
                for symbol in symbols
                if symbol.strip()
            )
        )
        if not normalized:
            raise ValueError("DATA_HUB_SHADOW_SYMBOLS_REQUIRED")
        observed_at = self._aware(self.clock())
        hub = self._build_hub(observed_at)
        reads: list[
            tuple[DataEnvelope, DataEnvelope, SourceReadMetrics, SourceReadMetrics]
        ] = []
        try:
            for symbol in normalized:
                primary, primary_metrics = self._legacy_read(
                    symbol,
                    observed_at,
                )
                started = time.perf_counter()
                shadow = hub.fetch(
                    DataDomain.MARKET,
                    symbol,
                    params={"timeframe": self.timeframe},
                )
                shadow_metrics = SourceReadMetrics(
                    source_id="alpaca_market",
                    latency_ms=(time.perf_counter() - started) * 1_000,
                    failure_count=int(shadow.status == DataStatus.FAILED),
                )
                reads.append(
                    (
                        primary,
                        shadow,
                        primary_metrics,
                        shadow_metrics,
                    )
                )
        finally:
            hub.close()

        effective_trading_date = trading_date or self._trading_date(
            reads,
            observed_at,
        )
        observations = tuple(
            observe_double_read(
                primary,
                shadow,
                observed_at=observed_at,
                primary_metrics=primary_metrics,
                shadow_metrics=shadow_metrics,
                trading_date=effective_trading_date,
            )
            for primary, shadow, primary_metrics, shadow_metrics in reads
        )
        saved_observations = sum(
            self.store.save_observation(observation)
            for observation in observations
        )
        report = generate_data_hub_quality_report(
            self.store.load_observations(),
            generated_at=observed_at,
        )
        saved_report = self.store.save_report(report)
        return ShadowCycleResult(
            observed_at=observed_at,
            trading_date=effective_trading_date,
            observations=observations,
            saved_observations=saved_observations,
            quality_report=report,
            saved_report=saved_report,
        )

    @staticmethod
    def _trading_date(
        reads: Sequence[
            tuple[DataEnvelope, DataEnvelope, SourceReadMetrics, SourceReadMetrics]
        ],
        observed_at: datetime,
    ) -> str:
        as_of_values = [
            envelope.as_of
            for primary, shadow, _, _ in reads
            for envelope in (primary, shadow)
            if envelope.status != DataStatus.FAILED
        ]
        reference = max(as_of_values, default=observed_at)
        return reference.astimezone(_ET).date().isoformat()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run read-only Data Hub market double reads.",
    )
    parser.add_argument("--symbols", default="AAPL,MSFT")
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--bars", type=int, default=120)
    parser.add_argument("--db", default=settings.daily_research_db)
    parser.add_argument("--trading-date", default="")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    symbols = [
        symbol.strip().upper()
        for symbol in args.symbols.split(",")
        if symbol.strip()
    ]
    config = TradingConfig(
        symbols=symbols,
        timeframe=args.timeframe,
        auto_trade_paper=False,
    )
    if not config.alpaca_api_key or not config.alpaca_secret_key:
        print(
            json.dumps(
                {
                    "mode": "READ_ONLY_SHADOW",
                    "successful": False,
                    "error": "ALPACA_MARKET_DATA_CREDENTIALS_REQUIRED",
                    "execution_input_switched": False,
                },
                sort_keys=True,
            )
        )
        return 2
    result = ShadowDataHubRunner(
        feed=AlpacaDataFeed(config),
        store=DataHubQualityStore(args.db),
        timeframe=args.timeframe,
        n_bars=args.bars,
    ).run(
        symbols,
        trading_date=args.trading_date or None,
    )
    print(json.dumps(result.to_dict(), sort_keys=True))
    return 0 if result.successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
