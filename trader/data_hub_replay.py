"""Auditable historical Data Hub replay evidence.

Historical replay accelerates value/normalization verification but never
pretends to be a live operational observation.  Replay rows and reports use
dedicated DuckDB tables and cannot satisfy the live execution-cutover gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd

from .config import settings
from .data_cache import fetch_alpaca_bars_window, get_bars
from .data_hub_quality import (
    DataHubQualityStore,
    generate_data_hub_quality_report,
)

_UTC = timezone.utc
_ET = ZoneInfo("America/New_York")
_PRICE_FIELDS = ("open", "high", "low", "close")
_VALUE_FIELDS = (*_PRICE_FIELDS, "volume")


@dataclass(frozen=True)
class HistoricalReplayObservation:
    replay_id: str
    run_id: str
    generated_at: datetime
    trading_date: str
    symbol: str
    timeframe: str
    primary_source: str
    shadow_source: str
    matched: bool
    differences: tuple[Mapping[str, Any], ...]
    primary: Mapping[str, float]
    shadow: Mapping[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "generated_at": self.generated_at.astimezone(_UTC).isoformat(),
            "differences": [dict(item) for item in self.differences],
            "primary": dict(self.primary),
            "shadow": dict(self.shadow),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]):
        return cls(
            replay_id=str(payload["replay_id"]),
            run_id=str(payload["run_id"]),
            generated_at=datetime.fromisoformat(
                str(payload["generated_at"])
            ),
            trading_date=str(payload["trading_date"]),
            symbol=str(payload["symbol"]),
            timeframe=str(payload["timeframe"]),
            primary_source=str(payload["primary_source"]),
            shadow_source=str(payload["shadow_source"]),
            matched=bool(payload["matched"]),
            differences=tuple(payload.get("differences") or ()),
            primary=dict(payload["primary"]),
            shadow=dict(payload["shadow"]),
        )


class DataHubReplayStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        conn = duckdb.connect(str(self.db_path))
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS data_hub_historical_replays (
                    replay_id TEXT PRIMARY KEY,
                    run_id TEXT,
                    generated_at TIMESTAMPTZ,
                    trading_date TEXT,
                    symbol TEXT,
                    payload TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS data_hub_replay_reports (
                    report_id TEXT PRIMARY KEY,
                    generated_at TIMESTAMPTZ,
                    passed BOOLEAN,
                    payload TEXT
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def save_observation(
        self,
        observation: HistoricalReplayObservation,
    ) -> bool:
        conn = duckdb.connect(str(self.db_path))
        try:
            exists = conn.execute(
                "SELECT 1 FROM data_hub_historical_replays "
                "WHERE replay_id=?",
                [observation.replay_id],
            ).fetchone()
            if exists:
                return False
            conn.execute(
                "INSERT INTO data_hub_historical_replays "
                "VALUES (?,?,?,?,?,?)",
                [
                    observation.replay_id,
                    observation.run_id,
                    observation.generated_at,
                    observation.trading_date,
                    observation.symbol,
                    json.dumps(
                        observation.to_dict(),
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                ],
            )
            conn.commit()
            return True
        finally:
            conn.close()

    def save_report(self, report: Mapping[str, Any]) -> bool:
        conn = duckdb.connect(str(self.db_path))
        try:
            exists = conn.execute(
                "SELECT 1 FROM data_hub_replay_reports WHERE report_id=?",
                [report["report_id"]],
            ).fetchone()
            if exists:
                return False
            conn.execute(
                "INSERT INTO data_hub_replay_reports VALUES (?,?,?,?)",
                [
                    report["report_id"],
                    datetime.fromisoformat(str(report["generated_at"])),
                    bool(report["passed"]),
                    json.dumps(
                        dict(report),
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                ],
            )
            conn.commit()
            return True
        finally:
            conn.close()


@dataclass(frozen=True)
class HistoricalReplayResult:
    observations: tuple[HistoricalReplayObservation, ...]
    report: Mapping[str, Any]
    saved_observations: int
    saved_report: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": "HISTORICAL_CORRECTNESS_REPLAY",
            "saved_observations": self.saved_observations,
            "saved_report": self.saved_report,
            "report": dict(self.report),
        }


BarsLoader = Callable[[str, str], pd.DataFrame]
WindowLoader = Callable[
    [str, str, datetime, datetime],
    pd.DataFrame,
]


class HistoricalDataHubReplayRunner:
    def __init__(
        self,
        *,
        store: DataHubReplayStore,
        local_loader: BarsLoader = get_bars,
        alpaca_loader: WindowLoader = fetch_alpaca_bars_window,
        timeframe: str = "1d",
        required_days: int = 20,
        tolerance_bps: float = 1.0,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if timeframe != "1d":
            raise ValueError("DATA_HUB_REPLAY_DAILY_TIMEFRAME_REQUIRED")
        if required_days < 1:
            raise ValueError("DATA_HUB_REPLAY_DAYS_INVALID")
        if not math.isfinite(tolerance_bps) or tolerance_bps < 0:
            raise ValueError("DATA_HUB_REPLAY_TOLERANCE_INVALID")
        self.store = store
        self.local_loader = local_loader
        self.alpaca_loader = alpaca_loader
        self.timeframe = timeframe
        self.required_days = required_days
        self.tolerance_bps = tolerance_bps
        self.clock = clock or (lambda: datetime.now(_UTC))

    def run(self, symbols: Iterable[str]) -> HistoricalReplayResult:
        normalized = tuple(
            dict.fromkeys(
                symbol.strip().upper()
                for symbol in symbols
                if symbol.strip()
            )
        )
        if not normalized:
            raise ValueError("DATA_HUB_REPLAY_SYMBOLS_REQUIRED")
        generated_at = self._aware(self.clock())
        local_frames = {
            symbol: self._daily_rows(
                self.local_loader(symbol, self.timeframe)
            )
            for symbol in normalized
        }
        if any(frame.empty for frame in local_frames.values()):
            raise RuntimeError("DATA_HUB_REPLAY_LOCAL_HISTORY_MISSING")
        local_start = min(
            frame.tail(self.required_days + 10)["timestamp_utc"].min()
            for frame in local_frames.values()
        )
        local_end = max(
            frame["timestamp_utc"].max()
            for frame in local_frames.values()
        )
        start = local_start.to_pydatetime()
        end = min(
            generated_at,
            local_end.to_pydatetime() + timedelta(days=2),
        )
        alpaca_frames = {
            symbol: self._daily_rows(
                self.alpaca_loader(
                    symbol,
                    self.timeframe,
                    start,
                    end,
                )
            )
            for symbol in normalized
        }
        if any(frame.empty for frame in alpaca_frames.values()):
            raise RuntimeError("DATA_HUB_REPLAY_ALPACA_HISTORY_MISSING")

        common_dates: set[str] | None = None
        indexed: dict[tuple[str, str], dict[str, Mapping[str, float]]] = {}
        for symbol in normalized:
            local = self._index(local_frames[symbol])
            alpaca = self._index(alpaca_frames[symbol])
            indexed[(symbol, "local")] = local
            indexed[(symbol, "alpaca")] = alpaca
            symbol_dates = set(local) & set(alpaca)
            common_dates = (
                symbol_dates
                if common_dates is None
                else common_dates & symbol_dates
            )
        selected_dates = sorted(common_dates or ())[-self.required_days :]
        if len(selected_dates) < self.required_days:
            raise RuntimeError("DATA_HUB_REPLAY_COMMON_DAYS_INSUFFICIENT")

        run_id = self._fingerprint(
            {
                "generated_at": generated_at.isoformat(),
                "symbols": normalized,
                "timeframe": self.timeframe,
                "dates": selected_dates,
            },
            prefix="data-hub-replay-run-",
        )
        observations = tuple(
            self._compare(
                run_id=run_id,
                generated_at=generated_at,
                trading_date=trading_date,
                symbol=symbol,
                primary=indexed[(symbol, "local")][trading_date],
                shadow=indexed[(symbol, "alpaca")][trading_date],
            )
            for trading_date in selected_dates
            for symbol in normalized
        )
        saved_observations = sum(
            self.store.save_observation(observation)
            for observation in observations
        )
        live_report = generate_data_hub_quality_report(
            DataHubQualityStore(self.store.db_path).load_observations(),
            generated_at=generated_at,
        )
        report = self._report(
            observations,
            symbols=normalized,
            dates=selected_dates,
            generated_at=generated_at,
            live_report=live_report,
        )
        saved_report = self.store.save_report(report)
        return HistoricalReplayResult(
            observations=observations,
            report=report,
            saved_observations=saved_observations,
            saved_report=saved_report,
        )

    @staticmethod
    def _aware(value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("DATA_HUB_REPLAY_CLOCK_TIMEZONE_REQUIRED")
        return value.astimezone(_UTC)

    @staticmethod
    def _daily_rows(frame: pd.DataFrame) -> pd.DataFrame:
        required = {"timestamp_utc", *_VALUE_FIELDS}
        normalized = frame.copy()
        normalized.columns = [
            str(column).lower() for column in normalized.columns
        ]
        if normalized.empty or not required.issubset(normalized.columns):
            return pd.DataFrame()
        normalized["timestamp_utc"] = pd.to_datetime(
            normalized["timestamp_utc"],
            utc=True,
        )
        normalized = normalized.sort_values("timestamp_utc")
        normalized["trading_date"] = (
            normalized["timestamp_utc"]
            .dt.tz_convert(_ET)
            .dt.date.astype(str)
        )
        return normalized.drop_duplicates(
            subset=["trading_date"],
            keep="last",
        )

    @staticmethod
    def _index(
        frame: pd.DataFrame,
    ) -> dict[str, Mapping[str, float]]:
        return {
            str(row["trading_date"]): {
                field: float(row[field])
                for field in _VALUE_FIELDS
            }
            for row in frame.to_dict(orient="records")
        }

    def _compare(
        self,
        *,
        run_id: str,
        generated_at: datetime,
        trading_date: str,
        symbol: str,
        primary: Mapping[str, float],
        shadow: Mapping[str, float],
    ) -> HistoricalReplayObservation:
        differences = []
        for field in _VALUE_FIELDS:
            left = float(primary[field])
            right = float(shadow[field])
            absolute = abs(left - right)
            difference_bps = (
                absolute / abs(left) * 10_000
                if left
                else (0.0 if right == 0 else float("inf"))
            )
            if difference_bps > self.tolerance_bps:
                differences.append(
                    {
                        "field": field,
                        "primary": left,
                        "shadow": right,
                        "absolute_difference": absolute,
                        "difference_bps": difference_bps,
                    }
                )
        raw = {
            "run_id": run_id,
            "trading_date": trading_date,
            "symbol": symbol,
            "timeframe": self.timeframe,
            "primary": dict(primary),
            "shadow": dict(shadow),
        }
        return HistoricalReplayObservation(
            replay_id=self._fingerprint(
                raw,
                prefix="data-hub-replay-",
            ),
            run_id=run_id,
            generated_at=generated_at,
            trading_date=trading_date,
            symbol=symbol,
            timeframe=self.timeframe,
            primary_source="local_bar_cache",
            shadow_source="alpaca_market",
            matched=not differences,
            differences=tuple(differences),
            primary=dict(primary),
            shadow=dict(shadow),
        )

    def _report(
        self,
        observations: Sequence[HistoricalReplayObservation],
        *,
        symbols: Sequence[str],
        dates: Sequence[str],
        generated_at: datetime,
        live_report: Mapping[str, Any],
    ) -> dict[str, Any]:
        differences = [
            difference
            for observation in observations
            for difference in observation.differences
        ]
        expected = self.required_days * len(symbols)
        replay_passed = (
            len(dates) == self.required_days
            and len(observations) == expected
            and not differences
        )
        live_non_window_gates = {
            key: bool(value)
            for key, value in dict(live_report["gates"]).items()
            if key != "observation_window"
        }
        report = {
            "generated_at": generated_at.isoformat(),
            "evidence_scope": "HISTORICAL_DATA_CORRECTNESS_ONLY",
            "required_days": self.required_days,
            "observed_days": len(dates),
            "window_start": dates[0] if dates else None,
            "window_end": dates[-1] if dates else None,
            "symbols": list(symbols),
            "expected_comparisons": expected,
            "comparisons": len(observations),
            "mismatched_comparisons": sum(
                not observation.matched
                for observation in observations
            ),
            "differences": len(differences),
            "tolerance_bps": self.tolerance_bps,
            "passed": replay_passed,
            "live_observed_trading_days": live_report[
                "observed_trading_days"
            ],
            "live_non_window_gates": live_non_window_gates,
            "accelerated_shadow_delivery_ready": (
                replay_passed
                and bool(live_non_window_gates)
                and all(live_non_window_gates.values())
            ),
            "execution_cutover_ready": bool(live_report["passed"]),
            "can_replace_live_observation_window": False,
            "execution_input_switched": False,
        }
        report["report_id"] = self._fingerprint(
            report,
            prefix="data-hub-replay-report-",
        )
        return report

    @staticmethod
    def _fingerprint(
        payload: Mapping[str, Any],
        *,
        prefix: str,
    ) -> str:
        encoded = json.dumps(
            dict(payload),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return prefix + hashlib.sha256(encoded.encode()).hexdigest()[:20]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay historical local/Alpaca daily bars.",
    )
    parser.add_argument("--symbols", default="AAPL,MSFT")
    parser.add_argument("--days", type=int, default=20)
    parser.add_argument("--tolerance-bps", type=float, default=1.0)
    parser.add_argument("--db", default=settings.daily_research_db)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    symbols = [
        symbol.strip().upper()
        for symbol in args.symbols.split(",")
        if symbol.strip()
    ]
    result = HistoricalDataHubReplayRunner(
        store=DataHubReplayStore(args.db),
        required_days=args.days,
        tolerance_bps=args.tolerance_bps,
    ).run(symbols)
    print(json.dumps(result.to_dict(), sort_keys=True))
    return 0 if result.report["accelerated_shadow_delivery_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
