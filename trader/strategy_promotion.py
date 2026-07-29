"""Quantitative candidate validation and append-only promotion audit."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from .engine import SimResult, simulate
from .strategy_candidates import StrategyCandidateStore
from .strategy_core import compute_signals

_EVIDENCE_KINDS = {"HOLDOUT", "HISTORICAL_REPLAY", "PAPER"}
_BARS_PER_YEAR = {
    "1m": 252 * 390,
    "5m": 252 * 78,
    "15m": 252 * 26,
    "30m": 252 * 13,
    "1h": 252 * 7,
    "1d": 252,
}


def _require_aware(value: datetime, code: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(code)


def _require_text(value: str, code: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(code)
    return normalized


@dataclass(frozen=True)
class StrategyMetrics:
    net_return: float
    sharpe: float
    max_drawdown: float
    trade_count: int
    win_rate: float
    fees: float
    slippage_cost: float
    bar_count: int
    session_count: int

    def __post_init__(self) -> None:
        numeric = (
            self.net_return,
            self.sharpe,
            self.max_drawdown,
            self.win_rate,
            self.fees,
            self.slippage_cost,
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("STRATEGY_METRICS_NON_FINITE")
        if not 0 <= self.max_drawdown <= 1:
            raise ValueError("STRATEGY_METRICS_DRAWDOWN_INVALID")
        if not 0 <= self.win_rate <= 1:
            raise ValueError("STRATEGY_METRICS_WIN_RATE_INVALID")
        if min(
            self.trade_count,
            self.bar_count,
            self.session_count,
        ) < 0:
            raise ValueError("STRATEGY_METRICS_COUNT_INVALID")
        if self.fees < 0 or self.slippage_cost < 0:
            raise ValueError("STRATEGY_METRICS_COST_INVALID")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> StrategyMetrics:
        return cls(**value)


@dataclass(frozen=True)
class PromotionPolicy:
    min_holdout_bars: int = 60
    min_holdout_trades: int = 5
    min_replay_bars: int = 100
    min_replay_trades: int = 5
    min_paper_sessions: int = 5
    min_paper_trades: int = 3
    min_net_return_advantage: float = 0.0
    min_sharpe: float = 0.0
    max_drawdown: float = 0.25
    max_drawdown_worsening: float = 0.02
    min_fee_bps: float = 1.0
    min_slippage_bps: float = 1.0

    def __post_init__(self) -> None:
        counts = (
            self.min_holdout_bars,
            self.min_holdout_trades,
            self.min_replay_bars,
            self.min_replay_trades,
            self.min_paper_sessions,
            self.min_paper_trades,
        )
        if min(counts) < 1:
            raise ValueError("PROMOTION_POLICY_MINIMUM_SAMPLE_INVALID")
        numeric = (
            self.min_net_return_advantage,
            self.min_sharpe,
            self.max_drawdown,
            self.max_drawdown_worsening,
            self.min_fee_bps,
            self.min_slippage_bps,
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("PROMOTION_POLICY_NON_FINITE")
        if not 0 <= self.max_drawdown <= 1:
            raise ValueError("PROMOTION_POLICY_DRAWDOWN_INVALID")
        if self.max_drawdown_worsening < 0:
            raise ValueError("PROMOTION_POLICY_DRAWDOWN_WORSENING_INVALID")
        if self.min_net_return_advantage < 0:
            raise ValueError("PROMOTION_POLICY_RETURN_ADVANTAGE_INVALID")
        if self.min_fee_bps < 0 or self.min_slippage_bps < 0:
            raise ValueError("PROMOTION_POLICY_COST_INVALID")


class StrategyPromotionStore:
    """Frozen comparisons and release events; never edits Runtime configuration."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self.candidates = StrategyCandidateStore(db_path)
        self._migrate()

    def _connect(self, *, read_only: bool = False):
        return duckdb.connect(self.db_path, read_only=read_only)

    def _migrate(self) -> None:
        connection = self._connect()
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS strategy_candidate_comparisons (
                    comparison_id TEXT PRIMARY KEY,
                    candidate_id TEXT,
                    evidence_kind TEXT,
                    evidence_version TEXT,
                    dataset_version TEXT,
                    window_start TIMESTAMPTZ,
                    window_end TIMESTAMPTZ,
                    champion_version TEXT,
                    candidate_metrics_json TEXT,
                    champion_metrics_json TEXT,
                    fee_bps DOUBLE,
                    slippage_bps DOUBLE,
                    created_at TIMESTAMPTZ
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS strategy_release_events (
                    event_id TEXT PRIMARY KEY,
                    strategy_name TEXT,
                    candidate_id TEXT,
                    event_type TEXT,
                    from_version TEXT,
                    to_version TEXT,
                    rollback_version TEXT,
                    evidence_ids_json TEXT,
                    reasons_json TEXT,
                    policy_json TEXT,
                    created_at TIMESTAMPTZ
                )
                """
            )
            connection.commit()
        finally:
            connection.close()

    def evaluate_bars(
        self,
        *,
        candidate_id: str,
        evidence_kind: str,
        evidence_version: str,
        dataset_version: str,
        bars: pd.DataFrame,
        timeframe: str,
        champion_parameters: dict[str, Any],
        fee_bps: float,
        slippage_bps: float,
        created_at: datetime,
    ) -> dict[str, Any]:
        """Run candidate and champion on the same frozen bars and cost model."""
        kind = evidence_kind.strip().upper()
        if kind not in {"HOLDOUT", "HISTORICAL_REPLAY"}:
            raise ValueError("STRATEGY_EVALUATION_BAR_KIND_INVALID")
        required = {
            "timestamp_utc",
            "open",
            "high",
            "low",
            "close",
            "volume",
        }
        if not isinstance(bars, pd.DataFrame) or not required.issubset(bars.columns):
            raise ValueError("STRATEGY_EVALUATION_BARS_INVALID")
        frame = bars.copy()
        frame["timestamp_utc"] = pd.to_datetime(
            frame["timestamp_utc"],
            utc=True,
            errors="coerce",
        )
        if (
            frame.empty
            or frame["timestamp_utc"].isna().any()
            or frame["timestamp_utc"].duplicated().any()
            or not frame["timestamp_utc"].is_monotonic_increasing
        ):
            raise ValueError("STRATEGY_EVALUATION_BAR_TIME_INVALID")
        candidate = self._candidate(candidate_id)
        candidate_result = simulate(
            compute_signals(
                frame,
                candidate["strategy_name"],
                **candidate["parameters"],
            ),
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
        )
        champion_result = simulate(
            compute_signals(
                frame,
                candidate["strategy_name"],
                **champion_parameters,
            ),
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
        )
        return self.record_comparison(
            candidate_id=candidate_id,
            evidence_kind=kind,
            evidence_version=evidence_version,
            dataset_version=dataset_version,
            window_start=frame["timestamp_utc"].iloc[0].to_pydatetime(),
            window_end=frame["timestamp_utc"].iloc[-1].to_pydatetime(),
            champion_version=candidate["base_strategy_version"],
            candidate_metrics=self._metrics(
                candidate_result,
                frame,
                timeframe=timeframe,
                slippage_bps=slippage_bps,
            ),
            champion_metrics=self._metrics(
                champion_result,
                frame,
                timeframe=timeframe,
                slippage_bps=slippage_bps,
            ),
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            created_at=created_at,
        )

    def record_comparison(
        self,
        *,
        candidate_id: str,
        evidence_kind: str,
        evidence_version: str,
        dataset_version: str,
        window_start: datetime,
        window_end: datetime,
        champion_version: str,
        candidate_metrics: StrategyMetrics,
        champion_metrics: StrategyMetrics,
        fee_bps: float,
        slippage_bps: float,
        created_at: datetime,
    ) -> dict[str, Any]:
        """Record frozen externally computed or locally replayed evidence."""
        candidate = self._candidate(candidate_id)
        kind = evidence_kind.strip().upper()
        if kind not in _EVIDENCE_KINDS:
            raise ValueError("STRATEGY_EVALUATION_KIND_INVALID")
        evidence = _require_text(
            evidence_version,
            "STRATEGY_EVALUATION_VERSION_REQUIRED",
        )
        dataset = _require_text(
            dataset_version,
            "STRATEGY_EVALUATION_DATASET_REQUIRED",
        )
        champion = _require_text(
            champion_version,
            "STRATEGY_EVALUATION_CHAMPION_REQUIRED",
        )
        if champion != candidate["base_strategy_version"]:
            raise ValueError("STRATEGY_EVALUATION_CHAMPION_MISMATCH")
        for value, code in (
            (window_start, "STRATEGY_EVALUATION_START_TZ_REQUIRED"),
            (window_end, "STRATEGY_EVALUATION_END_TZ_REQUIRED"),
            (created_at, "STRATEGY_EVALUATION_CREATED_TZ_REQUIRED"),
        ):
            _require_aware(value, code)
        if window_start >= window_end:
            raise ValueError("STRATEGY_EVALUATION_WINDOW_INVALID")
        if created_at < window_end:
            raise ValueError("STRATEGY_EVALUATION_FROM_FUTURE")
        if created_at < candidate["created_at"]:
            raise ValueError("STRATEGY_EVALUATION_BEFORE_CANDIDATE")
        boundary = candidate["boundary"]
        if kind == "HOLDOUT":
            if dataset != candidate["dataset_version"]:
                raise ValueError("STRATEGY_HOLDOUT_DATASET_MISMATCH")
            if (
                window_start < boundary["holdout_start"]
                or window_end > boundary["holdout_end"]
            ):
                raise ValueError("STRATEGY_HOLDOUT_WINDOW_INVALID")
        elif kind == "HISTORICAL_REPLAY":
            overlaps_holdout = (
                window_start <= boundary["holdout_end"]
                and window_end >= boundary["holdout_start"]
            )
            if overlaps_holdout:
                raise ValueError("STRATEGY_REPLAY_HOLDOUT_OVERLAP")
        elif window_start < candidate["created_at"]:
            raise ValueError("STRATEGY_PAPER_BEFORE_CANDIDATE")
        fee = float(fee_bps)
        slippage = float(slippage_bps)
        if (
            not math.isfinite(fee)
            or not math.isfinite(slippage)
            or fee < 0
            or slippage < 0
        ):
            raise ValueError("STRATEGY_EVALUATION_COST_INVALID")
        payload = {
            "candidate_id": candidate_id,
            "evidence_kind": kind,
            "evidence_version": evidence,
            "dataset_version": dataset,
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
            "champion_version": champion,
            "candidate_metrics": asdict(candidate_metrics),
            "champion_metrics": asdict(champion_metrics),
            "fee_bps": fee,
            "slippage_bps": slippage,
        }
        comparison_id = "strategy-comparison-" + self._digest(payload, 24)
        connection = self._connect()
        try:
            connection.execute(
                """
                INSERT INTO strategy_candidate_comparisons VALUES
                (?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT (comparison_id) DO NOTHING
                """,
                [
                    comparison_id,
                    candidate_id,
                    kind,
                    evidence,
                    dataset,
                    window_start,
                    window_end,
                    champion,
                    json.dumps(asdict(candidate_metrics), sort_keys=True),
                    json.dumps(asdict(champion_metrics), sort_keys=True),
                    fee,
                    slippage,
                    created_at,
                ],
            )
            connection.commit()
        finally:
            connection.close()
        record = self.get_comparison(comparison_id)
        if record is None:  # pragma: no cover
            raise RuntimeError("STRATEGY_EVALUATION_PERSIST_FAILED")
        return record

    def decide(
        self,
        candidate_id: str,
        *,
        policy: PromotionPolicy,
        created_at: datetime,
    ) -> dict[str, Any]:
        """Promote or reject from the latest evidence of every required kind."""
        _require_aware(created_at, "STRATEGY_PROMOTION_TIME_TZ_REQUIRED")
        candidate = self._candidate(candidate_id)
        current = self.current_champion(candidate["strategy_name"])
        if current is None:
            current = candidate["base_strategy_version"]
        if current == candidate["candidate_version"]:
            promoted = self._latest_promotion(candidate_id)
            if promoted is not None:
                return promoted
        latest_state = self._latest_state_event(candidate["strategy_name"])
        state_context = latest_state["event_id"] if latest_state else ""
        comparisons = {
            kind: self._latest_comparison(candidate_id, kind)
            for kind in sorted(_EVIDENCE_KINDS)
        }
        reasons: list[str] = []
        if current != candidate["base_strategy_version"]:
            reasons.append("STALE_BASE_VERSION")
        for kind, comparison in comparisons.items():
            if comparison is None:
                reasons.append(f"MISSING_{kind}")
                continue
            reasons.extend(self._comparison_reasons(kind, comparison, policy))
        event_type = "PROMOTED" if not reasons else "REJECTED"
        to_version = (
            candidate["candidate_version"]
            if event_type == "PROMOTED"
            else current
        )
        evidence_ids = sorted(
            comparison["comparison_id"]
            for comparison in comparisons.values()
            if comparison is not None
        )
        payload = {
            "strategy_name": candidate["strategy_name"],
            "candidate_id": candidate_id,
            "event_type": event_type,
            "from_version": current,
            "to_version": to_version,
            "rollback_version": current,
            "evidence_ids": evidence_ids,
            "reasons": sorted(reasons),
            "policy": asdict(policy),
            "state_context": state_context,
        }
        event_id = "strategy-release-" + self._digest(payload, 24)
        connection = self._connect()
        try:
            connection.execute(
                """
                INSERT INTO strategy_release_events VALUES
                (?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT (event_id) DO NOTHING
                """,
                [
                    event_id,
                    candidate["strategy_name"],
                    candidate_id,
                    event_type,
                    current,
                    to_version,
                    current,
                    json.dumps(evidence_ids),
                    json.dumps(sorted(reasons)),
                    json.dumps(asdict(policy), sort_keys=True),
                    created_at,
                ],
            )
            connection.commit()
        finally:
            connection.close()
        event = self.get_event(event_id)
        if event is None:  # pragma: no cover
            raise RuntimeError("STRATEGY_PROMOTION_PERSIST_FAILED")
        return event

    def rollback(
        self,
        promotion_event_id: str,
        *,
        reason: str,
        created_at: datetime,
    ) -> dict[str, Any]:
        """Append a rollback to the version saved by one promotion."""
        _require_aware(created_at, "STRATEGY_ROLLBACK_TIME_TZ_REQUIRED")
        normalized_reason = _require_text(
            reason,
            "STRATEGY_ROLLBACK_REASON_REQUIRED",
        )
        promotion = self.get_event(promotion_event_id)
        if promotion is None or promotion["event_type"] != "PROMOTED":
            raise ValueError("STRATEGY_ROLLBACK_PROMOTION_NOT_FOUND")
        payload = {
            "promotion_event_id": promotion_event_id,
            "strategy_name": promotion["strategy_name"],
            "candidate_id": promotion["candidate_id"],
            "from_version": promotion["to_version"],
            "to_version": promotion["rollback_version"],
            "reason": normalized_reason,
        }
        event_id = "strategy-rollback-" + self._digest(payload, 24)
        existing = self.get_event(event_id)
        if existing is not None:
            return existing
        if created_at <= promotion["created_at"]:
            raise ValueError("STRATEGY_ROLLBACK_TIME_INVALID")
        current = self.current_champion(promotion["strategy_name"])
        if current != promotion["to_version"]:
            raise ValueError("STRATEGY_ROLLBACK_STALE_PROMOTION")
        connection = self._connect()
        try:
            connection.execute(
                """
                INSERT INTO strategy_release_events VALUES
                (?,?,?,?,?,?,?,?,?,?,?)
                """,
                [
                    event_id,
                    promotion["strategy_name"],
                    promotion["candidate_id"],
                    "ROLLED_BACK",
                    promotion["to_version"],
                    promotion["rollback_version"],
                    promotion["rollback_version"],
                    json.dumps([promotion_event_id]),
                    json.dumps([normalized_reason]),
                    json.dumps({}),
                    created_at,
                ],
            )
            connection.commit()
        finally:
            connection.close()
        event = self.get_event(event_id)
        if event is None:  # pragma: no cover
            raise RuntimeError("STRATEGY_ROLLBACK_PERSIST_FAILED")
        return event

    def current_champion(self, strategy_name: str) -> str | None:
        state = self._latest_state_event(strategy_name)
        return state["to_version"] if state is not None else None

    def get_comparison(self, comparison_id: str) -> dict[str, Any] | None:
        connection = self._connect(read_only=True)
        try:
            row = connection.execute(
                """
                SELECT * FROM strategy_candidate_comparisons
                WHERE comparison_id=?
                """,
                [comparison_id],
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        return {
            "comparison_id": str(row[0]),
            "candidate_id": str(row[1]),
            "evidence_kind": str(row[2]),
            "evidence_version": str(row[3]),
            "dataset_version": str(row[4]),
            "window_start": row[5],
            "window_end": row[6],
            "champion_version": str(row[7]),
            "candidate_metrics": StrategyMetrics.from_dict(json.loads(row[8])),
            "champion_metrics": StrategyMetrics.from_dict(json.loads(row[9])),
            "fee_bps": float(row[10]),
            "slippage_bps": float(row[11]),
            "created_at": row[12],
        }

    def get_event(self, event_id: str) -> dict[str, Any] | None:
        connection = self._connect(read_only=True)
        try:
            row = connection.execute(
                """
                SELECT * FROM strategy_release_events
                WHERE event_id=?
                """,
                [event_id],
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        return {
            "event_id": str(row[0]),
            "strategy_name": str(row[1]),
            "candidate_id": str(row[2]),
            "event_type": str(row[3]),
            "from_version": str(row[4]),
            "to_version": str(row[5]),
            "rollback_version": str(row[6]),
            "evidence_ids": json.loads(row[7]),
            "reasons": json.loads(row[8]),
            "policy": json.loads(row[9]),
            "created_at": row[10],
        }

    def _candidate(self, candidate_id: str) -> dict[str, Any]:
        candidate = self.candidates.get(candidate_id)
        if candidate is None:
            raise ValueError("STRATEGY_PROMOTION_CANDIDATE_NOT_FOUND")
        return candidate

    def _latest_comparison(
        self,
        candidate_id: str,
        evidence_kind: str,
    ) -> dict[str, Any] | None:
        connection = self._connect(read_only=True)
        try:
            row = connection.execute(
                """
                SELECT comparison_id FROM strategy_candidate_comparisons
                WHERE candidate_id=? AND evidence_kind=?
                ORDER BY created_at DESC, comparison_id DESC
                LIMIT 1
                """,
                [candidate_id, evidence_kind],
            ).fetchone()
        finally:
            connection.close()
        return self.get_comparison(str(row[0])) if row is not None else None

    def _latest_state_event(self, strategy_name: str) -> dict[str, Any] | None:
        connection = self._connect(read_only=True)
        try:
            row = connection.execute(
                """
                SELECT event_id FROM strategy_release_events
                WHERE strategy_name=?
                  AND event_type IN ('PROMOTED', 'ROLLED_BACK')
                ORDER BY created_at DESC, event_id DESC
                LIMIT 1
                """,
                [strategy_name],
            ).fetchone()
        finally:
            connection.close()
        return self.get_event(str(row[0])) if row is not None else None

    def _latest_promotion(self, candidate_id: str) -> dict[str, Any] | None:
        connection = self._connect(read_only=True)
        try:
            row = connection.execute(
                """
                SELECT event_id FROM strategy_release_events
                WHERE candidate_id=? AND event_type='PROMOTED'
                ORDER BY created_at DESC, event_id DESC
                LIMIT 1
                """,
                [candidate_id],
            ).fetchone()
        finally:
            connection.close()
        return self.get_event(str(row[0])) if row is not None else None

    @staticmethod
    def _comparison_reasons(
        kind: str,
        comparison: dict[str, Any],
        policy: PromotionPolicy,
    ) -> list[str]:
        candidate = comparison["candidate_metrics"]
        champion = comparison["champion_metrics"]
        reasons = []
        min_bars = (
            policy.min_holdout_bars
            if kind == "HOLDOUT"
            else policy.min_replay_bars
        )
        min_trades = {
            "HOLDOUT": policy.min_holdout_trades,
            "HISTORICAL_REPLAY": policy.min_replay_trades,
            "PAPER": policy.min_paper_trades,
        }[kind]
        if kind != "PAPER" and candidate.bar_count < min_bars:
            reasons.append(f"{kind}_MIN_BARS")
        if kind == "PAPER" and candidate.session_count < policy.min_paper_sessions:
            reasons.append("PAPER_MIN_SESSIONS")
        if candidate.trade_count < min_trades:
            reasons.append(f"{kind}_MIN_TRADES")
        if comparison["fee_bps"] < policy.min_fee_bps:
            reasons.append(f"{kind}_FEE_MODEL_INSUFFICIENT")
        if comparison["slippage_bps"] < policy.min_slippage_bps:
            reasons.append(f"{kind}_SLIPPAGE_MODEL_INSUFFICIENT")
        if (
            candidate.net_return
            < champion.net_return + policy.min_net_return_advantage
        ):
            reasons.append(f"{kind}_RETURN_ADVANTAGE")
        if candidate.sharpe < policy.min_sharpe:
            reasons.append(f"{kind}_MIN_SHARPE")
        if candidate.sharpe < champion.sharpe:
            reasons.append(f"{kind}_SHARPE_VS_CHAMPION")
        if candidate.max_drawdown > policy.max_drawdown:
            reasons.append(f"{kind}_MAX_DRAWDOWN")
        if (
            candidate.max_drawdown
            > champion.max_drawdown + policy.max_drawdown_worsening
        ):
            reasons.append(f"{kind}_DRAWDOWN_VS_CHAMPION")
        return reasons

    @staticmethod
    def _metrics(
        result: SimResult,
        bars: pd.DataFrame,
        *,
        timeframe: str,
        slippage_bps: float,
    ) -> StrategyMetrics:
        returns = (
            result.equity_curve.pct_change()
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        std = float(returns.std()) if len(returns) > 1 else 0.0
        sharpe = (
            float(
                returns.mean()
                / std
                * np.sqrt(_BARS_PER_YEAR.get(timeframe, 252))
            )
            if std > 0
            else 0.0
        )
        peak = result.equity_curve.cummax()
        drawdown = (
            (peak - result.equity_curve)
            / peak.replace(0, np.nan)
        ).fillna(0)
        closed = [trade for trade in result.trades if trade.side == "SELL"]
        slippage_rate = max(float(slippage_bps), 0.0) / 10_000.0
        return StrategyMetrics(
            net_return=result.total_return,
            sharpe=sharpe,
            max_drawdown=float(drawdown.max()) if len(drawdown) else 0.0,
            trade_count=len(closed),
            win_rate=(
                sum(trade.ret > 0 for trade in closed) / len(closed)
                if closed
                else 0.0
            ),
            fees=float(sum(trade.fee for trade in result.trades)),
            slippage_cost=float(
                sum(
                    trade.qty * trade.price * slippage_rate
                    for trade in result.trades
                )
            ),
            bar_count=len(bars),
            session_count=int(
                pd.to_datetime(
                    bars["timestamp_utc"],
                    utc=True,
                ).dt.date.nunique()
            ),
        )

    @staticmethod
    def _digest(payload: dict[str, Any], length: int) -> str:
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(canonical.encode()).hexdigest()[:length]
