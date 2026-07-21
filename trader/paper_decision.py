"""Deterministic decision service for Alpaca Paper and shadow evaluation."""
from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .ai.safety import AIScorePolicy, AIScoreSnapshot, AIScoreValidator
from .models import Side


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("UTC_TIMESTAMP_REQUIRED")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class UniverseSnapshot:
    symbols: tuple[str, ...]
    source: str
    universe_version: str
    generated_at: datetime
    valid_until: datetime


class UniverseProvider:
    def __init__(self, allowed_symbols: Iterable[str], max_symbols: int = 20, max_pool_age_minutes: int = 1440) -> None:
        self.allowed = {str(s).strip().upper() for s in allowed_symbols if str(s).strip()}
        self.max_symbols = max_symbols
        self.max_age = timedelta(minutes=max_pool_age_minutes)

    def provide(
        self,
        *,
        cli_symbols: Iterable[str] = (),
        daily_pool: Any = None,
        manual_whitelist: Iterable[str] = (),
        now: datetime,
    ) -> UniverseSnapshot:
        now = _utc(now)
        source, generated = "cli", now
        symbols = list(cli_symbols)
        if daily_pool is not None:
            source = "daily_decision"
            raw_time = getattr(daily_pool, "updated_at", None) or daily_pool.get("updated_at")
            generated = datetime.fromisoformat(str(raw_time).replace("Z", "+00:00"))
            if now - _utc(generated) > self.max_age:
                raise ValueError("UNIVERSE_STALE")
            items = getattr(daily_pool, "items", None) or daily_pool.get("items", [])
            symbols = [getattr(item, "symbol", None) or item.get("symbol") for item in items]
        symbols.extend(manual_whitelist)
        clean = tuple(dict.fromkeys(str(s).strip().upper() for s in symbols if s and str(s).strip().upper() in self.allowed))[: self.max_symbols]
        raw = json.dumps({"symbols": clean, "source": source, "generated": _utc(generated).isoformat()}, sort_keys=True)
        version = hashlib.sha256(raw.encode()).hexdigest()[:20]
        return UniverseSnapshot(clean, source, version, _utc(generated), _utc(generated) + self.max_age)


@dataclass(frozen=True)
class StrategyStatistics:
    statistics_id: str
    symbol: str
    strategy: str
    strategy_version: str
    timeframe: str
    market_regime: str
    out_of_sample_net_return: float
    sharpe: float
    max_drawdown: float
    trade_count: int
    win_rate: float
    average_trade_return: float
    fees: float
    slippage: float
    data_start: datetime
    data_end: datetime
    evaluated_at: datetime
    statistics_version: str
    params: dict[str, Any] = field(default_factory=dict)

    def reliable(self, now: datetime, min_trades: int = 30, max_age_days: int = 90) -> bool:
        return (
            self.trade_count >= min_trades
            and self.data_end > self.data_start
            and now - _utc(self.evaluated_at) <= timedelta(days=max_age_days)
            and self.max_drawdown <= 1
        )


class StrategyStatisticsRepository:
    def __init__(self, records: Iterable[StrategyStatistics] = ()) -> None:
        self.records = tuple(records)

    def find(self, symbol: str, timeframe: str, market_regime: str, now: datetime) -> list[StrategyStatistics]:
        return [r for r in self.records if r.symbol == symbol and r.timeframe == timeframe and r.market_regime == market_regime and r.reliable(now)]

    @classmethod
    def from_json(cls, path: str) -> "StrategyStatisticsRepository":
        src = Path(path)
        if not path or not src.exists():
            return cls()
        records = []
        for row in json.loads(src.read_text(encoding="utf-8")):
            for key in ("data_start", "data_end", "evaluated_at"):
                row[key] = datetime.fromisoformat(row[key].replace("Z", "+00:00"))
            records.append(StrategyStatistics(**row))
        return cls(records)


@dataclass(frozen=True)
class StrategyDecision:
    decision_id: str
    symbol: str
    strategy: str
    strategy_version: str
    params: dict[str, Any]
    side: Side
    confidence: float
    target_weight: float
    valid_from: datetime
    valid_until: datetime
    market_regime: str
    candidate_source: str
    evidence: dict[str, Any]
    ai_advisory_run_id: str | None
    strategy_statistics_id: str
    data_version: str
    created_at: datetime
    reason_codes: tuple[str, ...]
    rejected_alternatives: tuple[dict[str, str], ...]
    universe_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["side"] = self.side.value
        for key in ("valid_from", "valid_until", "created_at"):
            value[key] = value[key].isoformat()
        return value


class PaperDecisionService:
    def __init__(self, *, allow_without_ai: bool = False, ai_max_age_minutes: int = 30, decision_ttl_minutes: int = 15) -> None:
        self.allow_without_ai = allow_without_ai
        self.ai_max_age = ai_max_age_minutes
        self.ttl = decision_ttl_minutes

    def decide(
        self,
        bars: Mapping[str, Any],
        positions: Mapping[str, Any],
        candidates: Iterable[Any],
        strategy_statistics: StrategyStatisticsRepository | Iterable[StrategyStatistics],
        ai_advisories: Mapping[str, AIScoreSnapshot],
        market_regime: str,
        now: datetime,
        *,
        timeframe: str = "5m",
        universe_version: str = "",
        data_version: str = "",
    ) -> list[StrategyDecision]:
        del positions  # explicit broker input; sizing remains downstream
        now = _utc(now)
        repo = strategy_statistics if isinstance(strategy_statistics, StrategyStatisticsRepository) else StrategyStatisticsRepository(strategy_statistics)
        decisions = []
        for candidate in candidates:
            symbol = getattr(candidate, "symbol", None) or candidate.get("symbol")
            score = float(getattr(candidate, "score", 0) if not isinstance(candidate, dict) else candidate.get("score", 0))
            stats = repo.find(symbol, timeframe, market_regime, now)
            if not stats or symbol not in bars:
                continue
            stats.sort(key=lambda r: (-r.out_of_sample_net_return, -r.sharpe, r.max_drawdown, -r.trade_count, r.strategy))
            chosen, alternatives = stats[0], stats[1:]
            advisory = ai_advisories.get(symbol)
            ai_result = AIScoreValidator(AIScorePolicy(0, self.ai_max_age), lambda: now).validate(advisory)
            if not ai_result.valid and not self.allow_without_ai:
                continue
            evidence = {}
            reasons = ["STRATEGY_STATISTICS_VALID", "REGIME_MATCH"]
            run_id = None
            if ai_result.valid and advisory is not None:
                evidence["ai"] = {"score": advisory.score, "provider": advisory.provider, "model": advisory.model}
                run_id = advisory.run_id
                reasons.append("AI_EVIDENCE_VALID")
            else:
                reasons.append("AI_NOT_USED")
            payload = {"symbol": symbol, "strategy": chosen.strategy, "stats": chosen.statistics_id, "regime": market_regime, "now": now.isoformat(), "universe": universe_version, "data": data_version}
            decision_id = "dec-" + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]
            confidence = max(0.0, min(1.0, 0.5 + chosen.sharpe / 10 - chosen.max_drawdown / 2))
            decisions.append(StrategyDecision(
                decision_id, symbol, chosen.strategy, chosen.strategy_version, dict(chosen.params),
                Side.BUY if score >= 50 else Side.SELL, confidence, 0.0, now,
                now + timedelta(minutes=self.ttl), market_regime,
                getattr(candidate, "source", None) or (candidate.get("source") if isinstance(candidate, dict) else None) or "runtime",
                evidence, run_id, chosen.statistics_id, data_version, now, tuple(reasons),
                tuple({"strategy": r.strategy, "reason_code": "LOWER_VALIDATED_NET_RETURN"} for r in alternatives),
                universe_version,
            ))
        return decisions


class AdvisoryWorker:
    """Single-flight non-blocking AgentManager runner; UI is not involved."""
    def __init__(self, manager: Any, timeout_seconds: int = 900) -> None:
        self.manager, self.timeout = manager, timeout_seconds
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="advisory")
        self._future: Future | None = None
        self._started = 0.0

    def start(self, context: Any, db_path: str) -> bool:
        if self._future and not self._future.done():
            return False
        self._started = time.monotonic()
        self._future = self._pool.submit(self.manager.run_cycle, context, db_path)
        return True

    def poll(self) -> Any | None:
        if not self._future:
            return None
        if self._future.done():
            return self._future.result()
        if time.monotonic() - self._started > self.timeout:
            self._future.cancel()
        return None
