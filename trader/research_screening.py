"""Deterministic broad-screen enrichment for the daily research batch."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable

from .daily_candidates import DailyCandidate, build_daily_candidates
from .paper_decision import StrategyStatistics, StrategyStatisticsRepository


def build_research_candidates(
    universe: Iterable[str],
    *,
    timeframe: str,
    strategy_statistics_path: str = "",
    market_regime: str = "",
    limit: int = 10,
    now: datetime | None = None,
    input_capture: dict[str, Any] | None = None,
) -> list[DailyCandidate]:
    """Rank without AI, adding reliable holdout evidence when available."""
    now = now or datetime.now(timezone.utc)
    bars_capture = (
        input_capture.setdefault("bars", {})
        if input_capture is not None
        else None
    )
    rows = build_daily_candidates(
        universe,
        timeframe=timeframe,
        ai_db_path=None,
        limit=max(1, int(limit)),
        include_anchors=False,
        input_capture=bars_capture,
        now=now,
    )
    repository = StrategyStatisticsRepository.from_json(strategy_statistics_path)
    if input_capture is not None:
        input_capture["strategy_statistics"] = repository.records
    for row in rows:
        records = _matching_statistics(
            repository,
            row.symbol,
            timeframe,
            market_regime,
            now,
        )
        if not records:
            row.risk_flags = _unique(
                [*row.risk_flags, "缺少当前环境的可靠 holdout 统计"]
            )
            if row.data_confidence == "高":
                row.data_confidence = "中"
            continue
        best = max(
            records,
            key=lambda item: (
                item.out_of_sample_net_return,
                item.sharpe,
                -item.max_drawdown,
                item.trade_count,
            ),
        )
        holdout = _holdout_score(best)
        row.score = round(row.score * 0.70 + holdout * 0.30, 1)
        row.reasons = _unique(
            [
                *row.reasons,
                (
                    f"Holdout {best.strategy}: "
                    f"净收益 {best.out_of_sample_net_return:+.1%}, "
                    f"Sharpe {best.sharpe:.2f}"
                ),
            ]
        )
        row.status = _status(row.score)
    rows.sort(key=lambda item: (-item.score, item.symbol))
    for rank, row in enumerate(rows, start=1):
        row.rank = rank
    return rows[: max(1, int(limit))]


def _matching_statistics(
    repository: StrategyStatisticsRepository,
    symbol: str,
    timeframe: str,
    market_regime: str,
    now: datetime,
) -> list[StrategyStatistics]:
    records = [
        record
        for record in repository.records
        if record.symbol == symbol
        and record.timeframe == timeframe
        and record.reliable(now)
    ]
    if market_regime:
        exact = [
            record for record in records if record.market_regime == market_regime
        ]
        if exact:
            return exact
    return records


def _holdout_score(record: StrategyStatistics) -> float:
    score = (
        50.0
        + record.out_of_sample_net_return * 100.0
        + record.sharpe * 8.0
        - record.max_drawdown * 50.0
        + (record.win_rate - 0.5) * 40.0
    )
    return max(0.0, min(100.0, score))


def _status(score: float) -> str:
    if score >= 75:
        return "ENTRY_READY"
    if score >= 68:
        return "WAIT_BREAKOUT"
    if score >= 60:
        return "WATCH"
    if score >= 50:
        return "BENCH"
    return "AVOID_NOW"


def _unique(values: list[str]) -> list[str]:
    result = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result
