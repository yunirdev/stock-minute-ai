"""
selection.py
选股模块：对 universe 跑策略共识打分，输出 Candidate 列表。

基础版：对每个标的跑所有策略，统计当前 bar 的看多/看空票数，
        score = 100 * 多票/总票，reasons["votes"] 可追溯到具体策略票。
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List

from .contracts import Selector
from .data_cache import get_bars
from .models import Candidate, new_id, utc_now
from .strategy_core import STRATEGY_OPTIONS, compute_signals

logger = logging.getLogger(__name__)


class ConsensusSelector:
    """实现 Selector Protocol —— 策略共识打分选股。"""

    def __init__(
        self,
        strategies: List[str] | None = None,
        min_bars: int = 60,
    ) -> None:
        self._strategies = strategies or list(STRATEGY_OPTIONS.keys())
        self._min_bars = min_bars

    def select(
        self,
        universe: List[str],
        timeframe: str,
        as_of: datetime,
    ) -> List[Candidate]:
        candidates: List[Candidate] = []
        for symbol in universe:
            try:
                cand = self._score(symbol, timeframe, as_of)
                if cand is not None:
                    candidates.append(cand)
            except Exception as exc:
                logger.warning("selection 跳过 %s: %s", symbol, exc)

        candidates.sort(key=lambda c: c.score, reverse=True)
        for i, c in enumerate(candidates):
            c.rank = i + 1
        return candidates

    def _score(
        self,
        symbol: str,
        timeframe: str,
        as_of: datetime,
    ) -> Candidate | None:
        df = get_bars(symbol, timeframe)
        if df is None or len(df) < self._min_bars:
            return None

        votes: Dict[str, int] = {}
        for strat in self._strategies:
            try:
                result = compute_signals(df.copy(), strat)
                sig = int(result.iloc[-1].get("strat_signal", 0))
                if sig != 0:
                    votes[strat] = sig  # +1 看多, -1 看空
            except Exception:
                pass

        total = len(self._strategies)
        bull = sum(1 for v in votes.values() if v > 0)
        score = round(100.0 * bull / total, 1) if total > 0 else 50.0

        return Candidate(
            symbol=symbol,
            score=score,
            rank=0,  # filled after sort
            reasons={"votes": votes, "total_strategies": total},
            as_of=as_of or utc_now(),
        )


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_default_selector: ConsensusSelector | None = None


def get_selector() -> ConsensusSelector:
    global _default_selector
    if _default_selector is None:
        _default_selector = ConsensusSelector()
    return _default_selector
