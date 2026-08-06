"""
selection.py
选股模块：对 universe 跑策略共识打分，输出 Candidate 列表。

基础版：对每个标的跑所有策略，统计当前 bar 的看多/看空票数，
        score = 100 * 多票/总票，reasons["votes"] 可追溯到具体策略票。
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List

from .data_cache import get_bars
from .models import Candidate, utc_now
from .strategy_core import STRATEGY_OPTIONS, compute_signals

if TYPE_CHECKING:
    from .ai.safety import AIScoreSnapshot

logger = logging.getLogger(__name__)


class ConsensusSelector:
    """实现 Selector —— 策略共识打分选股。"""

    def __init__(
        self,
        strategies: List[str] | None = None,
        min_bars: int = 60,
    ) -> None:
        default_strategies = (
            list(STRATEGY_OPTIONS.keys())
            if hasattr(STRATEGY_OPTIONS, "keys")
            else list(STRATEGY_OPTIONS)
        )
        self._strategies = strategies or default_strategies
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
        errored_strategies: List[str] = []
        for strat in self._strategies:
            try:
                result = compute_signals(df.copy(), strat)
                sig = int(result.iloc[-1].get("strat_signal", 0))
                if sig != 0:
                    votes[strat] = sig  # +1 看多, -1 看空
            except Exception as exc:
                # A strategy that errors is NOT the same as one that computed
                # and genuinely voted neutral — counting it in the denominator
                # silently dilutes the consensus score. Exclude it and record
                # which ones failed so a systematic data issue is visible
                # (e.g. in the UI/audit) instead of just quietly lowering
                # scores across the board.
                logger.warning("selection %s 策略 %s 跳过: %s", symbol, strat, exc)
                errored_strategies.append(strat)

        total = len(self._strategies) - len(errored_strategies)
        bull = sum(1 for v in votes.values() if v > 0)
        score = round(100.0 * bull / total, 1) if total > 0 else 50.0

        return Candidate(
            symbol=symbol,
            score=score,
            rank=0,  # filled after sort
            reasons={
                "votes": votes,
                "total_strategies": total,
                "errored_strategies": errored_strategies,
            },
            as_of=as_of or utc_now(),
        )


class AICandidateSelector:
    """选股模块（AI 直选版）：候选标的直接来自每日 AI 深度研究名单
    （已经过 AIScoreValidator 安全门），不再对 universe 跑 24 策略共识投票——
    历史上那层投票既不产生真正驱动交易的信号，也拖慢了每个 symbol 的处理。

    这里只做一个轻量技术确认：最新收盘价相对短期均线的位置必须跟 AI 判断的
    方向一致，用来避免在 AI 给出看多/看空结论后逆着当前动能立刻进场；确认
    动作不参与打分、不投票，纯粹是入场时机的过滤器。
    """

    def __init__(self, confirm_window: int = 5) -> None:
        self._window = confirm_window

    def select(
        self,
        ai_scores: Dict[str, "AIScoreSnapshot"],
        bars: Dict[str, list],
        score_threshold: float = 0.0,
    ) -> List[Candidate]:
        from .ai.safety import conviction_of

        candidates: List[Candidate] = []
        for symbol, snapshot in ai_scores.items():
            recommendation = getattr(snapshot, "recommendation", "HOLD")
            score = getattr(snapshot, "score", None)
            if recommendation not in ("BUY", "SELL") or score is None:
                continue
            # score_threshold 比的是方向无关的信心强度（conviction_of()，0-1 换算
            # 成 0-100），不是双极 score 本身——score 天然把 SELL 锁在 [0,50]，直接
            # 拿它跟这个门槛比，SELL 永远过不去，跟信号强弱无关。
            conviction = conviction_of(snapshot)
            if conviction is None or conviction * 100 < score_threshold:
                continue
            raw = bars.get(symbol)
            if not raw or len(raw) < self._window + 1:
                continue
            closes = [float(b.close) for b in raw[-self._window:]]
            sma = sum(closes) / len(closes)
            last_close = closes[-1]
            confirmed = (
                last_close >= sma if recommendation == "BUY" else last_close <= sma
            )
            if not confirmed:
                continue
            candidates.append(
                Candidate(
                    symbol=symbol,
                    score=float(score),
                    rank=0,
                    reasons={
                        "ai_recommendation": recommendation,
                        "trigger": f"sma{self._window}_confirm",
                        "run_id": getattr(snapshot, "run_id", None),
                    },
                )
            )
        candidates.sort(key=lambda c: c.score, reverse=True)
        for i, c in enumerate(candidates):
            c.rank = i + 1
        return candidates


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_default_selector: ConsensusSelector | None = None
