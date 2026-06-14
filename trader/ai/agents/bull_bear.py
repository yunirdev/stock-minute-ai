"""
trader/ai/agents/bull_bear.py
Bull vs Bear 辩论 Agent。

流程：
  1. BullAgent  → 构建最强看多论点（entry/upside/catalysts）
  2. BearAgent  → 构建最强看空论点（risks/downside/threats）
  3. JudgeAgent → 综合双方，输出最终得分 + 裁决
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from trader.contracts import AgentContext
from trader.models import Advisory
from .base import AgentBase

logger = logging.getLogger(__name__)

_BULL_SYSTEM = """You are a BULL analyst making the strongest possible LONG case.
Focus on: catalysts, growth drivers, competitive moat, upside scenarios.
Be assertive but realistic. Respond ONLY in valid JSON:
{
  "bull_score": <integer 0-100>,
  "thesis": "<main bull thesis in 1-2 sentences>",
  "catalysts": ["<catalyst 1>", "<catalyst 2>"],
  "upside_target": <float price or 0 if unknown>,
  "time_horizon": "1m" | "3m" | "6m" | "1y",
  "confidence": <float 0.0-1.0>
}"""

_BEAR_SYSTEM = """You are a BEAR analyst making the strongest possible SHORT/AVOID case.
Focus on: risks, competitive threats, valuation concerns, downside scenarios.
Be critical and rigorous. Respond ONLY in valid JSON:
{
  "bear_score": <integer 0-100>,
  "thesis": "<main bear thesis in 1-2 sentences>",
  "risks": ["<risk 1>", "<risk 2>"],
  "downside_target": <float price or 0 if unknown>,
  "stop_loss_pct": <float suggested stop loss % e.g. 5.0>,
  "confidence": <float 0.0-1.0>
}"""

_JUDGE_SYSTEM = """You are an impartial investment committee chair.
Review the bull and bear cases and make a final decision.
Respond ONLY in valid JSON:
{
  "verdict": "BUY" | "WATCHLIST" | "AVOID",
  "final_score": <integer 0-100>,
  "bull_weight": <float 0.0-1.0 how much you credit the bull case>,
  "bear_weight": <float 0.0-1.0 how much you credit the bear case>,
  "key_factor": "<the single most decisive factor>",
  "suggested_action": "<1 sentence recommendation>",
  "confidence": <float 0.0-1.0>
}"""


class BullBearDebate(AgentBase):
    """
    对 top candidates 运行 Bull-Bear 辩论，返回含裁决的 Advisory。
    只处理 ctx.candidates 中 score >= min_score 的标的（避免对差股浪费调用）。
    """

    role = "bull_bear"

    def __init__(
        self,
        client=None,
        min_score: float = 55.0,
        max_symbols: int = 3,
    ) -> None:
        from trader.ai.client import make_client
        self._client = client or make_client()
        self._min_score = min_score
        self._max_symbols = max_symbols

    def run(self, ctx: AgentContext) -> List[Advisory]:
        # 只辩论得分够高的 top candidates
        top = [
            c for c in sorted(ctx.candidates, key=lambda x: x.score, reverse=True)
            if c.score >= self._min_score
        ][: self._max_symbols]

        if not top:
            logger.info("BullBearDebate: 没有足够资质的 candidate（min_score=%.0f）",
                        self._min_score)
            return []

        # 聚合 advisory 信号作为辩论上下文
        tech_payload = {
            a.payload["symbol"]: a.payload
            for a in ctx.extra.get("technical_advisories", [])
            if "symbol" in a.payload
        } if ctx.extra else {}
        news_payload = {
            a.payload["symbol"]: a.payload
            for a in ctx.extra.get("news_advisories", [])
            if "symbol" in a.payload
        } if ctx.extra else {}

        advisories: List[Advisory] = []
        for cand in top:
            try:
                adv = self._debate(cand, tech_payload, news_payload)
                advisories.append(adv)
            except Exception as exc:
                logger.warning("BullBearDebate 跳过 %s: %s", cand.symbol, exc)
        return advisories

    def _debate(self, cand, tech: Dict, news: Dict) -> Advisory:
        sym = cand.symbol
        context = self._build_context(sym, cand, tech.get(sym, {}), news.get(sym, {}))

        bull = self._llm_json(
            self._client, _BULL_SYSTEM, f"Analyze {sym}:\n{context}",
            fallback={
                "bull_score": int(cand.score),
                "thesis": f"Technical momentum: score {cand.score:.0f}",
                "catalysts": list(cand.reasons.get("votes", {}).keys())[:2],
                "upside_target": 0,
                "time_horizon": "3m",
                "confidence": cand.score / 100,
            },
        )

        bear = self._llm_json(
            self._client, _BEAR_SYSTEM, f"Analyze {sym}:\n{context}",
            fallback={
                "bear_score": int(100 - cand.score),
                "thesis": "Mean reversion risk after momentum",
                "risks": ["Market-wide selloff", "Sector rotation"],
                "downside_target": 0,
                "stop_loss_pct": 5.0,
                "confidence": 0.4,
            },
        )

        judge_context = (
            f"Stock: {sym}\n"
            f"BULL CASE (score={bull.get('bull_score',50)}): {bull.get('thesis','')}\n"
            f"  Catalysts: {bull.get('catalysts', [])}\n\n"
            f"BEAR CASE (score={bear.get('bear_score',50)}): {bear.get('thesis','')}\n"
            f"  Risks: {bear.get('risks', [])}\n"
            f"  Stop loss: {bear.get('stop_loss_pct', 5)}%\n\n"
            f"Original consensus score: {cand.score:.0f}/100"
        )
        verdict = self._llm_json(
            self._client, _JUDGE_SYSTEM, judge_context,
            fallback={
                "verdict": "WATCHLIST" if cand.score < 65 else "BUY",
                "final_score": int(cand.score),
                "bull_weight": 0.5,
                "bear_weight": 0.5,
                "key_factor": "Consensus signal",
                "suggested_action": "Monitor closely",
                "confidence": 0.4,
            },
        )

        final_score = self._clamp_score(verdict.get("final_score", cand.score))
        confidence = float(verdict.get("confidence", final_score / 100))

        logger.info(
            "BullBear %s → %s score=%.0f conf=%.2f",
            sym, verdict.get("verdict", "?"), final_score, confidence,
        )

        return self._advisory(
            kind="bull_bear_debate",
            payload={
                "symbol": sym,
                "verdict": verdict.get("verdict", "WATCHLIST"),
                "final_score": final_score,
                "bull": {
                    "score": bull.get("bull_score", 50),
                    "thesis": bull.get("thesis", ""),
                    "catalysts": bull.get("catalysts", []),
                    "upside_target": bull.get("upside_target", 0),
                },
                "bear": {
                    "score": bear.get("bear_score", 50),
                    "thesis": bear.get("thesis", ""),
                    "risks": bear.get("risks", []),
                    "stop_loss_pct": bear.get("stop_loss_pct", 5.0),
                },
                "key_factor": verdict.get("key_factor", ""),
                "suggested_action": verdict.get("suggested_action", ""),
            },
            confidence=confidence,
            model=getattr(self._client, "_model", ""),
        )

    @staticmethod
    def _build_context(sym: str, cand, tech: Dict, news: Dict) -> str:
        lines = [f"Symbol: {sym}", f"Consensus score: {cand.score:.1f}/100"]
        if tech:
            lines.append(
                f"Technical: trend={tech.get('trend','?')} "
                f"momentum={tech.get('momentum','?')} "
                f"score={tech.get('technical_score','?')}"
            )
        if news:
            lines.append(
                f"News: sentiment={news.get('sentiment','?')} "
                f"score={news.get('news_score','?')} "
                f"catalysts={news.get('catalysts',[])}"
            )
        votes = cand.reasons.get("votes", {})
        if votes:
            lines.append(
                f"Strategy signals: "
                f"{sum(1 for v in votes.values() if v>0)} bullish, "
                f"{sum(1 for v in votes.values() if v<0)} bearish"
            )
        return "\n".join(lines)
