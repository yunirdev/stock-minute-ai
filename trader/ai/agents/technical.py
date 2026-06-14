"""
trader/ai/agents/technical.py
技术分析 Agent：读取候选标的的 bars + strategy_core 信号，
用 LLM 综合打分（0-100）并给出结构化理由。
"""
from __future__ import annotations

import logging
from typing import List

from trader.contracts import AgentContext
from trader.data_cache import get_bars
from trader.models import Advisory
from trader.strategy_core import STRATEGY_OPTIONS, compute_signals
from .base import AgentBase

logger = logging.getLogger(__name__)

_SYSTEM = """You are a quantitative technical analyst specializing in US equities.
You will receive technical indicators for a stock and must score it from 0 to 100:
  0-30 = bearish / avoid
  31-60 = neutral / watch
  61-100 = bullish / buy candidate

Respond ONLY with valid JSON in this exact format:
{
  "score": <integer 0-100>,
  "trend": "bullish" | "neutral" | "bearish",
  "momentum": "strong" | "moderate" | "weak",
  "key_signals": ["<signal 1>", "<signal 2>"],
  "reasoning": "<1-2 sentences>",
  "confidence": <float 0.0-1.0>
}"""


class TechnicalAgent(AgentBase):
    """用 LLM 对技术指标/信号综合打分。"""

    role = "technical"

    def __init__(self, client=None, timeframe: str = "5m", min_bars: int = 60) -> None:
        from trader.ai.client import make_client
        self._client = client or make_client()
        self._timeframe = timeframe
        self._min_bars = min_bars

    def run(self, ctx: AgentContext) -> List[Advisory]:
        advisories: List[Advisory] = []
        for cand in ctx.candidates:
            try:
                adv = self._analyze(cand.symbol)
                if adv:
                    advisories.append(adv)
            except Exception as exc:
                logger.warning("TechnicalAgent 跳过 %s: %s", cand.symbol, exc)
        return advisories

    def _analyze(self, symbol: str) -> Advisory | None:
        df = get_bars(symbol, self._timeframe)
        if df is None or len(df) < self._min_bars:
            logger.info("TechnicalAgent: %s 数据不足，跳过", symbol)
            return None

        # 运行多个策略，收集最新信号
        signals: dict = {}
        for strat in list(STRATEGY_OPTIONS.keys())[:6]:  # 最多6个策略
            try:
                result = compute_signals(df.copy(), strat)
                last = result.iloc[-1]
                sig = int(last.get("strat_signal", 0))
                px = float(last.get("strat_exec_px", last["close"]))
                if sig != 0:
                    signals[strat] = {"signal": sig, "exec_px": round(px, 2)}
            except Exception:
                pass

        # 基础指标摘要
        close = df["close"].iloc[-1]
        close_1d_ago = df["close"].iloc[max(0, len(df) - 78)]  # ~1天前(5min*78=390min)
        change_1d = (close - close_1d_ago) / close_1d_ago * 100 if close_1d_ago else 0

        vol_recent = df["volume"].iloc[-20:].mean()
        vol_baseline = df["volume"].mean()
        vol_ratio = vol_recent / vol_baseline if vol_baseline else 1.0

        bull_votes = sum(1 for v in signals.values() if v["signal"] > 0)
        bear_votes = sum(1 for v in signals.values() if v["signal"] < 0)

        user_prompt = f"""Stock: {symbol}
Current price: ${close:.2f}
1-day change: {change_1d:+.2f}%
Volume ratio (recent/avg): {vol_ratio:.2f}x

Strategy signals ({len(signals)} fired out of {len(STRATEGY_OPTIONS)}):
  Bullish: {bull_votes}  Bearish: {bear_votes}
Signal details: {signals}

Based on these technical indicators, provide your analysis."""

        result = self._llm_json(
            self._client, _SYSTEM, user_prompt,
            fallback={
                "score": 50 + (bull_votes - bear_votes) * 5,
                "trend": "bullish" if bull_votes > bear_votes else
                         "bearish" if bear_votes > bull_votes else "neutral",
                "momentum": "moderate",
                "key_signals": list(signals.keys())[:2],
                "reasoning": f"Rule-based: {bull_votes}bull/{bear_votes}bear signals",
                "confidence": 0.4,
            },
        )

        score = self._clamp_score(result.get("score", 50))
        return self._advisory(
            kind="technical",
            payload={
                "symbol": symbol,
                "technical_score": score,
                "trend": result.get("trend", "neutral"),
                "momentum": result.get("momentum", "moderate"),
                "key_signals": result.get("key_signals", []),
                "reasoning": result.get("reasoning", ""),
                "raw_signals": signals,
                "close": close,
                "change_1d_pct": round(change_1d, 2),
                "vol_ratio": round(vol_ratio, 2),
            },
            confidence=float(result.get("confidence", score / 100)),
            model=getattr(self._client, "_model", ""),
        )
