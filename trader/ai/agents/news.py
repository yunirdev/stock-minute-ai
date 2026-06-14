"""
trader/ai/agents/news.py
新闻情绪 Agent：读取 ctx.news（NewsEvent 列表），
用 LLM 对每个标的的新闻做情绪打分（0-100）。

数据来源优先级（按成本）：
  1. ctx.news（已由 news.py PriceMoveSource 产生的异动事件）
  2. yfinance news API（免费）
  3. 未来可接 NewsAPI / Benzinga
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List

from trader.contracts import AgentContext
from trader.models import Advisory, NewsEvent
from .base import AgentBase

logger = logging.getLogger(__name__)

_SYSTEM = """You are a financial news analyst specializing in US equities.
You will receive recent news and price movement events for a stock.
Score market sentiment from 0 to 100:
  0-30 = very negative / bearish catalyst
  31-60 = neutral / no clear catalyst
  61-100 = positive / bullish catalyst

Respond ONLY with valid JSON:
{
  "news_score": <integer 0-100>,
  "sentiment": "bullish" | "neutral" | "bearish",
  "catalysts": ["<key catalyst 1>", "<key catalyst 2>"],
  "risk_flags": ["<risk 1>"],
  "reasoning": "<1-2 sentences>",
  "confidence": <float 0.0-1.0>
}"""


def _fetch_yfinance_news(symbol: str, max_items: int = 5) -> List[str]:
    """用 yfinance 获取最新新闻标题（免费，无需 API key）。"""
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        news = ticker.news or []
        return [
            n.get("content", {}).get("title", "") or n.get("title", "")
            for n in news[:max_items]
            if n
        ]
    except Exception as exc:
        logger.debug("yfinance news fetch 失败 %s: %s", symbol, exc)
        return []


class NewsAgent(AgentBase):
    """新闻情绪 Agent：LLM 打分 + yfinance 新闻补充。"""

    role = "news"

    def __init__(self, client=None, use_yfinance: bool = True) -> None:
        from trader.ai.client import make_client
        self._client = client or make_client()
        self._use_yf = use_yfinance

    def run(self, ctx: AgentContext) -> List[Advisory]:
        # 按 symbol 聚合 NewsEvent
        by_symbol: Dict[str, List[NewsEvent]] = defaultdict(list)
        for event in ctx.news:
            if event.symbol:
                by_symbol[event.symbol].append(event)

        # 对没有新闻的 candidate 也跑（用 yfinance 补）
        for cand in ctx.candidates:
            if cand.symbol not in by_symbol:
                by_symbol[cand.symbol] = []

        advisories: List[Advisory] = []
        for symbol, events in by_symbol.items():
            try:
                adv = self._analyze(symbol, events)
                if adv:
                    advisories.append(adv)
            except Exception as exc:
                logger.warning("NewsAgent 跳过 %s: %s", symbol, exc)
        return advisories

    def _analyze(self, symbol: str, events: List[NewsEvent]) -> Advisory | None:
        # 整理新闻摘要
        news_lines: List[str] = []
        for e in events:
            news_lines.append(
                f"[{e.kind}] {e.title}"
                + (f" (severity={e.severity:.2f})" if e.severity > 0 else "")
            )

        # yfinance 补充
        if self._use_yf:
            yf_titles = _fetch_yfinance_news(symbol, max_items=5)
            for t in yf_titles:
                if t:
                    news_lines.append(f"[yfinance] {t}")

        if not news_lines:
            return self._advisory(
                kind="news",
                payload={"symbol": symbol, "news_score": 50,
                         "sentiment": "neutral", "catalysts": [],
                         "risk_flags": [], "reasoning": "无新闻数据"},
                confidence=0.2,
            )

        user_prompt = f"""Stock: {symbol}
Recent news and events ({len(news_lines)} items):
{chr(10).join(f'  - {line}' for line in news_lines[:10])}

Analyze the sentiment and identify key catalysts or risks."""

        result = self._llm_json(
            self._client, _SYSTEM, user_prompt,
            fallback={
                "news_score": 50,
                "sentiment": "neutral",
                "catalysts": [],
                "risk_flags": [],
                "reasoning": "LLM unavailable, defaulting to neutral",
                "confidence": 0.2,
            },
        )

        score = self._clamp_score(result.get("news_score", 50))
        return self._advisory(
            kind="news",
            payload={
                "symbol": symbol,
                "news_score": score,
                "sentiment": result.get("sentiment", "neutral"),
                "catalysts": result.get("catalysts", []),
                "risk_flags": result.get("risk_flags", []),
                "reasoning": result.get("reasoning", ""),
                "news_count": len(news_lines),
            },
            confidence=float(result.get("confidence", score / 100)),
            model=getattr(self._client, "_model", ""),
        )
