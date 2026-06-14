"""
trader/ai/agents/web_research.py
Web Research Agent：AI 自主生成搜索词 → 调 Agent-Reach CLI 抓取 → LLM 综合报告。

流程：
  1. 根据候选标的，LLM 生成 3-5 个精准搜索词（不同角度）
  2. 用 AgentReachClient 调用：Yahoo RSS / Twitter / Reddit / Jina Reader
  3. 合并所有结果，LLM 生成结构化热点报告
  4. 输出 Advisory(kind="web_research")，含 hotspots + risk_flags + summary

设计原则：
- Agent-Reach 未安装时自动降级（只用 yfinance RSS）
- LLM 不可用时返回原始抓取摘要（不崩溃）
- 不抓取需要登录的内容（Twitter/Reddit 仅在工具已安装时尝试）
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from trader.contracts import AgentContext
from trader.models import Advisory
from trader.ai.web_research import AgentReachClient, get_web_research_client
from .base import AgentBase

logger = logging.getLogger(__name__)

# ── 系统提示词 ───────────────────────────────────────────────────────────────

_QUERY_SYSTEM = """You are a financial research assistant.
Given a stock ticker and context, generate 3 targeted search queries to find:
1. Recent news and catalysts
2. Community/social sentiment
3. Potential risks or concerns

Respond ONLY in valid JSON:
{
  "queries": [
    {"query": "<search term>", "purpose": "news|sentiment|risk"},
    {"query": "<search term>", "purpose": "news|sentiment|risk"},
    {"query": "<search term>", "purpose": "news|sentiment|risk"}
  ]
}"""

_REPORT_SYSTEM = """You are a senior financial research analyst.
You have gathered raw web content about a stock. Analyze it and produce a hotspot report.

Respond ONLY in valid JSON:
{
  "hotspot_score": <integer 0-100, how hot/newsworthy is this stock right now>,
  "sentiment": "bullish" | "neutral" | "bearish" | "mixed",
  "hotspots": [
    {"topic": "<key topic>", "signal": "bullish|bearish|neutral", "source": "<where from>"}
  ],
  "risk_flags": ["<risk 1>", "<risk 2>"],
  "summary": "<2-3 sentence market narrative>",
  "confidence": <float 0.0-1.0>
}"""


class WebResearchAgent(AgentBase):
    """
    AI 驱动的网络热点研究 Agent。
    适合用于：盘前热点扫描、新闻异动确认、社区情绪检测。
    """

    role = "web_research"

    def __init__(
        self,
        client=None,
        web_client: AgentReachClient | None = None,
        max_symbols: int = 5,
        use_twitter: bool = True,
        use_reddit: bool = True,
    ) -> None:
        from trader.ai.client import make_client
        self._llm = client or make_client()
        self._web = web_client or get_web_research_client()
        self._max_symbols = max_symbols
        self._use_twitter = use_twitter
        self._use_reddit = use_reddit

    def run(self, ctx: AgentContext) -> List[Advisory]:
        # 取 top candidates 做研究
        symbols = [c.symbol for c in ctx.candidates[: self._max_symbols]]
        if not symbols:
            return []

        advisories: List[Advisory] = []
        for symbol in symbols:
            try:
                adv = self._research(symbol, ctx)
                if adv:
                    advisories.append(adv)
            except Exception as exc:
                logger.warning("WebResearchAgent 跳过 %s: %s", symbol, exc)
        return advisories

    def _research(self, symbol: str, ctx: AgentContext) -> Advisory | None:
        # Step 1: AI 生成搜索词
        queries = self._generate_queries(symbol, ctx)

        # Step 2: 执行搜索
        all_content: List[str] = []
        all_content += self._web.read_symbol_news(symbol, max_items=5)
        all_content += self._web.read_financial_rss(max_items_per_feed=3)

        for q in queries:
            query_str = q.get("query", "")
            purpose = q.get("purpose", "")
            if not query_str:
                continue

            if self._use_twitter and self._web.has_twitter():
                tweets = self._web.search_twitter(f"{query_str} ${symbol}", n=5)
                if tweets:
                    all_content.append(f"[Twitter/{purpose}] {'; '.join(tweets[:3])}")

            if self._use_reddit and self._web.has_reddit():
                posts = self._web.search_reddit(f"{query_str} {symbol}", n=3)
                if posts:
                    all_content.append(f"[Reddit/{purpose}] {'; '.join(posts[:2])}")

        # 过滤空内容，最多 25 条
        content_lines = [c.strip() for c in all_content if c.strip()][:25]
        if not content_lines:
            logger.info("WebResearchAgent: %s 无网络内容，跳过", symbol)
            return None

        # Step 3: LLM 综合报告
        return self._generate_report(symbol, content_lines)

    def _generate_queries(self, symbol: str, ctx: AgentContext) -> List[Dict[str, str]]:
        """让 LLM 为这个标的生成精准搜索词。"""
        # 提取已有 advisory 的上下文
        existing_signals = []
        for adv in (ctx.extra or {}).get("technical_advisories", []):
            if adv.payload.get("symbol") == symbol:
                existing_signals.append(
                    f"Technical: {adv.payload.get('trend','?')} trend, "
                    f"score={adv.payload.get('technical_score','?')}"
                )

        user_prompt = (
            f"Stock: {symbol}\n"
            f"Signals: {', '.join(existing_signals) or 'None available'}\n"
            f"Today's date: today\n"
            f"Generate 3 search queries to research this stock."
        )

        result = self._llm_json(
            self._llm, _QUERY_SYSTEM, user_prompt,
            fallback={
                "queries": [
                    {"query": f"{symbol} earnings news 2026", "purpose": "news"},
                    {"query": f"{symbol} stock catalyst reddit", "purpose": "sentiment"},
                    {"query": f"{symbol} risk downside concern", "purpose": "risk"},
                ]
            },
        )
        queries = result.get("queries", [])
        logger.debug("WebResearch %s: 生成 %d 个搜索词", symbol, len(queries))
        return queries

    def _generate_report(self, symbol: str, content_lines: List[str]) -> Advisory:
        """LLM 把抓取的内容综合成热点报告。"""
        content_text = "\n".join(f"  - {line}" for line in content_lines)
        user_prompt = (
            f"Stock: {symbol}\n"
            f"Gathered web content ({len(content_lines)} items):\n"
            f"{content_text}\n\n"
            f"Analyze this information and generate a hotspot report."
        )

        result = self._llm_json(
            self._llm, _REPORT_SYSTEM, user_prompt,
            fallback={
                "hotspot_score": 50,
                "sentiment": "neutral",
                "hotspots": [],
                "risk_flags": [],
                "summary": f"Web research gathered {len(content_lines)} items for {symbol}.",
                "confidence": 0.3,
            },
        )

        score = self._clamp_score(result.get("hotspot_score", 50))
        confidence = float(result.get("confidence", score / 100))

        logger.info(
            "WebResearch %s: hotspot_score=%.0f sentiment=%s",
            symbol, score, result.get("sentiment", "?"),
        )

        return self._advisory(
            kind="web_research",
            payload={
                "symbol": symbol,
                "hotspot_score": score,
                "sentiment": result.get("sentiment", "neutral"),
                "hotspots": result.get("hotspots", []),
                "risk_flags": result.get("risk_flags", []),
                "summary": result.get("summary", ""),
                "sources_count": len(content_lines),
                "twitter_available": self._web.has_twitter(),
                "reddit_available": self._web.has_reddit(),
            },
            confidence=confidence,
            model=getattr(self._llm, "_model", ""),
        )
