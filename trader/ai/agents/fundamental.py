"""
trader/ai/agents/fundamental.py
基本面 Agent — 分析公司财务健康度与估值。

数据来源（yfinance .info，免费）：
  - trailingPE / forwardPE / pegRatio   估值
  - revenueGrowth / earningsGrowth       成长性
  - profitMargins / returnOnEquity       盈利质量
  - debtToEquity / currentRatio         资产负债
  - targetMeanPrice / recommendationMean 分析师观点

局限：
  - yfinance 数据通常滞后 24-48h，且部分字段对小市值股票缺失
  - 不包含实时财报数据（需 Refinitiv / Bloomberg）
  - 无法访问管理层会议纪录、行业竞争深度分析
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from trader.models import AgentContext
from trader.models import Advisory
from .base import AgentBase

logger = logging.getLogger(__name__)

_NON_FUNDAMENTAL_SYMBOLS = {
    "ARKF",
    "ARKG",
    "ARKK",
    "ARKQ",
    "ARKW",
    "DIA",
    "GLD",
    "HYG",
    "IEF",
    "IWM",
    "LQD",
    "QQQ",
    "SHY",
    "SLV",
    "SPY",
    "TLT",
    "UNG",
    "UUP",
    "USO",
    "VIX",
    "VNQ",
    "VOO",
    "VXX",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLU",
    "XLV",
    "XLY",
    "XRT",
}

_SYSTEM = """You are a fundamental analyst evaluating US equities.
Score the stock from 0-100 based on valuation, growth, profitability, and balance sheet quality.

Respond ONLY with valid JSON:
{
  "fundamental_score": <integer 0-100>,
  "valuation": "cheap" | "fair" | "expensive" | "very_expensive",
  "growth_quality": "high" | "medium" | "low" | "declining",
  "financial_health": "strong" | "adequate" | "weak",
  "key_strengths": ["<strength 1>", "<strength 2>"],
  "key_risks": ["<risk 1>", "<risk 2>"],
  "reasoning": "<2-3 sentences>",
  "confidence": <float 0.0-1.0>
}

Scoring guidance:
  70-100: Strong fundamentals — reasonable valuation + solid growth + healthy balance sheet
  50-69:  Mixed — one or two positives offset by concerns
  30-49:  Weak — expensive or deteriorating fundamentals
  0-29:   Poor — overvalued + declining growth + stressed balance sheet
"""


class FundamentalAgent(AgentBase):
    """
    基本面 Agent：通过 yfinance 获取财务指标，LLM 综合评分。
    每个候选标的独立分析，不依赖其他 agent 输出。
    """

    role = "fundamental"

    def __init__(self, client=None) -> None:
        from trader.ai.client import make_client

        self._client = client or make_client()

    def run(self, ctx: AgentContext) -> List[Advisory]:
        advisories: List[Advisory] = []
        for cand in ctx.candidates:
            try:
                adv = self._analyze(cand.symbol)
                if adv:
                    advisories.append(adv)
            except Exception as exc:
                logger.warning("FundamentalAgent 跳过 %s: %s", cand.symbol, exc)
        return advisories

    # ── 核心逻辑 ─────────────────────────────────────────────────────────────

    def _analyze(self, symbol: str) -> Optional[Advisory]:
        info = self._fetch_info(symbol)
        if not info:
            logger.info("FundamentalAgent: %s 无基本面数据，跳过", symbol)
            return None

        # 规则预分（无 LLM 时也能给出数值，供 fallback 使用）
        pre_score = self._rule_score(info)

        result = self._llm_json(
            self._client,
            _SYSTEM,
            self._build_prompt(symbol, info),
            fallback={
                "fundamental_score": pre_score,
                "valuation": self._valuation_label(info.get("forward_pe")),
                "growth_quality": self._growth_label(info.get("revenue_growth")),
                "financial_health": "adequate",
                "key_strengths": [],
                "key_risks": [],
                "reasoning": f"Rule-based score: {pre_score}/100",
                "confidence": 0.3,
            },
        )

        score = self._clamp_score(result.get("fundamental_score", pre_score))
        return self._advisory(
            kind="fundamental",
            payload={
                "symbol": symbol,
                "fundamental_score": score,
                "valuation": result.get("valuation", "fair"),
                "growth_quality": result.get("growth_quality", "medium"),
                "financial_health": result.get("financial_health", "adequate"),
                "key_strengths": result.get("key_strengths", []),
                "key_risks": result.get("key_risks", []),
                "reasoning": result.get("reasoning", ""),
                # 原始数值保留，方便审查
                "pe_trailing": info.get("pe_trailing"),
                "pe_forward": info.get("forward_pe"),
                "peg_ratio": info.get("peg_ratio"),
                "revenue_growth_pct": _pct(info.get("revenue_growth")),
                "earnings_growth_pct": _pct(info.get("earnings_growth")),
                "profit_margin_pct": _pct(info.get("profit_margin")),
                "roe_pct": _pct(info.get("roe")),
                "debt_equity": info.get("debt_equity"),
                "current_ratio": info.get("current_ratio"),
                "analyst_target": info.get("target_price"),
                "analyst_rating": info.get("recommendation"),
                "sector": info.get("sector", ""),
                "market_cap": info.get("market_cap"),
            },
            confidence=float(result.get("confidence", score / 100)),
            model=getattr(self._client, "_model", ""),
        )

    # ── 数据获取 ─────────────────────────────────────────────────────────────

    def _fetch_info(self, symbol: str) -> Dict[str, Any]:
        symbol_u = symbol.upper()
        if (
            symbol_u.startswith("^")
            or symbol_u.endswith("=F")
            or symbol_u in _NON_FUNDAMENTAL_SYMBOLS
        ):
            logger.info(
                "FundamentalAgent: %s is ETF/index-like, skipping company fundamentals",
                symbol_u,
            )
            return {}

        try:
            import yfinance as yf

            raw = yf.Ticker(symbol).info
            if not raw or raw.get("quoteType") in {"ETF", "INDEX", "MUTUALFUND"}:
                return {}
            return {
                "pe_trailing": raw.get("trailingPE"),
                "forward_pe": raw.get("forwardPE"),
                "peg_ratio": raw.get("pegRatio"),
                "revenue_growth": raw.get("revenueGrowth"),  # decimal: 0.15 = 15%
                "earnings_growth": raw.get("earningsGrowth"),  # decimal
                "profit_margin": raw.get("profitMargins"),  # decimal
                "roe": raw.get("returnOnEquity"),  # decimal
                "debt_equity": raw.get("debtToEquity"),  # %
                "current_ratio": raw.get("currentRatio"),
                "market_cap": raw.get("marketCap"),
                "target_price": raw.get("targetMeanPrice"),
                "recommendation": raw.get(
                    "recommendationMean"
                ),  # 1=strong buy .. 5=sell
                "sector": raw.get("sector", ""),
                "industry": raw.get("industry", ""),
            }
        except Exception as exc:
            msg = str(exc)
            if (
                "No fundamentals data found" in msg
                or "HTTP Error 404" in msg
                or "quoteSummary" in msg
            ):
                logger.info(
                    "FundamentalAgent: %s has no company fundamentals, skipping", symbol
                )
            else:
                logger.warning("FundamentalAgent: fetch_info %s → %s", symbol, exc)
            return {}

    # ── 规则打分（LLM fallback）──────────────────────────────────────────────

    def _rule_score(self, info: Dict) -> int:
        score = 50.0

        fpe = info.get("forward_pe")
        if fpe is not None:
            if fpe < 12:
                score += 15
            elif fpe < 18:
                score += 8
            elif fpe < 25:
                score += 2
            elif fpe < 35:
                score -= 5
            else:
                score -= 15

        rev_g = info.get("revenue_growth")
        if rev_g is not None:
            if rev_g > 0.30:
                score += 12
            elif rev_g > 0.15:
                score += 7
            elif rev_g > 0.05:
                score += 2
            elif rev_g < 0:
                score -= 10

        margin = info.get("profit_margin")
        if margin is not None:
            if margin > 0.25:
                score += 8
            elif margin > 0.10:
                score += 3
            elif margin < 0:
                score -= 12

        de = info.get("debt_equity")
        if de is not None:
            if de < 30:
                score += 5
            elif de > 150:
                score -= 8

        rec = info.get("recommendation")
        if rec is not None:
            score += (3.0 - rec) * 4  # 1=buy→+8, 3=hold→0, 5=sell→-8

        return max(0, min(100, int(score)))

    @staticmethod
    def _valuation_label(fpe) -> str:
        if fpe is None:
            return "unknown"
        if fpe < 15:
            return "cheap"
        if fpe < 25:
            return "fair"
        if fpe < 40:
            return "expensive"
        return "very_expensive"

    @staticmethod
    def _growth_label(rev_g) -> str:
        if rev_g is None:
            return "unknown"
        if rev_g > 0.20:
            return "high"
        if rev_g > 0.05:
            return "medium"
        if rev_g >= 0:
            return "low"
        return "declining"

    # ── Prompt 构建 ──────────────────────────────────────────────────────────

    def _build_prompt(self, symbol: str, info: Dict) -> str:
        def f(v, suffix="", factor=1.0) -> str:
            if v is None:
                return "N/A"
            return f"{float(v) * factor:.1f}{suffix}"

        rec_map = {1: "Strong Buy", 2: "Buy", 3: "Hold", 4: "Sell", 5: "Strong Sell"}
        rec_val = info.get("recommendation")
        rec_str = (
            rec_map.get(round(rec_val) if rec_val else 0, f"{rec_val:.1f}")
            if rec_val
            else "N/A"
        )

        mcap = info.get("market_cap")
        mcap_str = (
            f"${mcap / 1e9:.1f}B"
            if mcap and mcap >= 1e9
            else (f"${mcap / 1e6:.0f}M" if mcap else "N/A")
        )

        return f"""FUNDAMENTAL ANALYSIS REQUEST: {symbol}
Sector: {info.get("sector", "N/A")} | Industry: {info.get("industry", "N/A")}
Market Cap: {mcap_str}

VALUATION:
  Trailing P/E:  {f(info.get("pe_trailing"), "x")}
  Forward P/E:   {f(info.get("forward_pe"), "x")}
  PEG Ratio:     {f(info.get("peg_ratio"))}
  Analyst Target: ${f(info.get("target_price"))} | Consensus: {rec_str}

GROWTH (YoY):
  Revenue Growth:  {f(info.get("revenue_growth"), "%", 100)}
  Earnings Growth: {f(info.get("earnings_growth"), "%", 100)}

PROFITABILITY:
  Net Margin: {f(info.get("profit_margin"), "%", 100)}
  ROE:        {f(info.get("roe"), "%", 100)}

BALANCE SHEET:
  Debt/Equity:    {f(info.get("debt_equity"), "%")}
  Current Ratio:  {f(info.get("current_ratio"))}

Note: Data from yfinance, may lag 24-48h. Score the stock 0-100."""


def _pct(v) -> Optional[float]:
    if v is None:
        return None
    return round(float(v) * 100, 1)
