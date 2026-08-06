"""
trader/ai/agents/macro.py
宏观环境 Agent — 评估美股当前的宏观流动性与风险偏好。

数据来源（全部免费，通过 yfinance 获取）：
  ^VIX   → 恐慌指数（不确定性）
  TLT    → 20+ 年美债 ETF（长端利率方向）
  ^TNX   → 10 年期美债收益率
  UUP    → 美元指数 ETF（DXY 代理；非精确）
  GLD    → 黄金 ETF（避险 / 通胀代理）
  ^GSPC  → 标普 500（大盘动量）

局限：
  - UUP 是 DXY 的近似代理，并非精确美元指数
  - 不包含 M2、美联储资产负债表等深度流动性数据
    （需要 FRED API key，目前未接入）

输出：每个 candidate symbol 输出一份相同的宏观 advisory，
供综合分聚合时加权使用。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from trader.models import AgentContext
from trader.models import Advisory
from .base import AgentBase

logger = logging.getLogger(__name__)

_MACRO_TICKERS = {
    "vix":     "^VIX",
    "bonds":   "TLT",
    "rate10y": "^TNX",
    "dollar":  "UUP",
    "gold":    "GLD",
    "spx":     "^GSPC",
}

_SYSTEM = """You are a macro strategist analyzing the US equity market environment.
Based on key macro proxy indicators, assess the current regime and its implications for stocks.

Respond ONLY with valid JSON:
{
  "macro_score": <integer 0-100, higher = more bullish macro environment for equities>,
  "regime": "risk_on" | "neutral" | "risk_off",
  "vix_regime": "low" | "normal" | "elevated" | "extreme",
  "rate_outlook": "bullish_for_equities" | "neutral" | "bearish_for_equities",
  "dollar_signal": "tailwind" | "neutral" | "headwind",
  "liquidity": "expanding" | "neutral" | "tightening",
  "key_factors": ["<factor 1>", "<factor 2>", "<factor 3>"],
  "reasoning": "<2-3 sentences>",
  "confidence": <float 0.0-1.0>
}

Scoring guidance:
  80-100 = strong risk-on: low VIX, falling rates, weak dollar, rising bonds, risk assets up
  60-79  = mild risk-on: most signals positive
  40-59  = neutral: mixed signals
  20-39  = mild risk-off: some stress indicators elevated
  0-19   = strong risk-off: high VIX, rates spiking, flight-to-quality
"""


class MacroAgent(AgentBase):
    """
    宏观环境 Agent。
    一次 LLM 调用产出全市场宏观判断，再复制给每个候选标的。
    """

    role = "macro"
    _LOOKBACK_DAYS = "3mo"

    def __init__(self, client=None) -> None:
        from trader.ai.client import make_client
        self._client = client or make_client()

    def run(self, ctx: AgentContext) -> List[Advisory]:
        try:
            raw = self._fetch_macro()
        except Exception as exc:
            logger.warning("MacroAgent: 数据拉取失败 → %s", exc)
            raw = {}

        result = self._llm_json(
            self._client, _SYSTEM,
            self._build_prompt(raw),
            fallback={
                "macro_score": 50,
                "regime": "neutral",
                "vix_regime": "normal",
                "rate_outlook": "neutral",
                "dollar_signal": "neutral",
                "liquidity": "neutral",
                "key_factors": ["insufficient data"],
                "reasoning": "LLM unavailable; defaulting to neutral macro.",
                "confidence": 0.2,
            },
        )

        score = self._clamp_score(result.get("macro_score", 50))
        payload_base: Dict[str, Any] = {
            "macro_score": score,
            "regime": result.get("regime", "neutral"),
            "vix_level": raw.get("vix_current"),
            "vix_regime": result.get("vix_regime", "normal"),
            "rate_outlook": result.get("rate_outlook", "neutral"),
            "dollar_signal": result.get("dollar_signal", "neutral"),
            "liquidity": result.get("liquidity", "neutral"),
            "key_factors": result.get("key_factors", []),
            "reasoning": result.get("reasoning", ""),
            # raw 数值供审查
            "rate10y": raw.get("rate10y_current"),
            # _MACRO_TICKERS 的 key 是 "bonds"（→TLT），产出的键名是
            # bonds_1m_ret；原来读 tlt_1m_ret 永远是 None
            "tlt_1m_ret": raw.get("bonds_1m_ret"),
            "dollar_1m_ret": raw.get("dollar_1m_ret"),
            "gold_1m_ret": raw.get("gold_1m_ret"),
            "spx_1m_ret": raw.get("spx_1m_ret"),
        }

        symbols = [c.symbol for c in ctx.candidates] or ["MARKET"]
        advisories: List[Advisory] = []
        for sym in symbols:
            advisories.append(self._advisory(
                kind="macro",
                payload=dict(payload_base, symbol=sym),
                confidence=float(result.get("confidence", score / 100)),
                model=getattr(self._client, "_model", ""),
                is_fallback=self._is_fallback_result(result),
            ))
        logger.info("MacroAgent: regime=%s score=%d symbols=%d",
                    payload_base["regime"], score, len(advisories))
        return advisories

    # ── 数据获取 ─────────────────────────────────────────────────────────────

    def _fetch_macro(self) -> Dict[str, Optional[float]]:
        import yfinance as yf

        data: Dict[str, Optional[float]] = {}
        for name, ticker in _MACRO_TICKERS.items():
            try:
                hist = yf.Ticker(ticker).history(
                    period=self._LOOKBACK_DAYS, interval="1d", auto_adjust=True)
                if hist.empty:
                    continue
                close = hist["Close"].dropna()
                data[f"{name}_current"] = float(close.iloc[-1])
                for n_days, label in [(5, "5d"), (20, "1m"), (60, "3m")]:
                    if len(close) > n_days:
                        ret = (close.iloc[-1] / close.iloc[-n_days - 1] - 1) * 100
                        data[f"{name}_{label}_ret"] = round(float(ret), 2)
            except Exception as exc:
                logger.debug("MacroAgent: fetch %s 失败 → %s", ticker, exc)

        return data

    def _build_prompt(self, d: Dict) -> str:
        def v(key: str) -> str:
            val = d.get(key)
            return f"{val:.2f}" if val is not None else "N/A"

        return f"""US MACRO ENVIRONMENT — PROXY INDICATORS

VOLATILITY (VIX):
  Current: {v('vix_current')} | 5d chg: {v('vix_5d_ret')}% | 1m chg: {v('vix_1m_ret')}%
  Interpretation: <15=low, 15-25=normal, 25-35=elevated, >35=extreme

INTEREST RATES:
  10Y Yield: {v('rate10y_current')}% | 1m chg: {v('rate10y_1m_ret')}% (relative change of the yield, NOT basis points)
  TLT (Long Bond ETF): 1m ret={v('bonds_1m_ret')}% | 3m ret={v('bonds_3m_ret')}%

US DOLLAR (UUP — DXY proxy):
  1m return: {v('dollar_1m_ret')}% | 3m return: {v('dollar_3m_ret')}%

GOLD (Inflation/Risk-off proxy):
  1m return: {v('gold_1m_ret')}% | 3m return: {v('gold_3m_ret')}%

S&P 500:
  1m return: {v('spx_1m_ret')}% | 3m return: {v('spx_3m_ret')}%

Note: UUP is a proxy for USD index (not exact DXY).
Analyze the macro regime implications for US equity investors."""
