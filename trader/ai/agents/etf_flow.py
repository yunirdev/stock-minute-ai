"""
trader/ai/agents/etf_flow.py
ETF 资金流 Agent — 通过 ETF 相对成交量和价格动量估算资金流向。

数据来源：yfinance（免费）
⚠ 局限（请知悉）：
  - 此处的"资金流"是代理指标：相对成交量 × 价格动量
  - 并非真实 ETF 申购/赎回流量（真实数据需 Bloomberg / etf.com / Morningstar）
  - 适合方向性判断，不适合精确定量

监控标的：
  市值维度：SPY(大盘) / QQQ(科技权重) / IWM(小盘)
  行业 SPDR：XLK / XLF / XLE / XLV / XLI / XLY / XLP / XLU / XLB / XLRE / XLC
  主题：SMH(半导体) / SOXX(半导体) / IBB(生技) / GLD(黄金) / TLT(长债)

评分逻辑（纯算法，无 LLM）：
  ① 大盘 ETF 信号（40%）：SPY / QQQ / IWM 量价综合
  ② 行业 ETF 信号（40%）：候选标的所属行业 ETF 强弱
  ③ 风险偏好信号（20%）：stock vs bond / gold 相对强弱

独立运行，不依赖其他 agent 输出。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from trader.models import AgentContext
from trader.models import Advisory
from .base import AgentBase

logger = logging.getLogger(__name__)

# 大盘 ETF
_BROAD_ETFS = ["SPY", "QQQ", "IWM"]

# 行业 SPDR → yfinance sector 名称映射
_SECTOR_ETF: Dict[str, str] = {
    "Technology":             "XLK",
    "Financial Services":     "XLF",
    "Energy":                 "XLE",
    "Health Care":            "XLV",
    "Industrials":            "XLI",
    "Consumer Cyclical":      "XLY",
    "Consumer Defensive":     "XLP",
    "Utilities":              "XLU",
    "Basic Materials":        "XLB",
    "Real Estate":            "XLRE",
    "Communication Services": "XLC",
}

# 风险偏好代理
_RISK_PROXY = {"risk_on": "SPY", "risk_off": "TLT", "gold": "GLD"}


class ETFFlowAgent(AgentBase):
    """
    ETF 资金流 Agent（纯算法，无 LLM 调用）。
    评估大盘资金流向 + 候选标的行业轮动信号。
    """

    role = "etf_flow"

    def __init__(self, client=None) -> None:
        pass  # 不需要 LLM

    def run(self, ctx: AgentContext) -> List[Advisory]:
        # 一次性批量获取所有 ETF 数据（缓存避免重复请求）
        etf_cache: Dict[str, Any] = {}
        all_etfs = _BROAD_ETFS + list(_SECTOR_ETF.values()) + list(_RISK_PROXY.values())
        for ticker in set(all_etfs):
            try:
                etf_cache[ticker] = _fetch_etf_signal(ticker)
            except Exception as exc:
                logger.debug("ETFFlowAgent: fetch %s 失败 → %s", ticker, exc)

        # 获取每个 candidate 的行业
        sector_cache: Dict[str, str] = {}
        for cand in ctx.candidates:
            try:
                sector_cache[cand.symbol] = _get_sector(cand.symbol)
            except Exception:
                sector_cache[cand.symbol] = ""

        advisories: List[Advisory] = []
        for cand in ctx.candidates:
            try:
                adv = self._analyze(cand.symbol, sector_cache[cand.symbol], etf_cache)
                if adv:
                    advisories.append(adv)
            except Exception as exc:
                logger.warning("ETFFlowAgent 跳过 %s: %s", cand.symbol, exc)
        return advisories

    # ── 核心分析 ─────────────────────────────────────────────────────────────

    def _analyze(
        self, symbol: str, sector: str, cache: Dict[str, Any]
    ) -> Optional[Advisory]:

        # ① 大盘 ETF 综合信号（40%）
        broad_scores = [cache[t]["flow_score"] for t in _BROAD_ETFS if t in cache and cache[t]]
        broad_signal = _avg(broad_scores) if broad_scores else 50.0

        # ② 行业 ETF 信号（40%）
        sector_etf = _SECTOR_ETF.get(sector, "")
        sector_signal = 50.0
        sector_detail: Dict = {}
        if sector_etf and sector_etf in cache and cache[sector_etf]:
            s = cache[sector_etf]
            sector_signal = s["flow_score"]
            sector_detail = {"etf": sector_etf, **{k: s[k] for k in
                             ("ret_1w", "ret_1m", "vol_ratio_5d") if k in s}}

        # ③ 风险偏好信号（20%）
        risk_on  = (cache.get("SPY") or {}).get("ret_1m", 0) or 0
        risk_off = (cache.get("TLT") or {}).get("ret_1m", 0) or 0
        gold     = (cache.get("GLD") or {}).get("ret_1m", 0) or 0
        risk_score = 50 + (risk_on - risk_off) * 1.5

        # 综合
        composite = broad_signal * 0.40 + sector_signal * 0.40 + risk_score * 0.20
        etf_score = int(max(0, min(100, composite)))

        # 市场状态标签
        if broad_signal >= 60 and sector_signal >= 55:
            market_flow = "risk_on"
        elif broad_signal <= 40 and sector_signal <= 45:
            market_flow = "risk_off"
        else:
            market_flow = "neutral"

        sector_flow = ("inflow" if sector_signal >= 58 else
                       "outflow" if sector_signal <= 42 else "neutral")

        logger.info("ETFFlowAgent %s: score=%d broad=%.0f sector=%.0f(%s)",
                    symbol, etf_score, broad_signal, sector_signal, sector_etf or "N/A")

        return self._advisory(
            kind="etf_flow",
            payload={
                "symbol": symbol,
                "etf_score": etf_score,
                "market_flow": market_flow,
                "sector": sector,
                "sector_etf": sector_etf,
                "sector_flow": sector_flow,
                "broad_signal": round(broad_signal, 1),
                "sector_detail": sector_detail,
                "risk_on_1m_ret": round(risk_on, 2),
                "risk_off_1m_ret": round(risk_off, 2),
                "gold_1m_ret": round(gold, 2),
                "data_note": "流向为代理指标（量价），非真实ETF申购赎回数据",
            },
            confidence=0.5 if sector_etf else 0.3,
            model="algorithmic",
            # 大盘、行业、风险偏好三路数据全部缺失时，composite 会正好是
            # 50*0.4+50*0.4+50*0.2 = 50 —— 一个"看起来很中性"但其实什么
            # 都没读到的分数。标成 fallback，别让它占 10% 权重。
            is_fallback=not broad_scores and not sector_detail and not (risk_on or risk_off),
        )


# ── 辅助函数 ──────────────────────────────────────────────────────────────────

def _fetch_etf_signal(ticker: str) -> Optional[Dict[str, Any]]:
    """获取 ETF 的近期量价信号。"""
    import yfinance as yf
    hist = yf.Ticker(ticker).history(period="3mo", interval="1d", auto_adjust=True)
    if hist.empty or len(hist) < 10:
        return None

    close  = hist["Close"].dropna()
    volume = hist["Volume"].dropna()

    ret_1w = _ret(close, 5)
    ret_1m = _ret(close, 21)
    vol_ratio_5d = (
        float(volume.tail(5).mean() / volume.mean())
        if len(volume) >= 21 and float(volume.mean()) > 0 else None
    )

    # Flow score: 基于量价共振
    score = 50.0
    if ret_1m is not None:
        score += ret_1m * 1.5        # 1% 涨幅 → +1.5分
    if ret_1w is not None:
        score += ret_1w * 1.0        # 近期动量权重较低
    if vol_ratio_5d is not None:
        # 量增价涨 = 强流入；量增价跌 = 强流出
        if ret_1w is not None and ret_1w > 0 and vol_ratio_5d > 1.1:
            score += 5               # 放量上涨
        elif ret_1w is not None and ret_1w < 0 and vol_ratio_5d > 1.1:
            score -= 5               # 放量下跌

    return {
        "ret_1w":       round(ret_1w, 2) if ret_1w is not None else None,
        "ret_1m":       round(ret_1m, 2) if ret_1m is not None else None,
        "vol_ratio_5d": round(vol_ratio_5d, 2) if vol_ratio_5d is not None else None,
        "flow_score":   max(0.0, min(100.0, score)),
    }


def _get_sector(symbol: str) -> str:
    # ETF/指数本身没有 sector 字段（不是公司），yfinance 查询会触发 404 + 内部
    # error 日志。候选标的若恰好是 QQQ/SPY 等大盘 ETF（在默认 universe 中），
    # 提前跳过即可，省一次必失败的网络请求。
    if symbol in _BROAD_ETFS or symbol in _SECTOR_ETF.values() or symbol in _RISK_PROXY.values():
        return ""
    try:
        import yfinance as yf
        return yf.Ticker(symbol).info.get("sector", "") or ""
    except Exception:
        return ""


def _ret(series, n: int) -> Optional[float]:
    if series is None or len(series) < n + 1:
        return None
    return float((series.iloc[-1] / series.iloc[-n - 1] - 1) * 100)


def _avg(lst: List[float]) -> float:
    return sum(lst) / len(lst) if lst else 50.0
