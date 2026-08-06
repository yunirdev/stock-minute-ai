"""
trader/ai/agents/options.py
期权市场 Agent — 通过期权链数据评估机构博弈方向。

数据来源：yfinance（免费，实时期权链）
⚠ 局限（请知悉）：
  - 仅 CBOE/市场期权数据，不含场外 OTC 大宗交易
  - 散户期权与机构期权均含在内，PCR 非纯机构信号
  - 临近到期日（< 3 天）IV 失真严重，自动跳过

评分因子（纯算法，无 LLM）：
  ① Put/Call 量比（PCR vol）    30% — < 0.7 偏多，> 1.2 偏空
  ② ATM 隐含波动率（IV）        20% — 低 IV 环境更有利于续涨
  ③ IV Skew（OTM Put vs Call）  25% — 负 skew 表示下行保护需求强
  ④ Max Pain vs 当前价          15% — 最大痛点磁力方向
  ⑤ 期权活跃度（opt/stock ratio）10% — 异常活跃 = 信号强度提升

独立运行，不依赖其他 agent 输出。
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple

from trader.models import AgentContext
from trader.models import Advisory
from .base import AgentBase

logger = logging.getLogger(__name__)

_MIN_EXPIRY_DAYS = 3  # 跳过即将到期（IV 失真）
_MAX_EXPIRY_DAYS = 60  # 只看近月期权


class OptionsAgent(AgentBase):
    """
    期权市场 Agent（纯算法，无 LLM 调用）。
    通过 PCR / IV / Skew / MaxPain 评估市场隐含方向。
    """

    role = "options"

    def __init__(self, client=None) -> None:
        pass  # 不需要 LLM

    def run(self, ctx: AgentContext) -> List[Advisory]:
        advisories: List[Advisory] = []
        for cand in ctx.candidates:
            try:
                adv = self._analyze(cand.symbol)
                if adv:
                    advisories.append(adv)
            except Exception as exc:
                logger.warning("OptionsAgent 跳过 %s: %s", cand.symbol, exc)
        return advisories

    # ── 核心分析 ─────────────────────────────────────────────────────────────

    def _analyze(self, symbol: str) -> Optional[Advisory]:
        import yfinance as yf

        ticker = yf.Ticker(symbol)

        # 选择合适的期权到期日
        expirations = ticker.options
        if not expirations:
            logger.info("OptionsAgent %s: 无期权数据", symbol)
            return None

        expiry = _pick_expiry(expirations)
        if not expiry:
            logger.info("OptionsAgent %s: 无合适到期日", symbol)
            return None

        chain = ticker.option_chain(expiry)
        calls = chain.calls
        puts = chain.puts

        if calls.empty or puts.empty:
            return None

        # 当前价格（用 calls bid/ask 中值区间估算）
        current_price = _estimate_price(ticker, calls)

        # ① PCR by volume
        call_vol = float(calls["volume"].fillna(0).sum())
        put_vol = float(puts["volume"].fillna(0).sum())
        pcr_vol = (put_vol / call_vol) if call_vol > 0 else 1.0

        # ② ATM IV（±5% 价格范围内）
        atm_iv = _atm_iv(calls, puts, current_price, band=0.05)

        # ③ IV Skew（OTM puts 90-97% vs OTM calls 103-110%）
        put_skew = _otm_iv(puts, current_price, lo=0.90, hi=0.97, kind="put")
        call_skew = _otm_iv(calls, current_price, lo=1.03, hi=1.10, kind="call")
        iv_skew = (put_skew - call_skew) if (put_skew and call_skew) else None

        # ④ Max Pain（让期权买方损失最大的行权价）
        max_pain = _calc_max_pain(calls, puts)
        pain_diff_pct = (
            ((max_pain / current_price) - 1) * 100
            if (max_pain and current_price)
            else None
        )

        # ⑤ 期权成交量 vs 正股
        opt_vol_total = call_vol + put_vol
        stock_vol = _stock_volume(ticker)
        opt_ratio = (opt_vol_total / stock_vol) if stock_vol else None

        # 综合打分
        score, factors = _composite(
            pcr_vol=pcr_vol,
            atm_iv=atm_iv,
            iv_skew=iv_skew,
            pain_diff_pct=pain_diff_pct,
            opt_ratio=opt_ratio,
        )
        options_score = self._clamp_score(int(score))
        confidence = min(0.85, 0.4 + len(factors) * 0.09)

        # 市场状态标签
        if pcr_vol < 0.7:
            sentiment = "bullish"
        elif pcr_vol > 1.2:
            sentiment = "bearish"
        else:
            sentiment = "neutral"

        logger.info(
            "OptionsAgent %s: score=%d PCR=%.2f IV=%.1f%% skew=%.1f maxpain=%.1f%%",
            symbol,
            options_score,
            pcr_vol,
            (atm_iv or 0) * 100,
            iv_skew or 0,
            pain_diff_pct or 0,
        )

        return self._advisory(
            kind="options",
            payload={
                "symbol": symbol,
                "options_score": options_score,
                "sentiment": sentiment,
                "expiry": expiry,
                "pcr_vol": round(pcr_vol, 2),
                "atm_iv_pct": round((atm_iv or 0) * 100, 1),
                "iv_skew_pct": round((iv_skew or 0) * 100, 1) if iv_skew else None,
                "max_pain": round(max_pain, 2) if max_pain else None,
                "max_pain_diff_pct": round(pain_diff_pct, 1) if pain_diff_pct else None,
                "opt_ratio": round(opt_ratio, 2) if opt_ratio else None,
                "factors_used": factors,
                "data_note": "PCR/IV含散户期权，仅供参考方向判断",
            },
            confidence=confidence,
            model="algorithmic",
            # 同 quant：没有任何因子参与 = 分数就是初始值，不是真实判断
            is_fallback=not factors,
        )


# ── 辅助函数 ──────────────────────────────────────────────────────────────────


def _pick_expiry(expirations: tuple) -> Optional[str]:
    """选择离现在 3-60 天的最近到期日。"""
    from datetime import date, datetime

    today = date.today()
    for exp in expirations:
        try:
            d = datetime.strptime(exp, "%Y-%m-%d").date()
            delta = (d - today).days
            if _MIN_EXPIRY_DAYS <= delta <= _MAX_EXPIRY_DAYS:
                return exp
        except Exception:
            continue
    return None


def _estimate_price(ticker, calls) -> float:
    """从 ticker.info 或期权 strike 中位数估算当前价。"""
    try:
        p = ticker.info.get("currentPrice") or ticker.info.get("regularMarketPrice")
        if p:
            return float(p)
    except Exception:
        pass
    return float(calls["strike"].median())


def _atm_iv(calls, puts, price: float, band: float = 0.05) -> Optional[float]:
    """ATM 期权的平均 IV（price ± band%）。"""
    try:
        lo, hi = price * (1 - band), price * (1 + band)
        c_iv = calls[(calls["strike"] >= lo) & (calls["strike"] <= hi)][
            "impliedVolatility"
        ].dropna()
        p_iv = puts[(puts["strike"] >= lo) & (puts["strike"] <= hi)][
            "impliedVolatility"
        ].dropna()
        all_iv = list(c_iv) + list(p_iv)
        return float(sum(all_iv) / len(all_iv)) if all_iv else None
    except Exception:
        return None


def _otm_iv(df, price: float, lo: float, hi: float, kind: str) -> Optional[float]:
    """OTM 期权在 [lo, hi] 价格倍数范围的平均 IV。"""
    try:
        col = "strike"
        strike_lo, strike_hi = price * lo, price * hi
        mask = (df[col] >= strike_lo) & (df[col] <= strike_hi)
        iv = df[mask]["impliedVolatility"].dropna()
        return float(iv.mean()) if len(iv) >= 2 else None
    except Exception:
        return None


def _calc_max_pain(calls, puts) -> Optional[float]:
    """计算 Max Pain（让期权卖方损失最小的行权价）。"""
    try:
        all_strikes = sorted(set(calls["strike"].tolist() + puts["strike"].tolist()))
        if not all_strikes:
            return None
        min_pain, pain_strike = float("inf"), all_strikes[0]
        for s in all_strikes:
            call_pain = float(
                calls[calls["strike"] < s]["openInterest"].fillna(0).sum()
                * (s - calls[calls["strike"] < s]["strike"]).clip(lower=0).sum()
                if not calls.empty
                else 0
            )
            put_pain = float(
                puts[puts["strike"] > s]["openInterest"].fillna(0).sum()
                * (puts[puts["strike"] > s]["strike"] - s).clip(lower=0).sum()
                if not puts.empty
                else 0
            )
            total = call_pain + put_pain
            if total < min_pain:
                min_pain, pain_strike = total, s
        return pain_strike
    except Exception:
        return None


def _stock_volume(ticker) -> Optional[float]:
    try:
        v = ticker.info.get("volume") or ticker.info.get("averageVolume")
        return float(v) if v else None
    except Exception:
        return None


def _composite(
    pcr_vol: float,
    atm_iv: Optional[float],
    iv_skew: Optional[float],
    pain_diff_pct: Optional[float],
    opt_ratio: Optional[float],
) -> Tuple[float, List[str]]:
    score = 50.0
    factors: List[str] = []

    # ① PCR（30%）：低 PCR = 多头；高 PCR = 空头
    pcr_pts = _pcr_score(pcr_vol)
    score += pcr_pts * 0.30
    factors.append(f"pcr={pcr_vol:.2f}")

    # ② ATM IV（20%）：低 IV 环境利好续涨
    if atm_iv is not None:
        # IV 20% → neutral；<10% → +10；>50% → -15
        iv_pts = -30 * math.tanh((atm_iv - 0.20) * 4)
        score += iv_pts * 0.20
        factors.append(f"iv={atm_iv:.1%}")

    # ③ IV Skew（25%）：正 skew（puts > calls）= 熊市防御
    if iv_skew is not None:
        skew_pts = -40 * math.tanh(iv_skew * 10)
        score += skew_pts * 0.25
        factors.append(f"skew={iv_skew:+.1%}")

    # ④ Max Pain（15%）：最大痛点高于当前价 = 偏多
    if pain_diff_pct is not None:
        pain_pts = min(20, max(-20, pain_diff_pct * 2))
        score += pain_pts * 0.15
        factors.append(f"maxpain_diff={pain_diff_pct:+.1f}%")

    # ⑤ 期权活跃度（10%）：异常高 = 增加信号权重（不改方向）
    if opt_ratio is not None:
        factors.append(f"opt_ratio={opt_ratio:.2f}x")

    return max(0.0, min(100.0, score)), factors


def _pcr_score(pcr: float) -> float:
    """PCR → [-50, 50]：低 PCR 返回正值（多头）。"""
    if pcr < 0.5:
        return 30
    if pcr < 0.7:
        return 18
    if pcr < 0.9:
        return 8
    if pcr < 1.1:
        return 0
    if pcr < 1.3:
        return -12
    if pcr < 1.6:
        return -22
    return -35
