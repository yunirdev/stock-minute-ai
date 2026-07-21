"""
trader/ai/agents/quant.py
量化因子 Agent — 从价格和成交量数据计算统计因子。

数据来源：
  - yfinance 日线数据（免费）：动量 / Beta / 分析师覆盖度
  - DuckDB 本地 bars（已有）：短期 HV / RSI / 量比

无 LLM 调用，纯算法打分。各因子权重：
  动量 (12-1月)  30%  — 趋势延续倾向
  短期动量 (1月) 20%  — 近期价格行为
  波动率制度     20%  — 低 HV 相对均值 = 稳定上涨环境
  RSI 位置       15%  — 技术强弱（避开极端超买超卖）
  量比           15%  — 成交量放大 = 资金参与度

设计原则：独立运行，不依赖其他 agent 的输出。
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional

from trader.contracts import AgentContext
from trader.models import Advisory
from .base import AgentBase

logger = logging.getLogger(__name__)

_MIN_BARS_LOCAL = 30  # DuckDB bars 最少数量
_MIN_DAYS_DAILY = 20  # yfinance 日线最少天数


class QuantAgent(AgentBase):
    """
    量化因子 Agent。
    纯算法打分（无 LLM 调用），速度快，适合高频批量分析。
    """

    role = "quant"

    def __init__(self, client=None) -> None:
        pass  # 无需 LLM client

    def run(self, ctx: AgentContext) -> List[Advisory]:
        # 提前获取 SPY 基准（计算 beta 用）
        spx_daily = _fetch_daily("SPY", "1y")

        advisories: List[Advisory] = []
        for cand in ctx.candidates:
            try:
                adv = self._analyze(cand.symbol, spx_daily)
                if adv:
                    advisories.append(adv)
            except Exception as exc:
                logger.warning("QuantAgent 跳过 %s: %s", cand.symbol, exc)
        return advisories

    # ── 核心分析 ─────────────────────────────────────────────────────────────

    def _analyze(self, symbol: str, spx_daily) -> Optional[Advisory]:
        # ① 日线动量 & Beta（yfinance）
        daily = _fetch_daily(symbol, "1y")
        mom_1m = mom_3m = mom_6m = mom_12m = beta = None
        if daily is not None and len(daily) >= _MIN_DAYS_DAILY:
            mom_1m = _ret(daily, 21)
            mom_3m = _ret(daily, 63)
            mom_6m = _ret(daily, 126)
            # 12-1 月动量（跳过最近 1 月避免短期反转）
            if len(daily) >= 252:
                mom_12m = _ret_range(daily, 252, 21)
            beta = _compute_beta(daily, spx_daily)

        # ② 本地 bars（HV / RSI / 量比）
        hv_ratio = vol_ratio = rsi = None
        try:
            from trader.data_cache import get_bars

            df = get_bars(symbol, "5m")
            if df is not None and len(df) >= _MIN_BARS_LOCAL:
                hv_ratio = _hist_vol_ratio(df)
                vol_ratio = _volume_ratio(df)
                rsi = _rsi(df["close"])
        except Exception:
            pass

        # ③ 综合打分
        score, factors = _composite_score(
            mom_1m=mom_1m,
            mom_3m=mom_3m,
            mom_6m=mom_6m,
            mom_12m=mom_12m,
            hv_ratio=hv_ratio,
            vol_ratio=vol_ratio,
            rsi=rsi,
            beta=beta,
        )

        quant_score = self._clamp_score(int(score))
        confidence = _score_confidence(factors)

        logger.info(
            "QuantAgent %s: score=%d conf=%.2f factors=%s",
            symbol,
            quant_score,
            confidence,
            factors,
        )

        return self._advisory(
            kind="quant",
            payload={
                "symbol": symbol,
                "quant_score": quant_score,
                "momentum_1m_pct": _r2(mom_1m),
                "momentum_3m_pct": _r2(mom_3m),
                "momentum_6m_pct": _r2(mom_6m),
                "momentum_12m1_pct": _r2(mom_12m),
                "beta": _r2(beta),
                "hv_ratio": _r2(hv_ratio),
                "vol_ratio": _r2(vol_ratio),
                "rsi": _r2(rsi),
                "factors_used": factors,
                "note": "Pure algorithmic score — no LLM.",
            },
            confidence=confidence,
            model="algorithmic",
        )


# ── 因子计算函数 ──────────────────────────────────────────────────────────────


def _fetch_daily(symbol: str, period: str = "1y"):
    """从 yfinance 获取日线收盘价 Series。"""
    try:
        import yfinance as yf

        hist = yf.Ticker(symbol).history(period=period, interval="1d", auto_adjust=True)
        if hist.empty:
            return None
        return hist["Close"].dropna()
    except Exception:
        return None


def _ret(series, n: int) -> Optional[float]:
    if series is None or len(series) < n + 1:
        return None
    return float((series.iloc[-1] / series.iloc[-n - 1] - 1) * 100)


def _ret_range(series, start: int, end: int) -> Optional[float]:
    """从 start 天前到 end 天前的收益（12-1月动量）。"""
    if series is None or len(series) < start + 1:
        return None
    return float((series.iloc[-end] / series.iloc[-start - 1] - 1) * 100)


def _compute_beta(sym_daily, spx_daily, n: int = 126) -> Optional[float]:
    """用过去 n 日日收益计算相对 SPY 的 Beta。"""
    if sym_daily is None or spx_daily is None:
        return None
    try:
        import pandas as pd

        s = sym_daily.pct_change().dropna().iloc[-n:]
        m = spx_daily.pct_change().dropna().iloc[-n:]
        combined = pd.concat([s, m], axis=1, join="inner").dropna()
        if len(combined) < 30:
            return None
        cov = combined.iloc[:, 0].cov(combined.iloc[:, 1])
        var = combined.iloc[:, 1].var()
        return round(float(cov / var), 2) if var > 0 else None
    except Exception:
        return None


def _hist_vol_ratio(df) -> Optional[float]:
    """当前 HV（近 20 根 bar）/ 长期 HV（全部 bar），< 1 = 相对平静。"""
    try:
        rets = df["close"].pct_change().dropna()
        if len(rets) < 40:
            return None
        hv_short = float(rets.tail(20).std())
        hv_long = float(rets.std())
        return round(hv_short / hv_long, 2) if hv_long > 0 else None
    except Exception:
        return None


def _volume_ratio(df) -> Optional[float]:
    """近 20 根 bar 均量 / 全局均量。"""
    try:
        if "volume" not in df.columns or len(df) < 40:
            return None
        vol = df["volume"]
        short_avg = float(vol.tail(20).mean())
        long_avg = float(vol.mean())
        return round(short_avg / long_avg, 2) if long_avg > 0 else None
    except Exception:
        return None


def _rsi(close, period: int = 14) -> Optional[float]:
    try:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_g = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_l = loss.ewm(com=period - 1, min_periods=period).mean()
        rs = avg_g / avg_l.replace(0, float("nan"))
        rsi = 100 - (100 / (1 + rs))
        last = rsi.iloc[-1]
        return round(float(last), 1) if not math.isnan(last) else None
    except Exception:
        return None


def _composite_score(
    mom_1m,
    mom_3m,
    mom_6m,
    mom_12m,
    hv_ratio,
    vol_ratio,
    rsi,
    beta,
) -> tuple[float, list]:
    score = 50.0
    factors: List[str] = []

    # ① 12-1 月动量（经典因子，30%）
    if mom_12m is not None:
        contrib = _sigmoid_score(mom_12m, center=0, scale=20) * 0.30
        score = score * 0.70 + (50 + contrib) * 0.30
        factors.append(f"mom_12m={mom_12m:+.1f}%")
    elif mom_3m is not None:
        contrib = _sigmoid_score(mom_3m, center=0, scale=10) * 0.30
        score = score * 0.70 + (50 + contrib) * 0.30
        factors.append(f"mom_3m={mom_3m:+.1f}%(proxy)")

    # ② 短期动量 1m（20%）
    if mom_1m is not None:
        contrib = _sigmoid_score(mom_1m, center=0, scale=8) * 0.20
        score += contrib
        factors.append(f"mom_1m={mom_1m:+.1f}%")

    # ③ 波动率制度（20%）— hv_ratio < 1 稳定，> 1.5 高波
    if hv_ratio is not None:
        vol_pts = -25 * math.tanh((hv_ratio - 1.0) * 2)
        score += vol_pts * 0.20
        factors.append(f"hv_ratio={hv_ratio:.2f}")

    # ④ RSI 位置（15%）
    if rsi is not None:
        if 45 <= rsi <= 65:
            rsi_pts = 10  # sweet spot
        elif 30 <= rsi < 45:
            rsi_pts = 0  # weak but not capitulation
        elif 65 < rsi <= 80:
            rsi_pts = 5  # overbought but trending
        elif rsi > 80:
            rsi_pts = -5  # extreme overbought
        else:
            rsi_pts = -8  # below 30, distress
        score += rsi_pts * 0.15
        factors.append(f"rsi={rsi:.1f}")

    # ⑤ 量比（15%）— 成交量放大表明资金参与
    if vol_ratio is not None:
        vr_pts = min(15, (vol_ratio - 1.0) * 12)
        score += vr_pts * 0.15
        factors.append(f"vol_ratio={vol_ratio:.2f}x")

    return max(0.0, min(100.0, score)), factors


def _sigmoid_score(x: float, center: float, scale: float) -> float:
    """将 x 映射到 [-50, 50]，中心为 center，scale 控制斜率。"""
    z = (x - center) / scale
    return 100 / (1 + math.exp(-z)) - 50


def _score_confidence(factors: List[str]) -> float:
    """因子越多，置信度越高。"""
    return min(0.9, 0.3 + len(factors) * 0.1)


def _r2(v) -> Optional[float]:
    return round(float(v), 2) if v is not None else None
