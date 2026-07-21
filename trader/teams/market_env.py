"""
trader/teams/market_env.py
T0 市场环境团队 — 每日开盘前跑一次，判断当前市场所处阶段。

判断逻辑（优先级递减）：
  1. VIX > 25 → HIGH_VOL（不管方向，先管风险）
  2. SPY > 200MA → 看多方向（BULL_TREND）
  3. SPY < 200MA → 看空方向（BEAR_TREND）
  4. 其余 → NEUTRAL

数据来源：yfinance（免费，无需 API Key）
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .base import MarketRegime, RegimeType, TeamOutput

logger = logging.getLogger(__name__)

_VIX_HIGH_THRESHOLD    = 25.0
_VIX_EXTREME_THRESHOLD = 35.0

_REGIME_CACHE_FILE = Path(__file__).resolve().parents[2] / "conf" / "market_regime.json"


def write_regime_cache(regime: MarketRegime) -> None:
    """把 MarketRegime 序列化到 conf/market_regime.json，供 runtime.py 消费。"""
    try:
        _REGIME_CACHE_FILE.write_text(
            json.dumps({
                "regime": regime.regime.value,
                "vix": regime.vix,
                "spy_vs_200ma_pct": regime.spy_vs_200ma_pct,
                "confidence": regime.confidence,
                "notes": regime.notes,
                "as_of": regime.as_of.isoformat(),
            }),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("write_regime_cache 失败: %s", exc)


def read_regime_cache() -> Optional[MarketRegime]:
    """从 conf/market_regime.json 读取上次计算的 MarketRegime。文件不存在则返回 None。"""
    try:
        if not _REGIME_CACHE_FILE.exists():
            return None
        data = json.loads(_REGIME_CACHE_FILE.read_text(encoding="utf-8"))
        return MarketRegime(
            regime=RegimeType(data["regime"]),
            vix=data.get("vix"),
            spy_vs_200ma_pct=data.get("spy_vs_200ma_pct"),
            spy_price=None,
            ma200=None,
            confidence=data.get("confidence", 0.5),
            notes=data.get("notes", ""),
            as_of=datetime.fromisoformat(data["as_of"]),
        )
    except Exception as exc:
        logger.warning("read_regime_cache 失败: %s", exc)
        return None


def run_market_env() -> TeamOutput:
    """运行 T0 分析，返回 TeamOutput，data["regime"] 为 MarketRegime。"""
    out = TeamOutput(team="T0")
    t0 = time.time()
    try:
        regime = _compute_regime()
        out.data["regime"] = regime
        out.data["regime_label"] = regime.label
        out.data["vix"] = regime.vix
        out.data["spy_vs_200ma_pct"] = regime.spy_vs_200ma_pct
        out.status = "ok"
        write_regime_cache(regime)
        logger.info(
            "T0 市场环境: %s (VIX=%.1f, SPY vs 200MA=%.2f%%)",
            regime.label,
            regime.vix or 0,
            regime.spy_vs_200ma_pct or 0,
        )
    except Exception as exc:
        out.add_error(str(exc))
        logger.warning("T0 market_env 失败: %s", exc)
    finally:
        out.duration_ms = (time.time() - t0) * 1000
    return out


def _compute_regime() -> MarketRegime:
    """从 yfinance 获取 SPY + VIX，输出 MarketRegime。"""
    vix = _fetch_vix()
    spy_price, ma200 = _fetch_spy_ma200()

    notes_parts: list[str] = []

    # 计算 SPY 偏离 200MA 的百分比
    spy_vs_200ma_pct: Optional[float] = None
    if spy_price and ma200 and ma200 > 0:
        spy_vs_200ma_pct = round((spy_price - ma200) / ma200 * 100, 2)
        notes_parts.append(f"SPY={spy_price:.2f} 200MA={ma200:.2f} 偏离={spy_vs_200ma_pct:+.2f}%")

    if vix:
        notes_parts.append(f"VIX={vix:.1f}")

    # 判断 Regime
    if vix and vix > _VIX_EXTREME_THRESHOLD:
        regime = RegimeType.HIGH_VOL
        confidence = 0.90
        notes_parts.append(f"VIX>{_VIX_EXTREME_THRESHOLD} 极端恐慌")
    elif vix and vix > _VIX_HIGH_THRESHOLD:
        regime = RegimeType.HIGH_VOL
        confidence = 0.80
        notes_parts.append(f"VIX>{_VIX_HIGH_THRESHOLD} 高波动")
    elif spy_vs_200ma_pct is not None:
        if spy_vs_200ma_pct > 0:
            regime = RegimeType.BULL_TREND
            confidence = min(0.90, 0.60 + abs(spy_vs_200ma_pct) * 0.02)
        else:
            regime = RegimeType.BEAR_TREND
            confidence = min(0.90, 0.60 + abs(spy_vs_200ma_pct) * 0.02)
    else:
        regime = RegimeType.NEUTRAL
        confidence = 0.40
        notes_parts.append("数据不足，默认中性")

    return MarketRegime(
        regime=regime,
        vix=vix,
        spy_vs_200ma_pct=spy_vs_200ma_pct,
        spy_price=spy_price,
        ma200=ma200,
        confidence=confidence,
        notes=" | ".join(notes_parts),
        as_of=datetime.now(timezone.utc),
    )


def _fetch_vix() -> Optional[float]:
    try:
        import yfinance as yf
        vix = yf.Ticker("^VIX").history(period="2d", interval="1d", auto_adjust=True)
        if vix.empty:
            return None
        return float(vix["Close"].iloc[-1])
    except Exception as exc:
        logger.warning("获取 VIX 失败: %s", exc)
        return None


def _fetch_spy_ma200() -> tuple[Optional[float], Optional[float]]:
    try:
        import yfinance as yf
        spy = yf.Ticker("SPY").history(period="1y", interval="1d", auto_adjust=True)
        if spy.empty or len(spy) < 10:
            return None, None
        price = float(spy["Close"].iloc[-1])
        ma200 = float(spy["Close"].tail(200).mean()) if len(spy) >= 200 else float(spy["Close"].mean())
        return price, ma200
    except Exception as exc:
        logger.warning("获取 SPY 失败: %s", exc)
        return None, None
