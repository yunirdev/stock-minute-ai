"""
trader/teams/base.py
团队框架公共数据结构。

MarketRegime  — T0 输出的市场环境枚举
TeamOutput    — 所有团队统一输出结构，存入 DuckDB 供 T5 维护团队消费
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class RegimeType(str, Enum):
    BULL_TREND  = "bull_trend"    # 趋势多头：SPY > 200MA，VIX 低
    BEAR_TREND  = "bear_trend"    # 趋势空头：SPY < 200MA
    HIGH_VOL    = "high_vol"      # 高波动：VIX > 25
    NEUTRAL     = "neutral"       # 震荡中性


@dataclass
class MarketRegime:
    regime: RegimeType
    vix: Optional[float]
    spy_vs_200ma_pct: Optional[float]   # SPY 偏离 200MA 的百分比，正=上方
    spy_price: Optional[float]
    ma200: Optional[float]
    confidence: float                    # 0-1
    notes: str
    as_of: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # 方便读取的属性
    @property
    def label(self) -> str:
        return {
            RegimeType.BULL_TREND: "趋势多头",
            RegimeType.BEAR_TREND: "趋势空头",
            RegimeType.HIGH_VOL:   "高波动",
            RegimeType.NEUTRAL:    "震荡中性",
        }[self.regime]

    @property
    def color(self) -> str:
        return {
            RegimeType.BULL_TREND: "#3fb950",
            RegimeType.BEAR_TREND: "#f85149",
            RegimeType.HIGH_VOL:   "#d29922",
            RegimeType.NEUTRAL:    "#8b949e",
        }[self.regime]


@dataclass
class TeamOutput:
    team: str                                          # "T0" | "T1" | ...
    status: str = "ok"                                 # "ok" | "error" | "skipped"
    data: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    duration_ms: float = 0.0
    as_of: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.status = "error"
