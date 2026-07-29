"""
trader/factors/__init__.py
全局因子注册表。

用法：
    from trader.factors import FACTOR_REGISTRY, get_factor
    factor = get_factor("RSI_14")
    values = factor.compute(df)
"""
from __future__ import annotations

from .base import Factor, FactorMeta
from .technical import (
    ATRRatio, BBPosition, HVRatio, MACDHist, MACross20_60, MACross5_20,
    Momentum21, Momentum63, OBVSlope, RSI14, StochK, VolumeSurge,
)

FACTOR_REGISTRY: dict[str, Factor] = {
    "RSI_14":         RSI14(),
    "MACD_Hist":      MACDHist(),
    "Momentum_21":    Momentum21(),
    "Momentum_63":    Momentum63(),
    "MA_Cross_5_20":  MACross5_20(),
    "MA_Cross_20_60": MACross20_60(),
    "BB_Position":    BBPosition(),
    "Stoch_K":        StochK(),
    "ATR_Ratio":      ATRRatio(),
    "HV_Ratio":       HVRatio(),
    "Volume_Surge":   VolumeSurge(),
    "OBV_Slope":      OBVSlope(),
}


def get_factor(name: str) -> Factor:
    if name not in FACTOR_REGISTRY:
        raise KeyError(f"因子 '{name}' 不存在，可用: {list(FACTOR_REGISTRY)}")
    return FACTOR_REGISTRY[name]


__all__ = ["Factor", "FactorMeta", "FACTOR_REGISTRY", "get_factor"]
