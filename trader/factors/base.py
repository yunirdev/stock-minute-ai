"""
trader/factors/base.py
因子基类与元数据定义。

每个因子实现两个方法：
  compute(df) → pd.Series   原始因子值（NaN = 数据不足）
  signal(df)  → pd.Series   +1 看多 / -1 看空 / 0 中性（默认基于分位数）
"""
from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FactorMeta:
    name: str
    category: str        # momentum | trend | volatility | volume
    description: str
    lookback: int        # 所需最少 bar 数


class Factor:
    meta: FactorMeta

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """计算因子值序列，长度与 df 相同，数据不足处填 NaN。"""
        ...

    def signal(self, df: pd.DataFrame) -> pd.Series:
        """默认信号：高于 60 分位 → +1，低于 40 分位 → -1，其余 0。"""
        vals = self.compute(df)
        valid = vals.dropna()
        if valid.empty:
            return pd.Series(0, index=vals.index, dtype=int)
        p40 = float(valid.quantile(0.40))
        p60 = float(valid.quantile(0.60))
        sig = pd.Series(0, index=vals.index, dtype=int)
        sig[vals > p60] = 1
        sig[vals < p40] = -1
        return sig.where(vals.notna(), other=0)
