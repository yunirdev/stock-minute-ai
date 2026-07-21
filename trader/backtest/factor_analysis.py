"""
trader/backtest/factor_analysis.py
单标的因子有效性分析（Alphalens-style 简化版）。

核心指标：
  IC (Information Coefficient)
      Spearman 相关系数(因子值, N期后收益)，衡量因子的方向预测力
      IC > 0.05 通常被视为有意义，ICIR > 0.5 可用
  滚动 IC
      在 rolling_window 个 bar 的窗口上滑动计算 IC，观察因子稳定性
  分位数收益率
      按因子值分 N 组，比较各组的平均前瞻收益率
      单调递增（Q1→QN）说明因子方向性好

单标的分析逻辑：
  - 把历史所有 bar 的因子值按大小排序分为 N 分位
  - Q1=因子值最低的 20%，Q5=最高的 20%
  - 计算各分位内所有 bar 的 forward_period 期后收益均值
  - 如果 Q5 收益显著高于 Q1，说明因子高值对应后续上涨（正向因子）
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats

if TYPE_CHECKING:
    from trader.factors.base import Factor


@dataclass
class FactorAnalysisResult:
    factor_name: str
    symbol: str
    ic_series: pd.Series          # 滚动 IC 时间序列
    ic_mean: float                 # 平均 IC
    ic_std: float                  # IC 标准差
    icir: float                    # IC / std，类信息比率
    quantile_returns: pd.Series    # 各分位组平均前瞻收益率（%），index=Q1..QN
    cumulative_by_quantile: pd.DataFrame  # 各分位组累计收益，列=Q1..QN
    n_valid: int
    forward_period: int
    n_quantiles: int


def run_factor_analysis(
    df: pd.DataFrame,
    factor: "Factor",
    symbol: str = "",
    forward_period: int = 5,
    n_quantiles: int = 5,
    rolling_window: int = 20,
    max_bars: int = 1000,
) -> FactorAnalysisResult:
    """
    对单标的跑因子分析。

    Parameters
    ----------
    df             : OHLCV DataFrame，列 open/high/low/close/volume
    factor         : Factor 实例
    symbol         : 标的代码（仅用于报告显示）
    forward_period : 前瞻期数（bar）
    n_quantiles    : 分组数（默认5，即五分位）
    rolling_window : 滚动 IC 窗口大小（bar）
    max_bars       : 最多使用多少根 bar（太大会变慢）
    """
    if len(df) > max_bars:
        df = df.tail(max_bars).copy()

    # 计算因子值
    factor_vals = factor.compute(df).rename("factor")

    # 前瞻收益率（t 时刻因子 → t+n 时刻收益）
    fwd_ret = df["close"].pct_change(forward_period).shift(-forward_period).rename("fwd_ret")

    combined = pd.concat([factor_vals, fwd_ret], axis=1).dropna()

    min_rows = rolling_window + n_quantiles + forward_period
    if len(combined) < min_rows:
        raise ValueError(
            f"有效数据仅 {len(combined)} 行，需至少 {min_rows} 行。"
            f"请增加 bars_lookback 或缩短 forward_period。"
        )

    f = combined["factor"]
    r = combined["fwd_ret"]

    # ── 滚动 IC ──────────────────────────────────────────────────────────────
    ic_vals: list[float] = []
    ic_idx: list = []
    for i in range(rolling_window, len(f) + 1):
        fw = f.iloc[i - rolling_window:i]
        rw = r.iloc[i - rolling_window:i]
        ic, _ = stats.spearmanr(fw, rw)
        ic_vals.append(float(ic) if not np.isnan(ic) else 0.0)
        ic_idx.append(f.index[i - 1])
    ic_series = pd.Series(ic_vals, index=ic_idx, name="IC")

    ic_mean = float(ic_series.mean())
    ic_std = float(ic_series.std()) or 1e-9

    # ── 分位数收益率 ──────────────────────────────────────────────────────────
    labels = [f"Q{i + 1}" for i in range(n_quantiles)]
    try:
        q_bins = pd.qcut(f, n_quantiles, labels=labels, duplicates="drop")
    except ValueError:
        q_bins = pd.qcut(f.rank(method="first"), n_quantiles, labels=labels, duplicates="drop")

    q_data = pd.concat([q_bins.rename("quantile"), r], axis=1)
    quantile_returns = (
        q_data.groupby("quantile", observed=True)["fwd_ret"].mean() * 100
    )

    # ── 各分位累计收益 ────────────────────────────────────────────────────────
    cum_df = {}
    for lbl in labels:
        mask = q_bins == lbl
        if mask.sum() == 0:
            continue
        rets = r[mask]
        # 把分散的 bar 按时间顺序排列，做累计收益
        cum_df[lbl] = (1 + rets.sort_index()).cumprod() - 1

    cumulative_by_quantile = pd.DataFrame(cum_df) if cum_df else pd.DataFrame()

    return FactorAnalysisResult(
        factor_name=factor.meta.name,
        symbol=symbol,
        ic_series=ic_series,
        ic_mean=ic_mean,
        ic_std=ic_std,
        icir=ic_mean / ic_std,
        quantile_returns=quantile_returns,
        cumulative_by_quantile=cumulative_by_quantile,
        n_valid=len(combined),
        forward_period=forward_period,
        n_quantiles=n_quantiles,
    )
