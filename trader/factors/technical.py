"""
trader/factors/technical.py
12 个纯 TA 因子，全部基于 OHLCV 数据，无外部 API 依赖。

因子列表：
  RSI_14          动量 — RSI(14)，超卖反弹 / 超买回落
  MACD_Hist       动量 — MACD 柱（EMA12-EMA26-Signal9）
  Momentum_21     动量 — 21 bar 价格涨幅 %
  Momentum_63     动量 — 63 bar 价格涨幅 %
  MA_Cross_5_20   趋势 — EMA5/EMA20 差值（归一化）
  MA_Cross_20_60  趋势 — EMA20/EMA60 差值（归一化）
  BB_Position     趋势 — 价格在布林带中的位置 (0=下轨, 1=上轨)
  Stoch_K         趋势 — Stochastic %K(14)
  ATR_Ratio       波动 — 近期 ATR / 历史 ATR（<1=平静, >1=高波）
  HV_Ratio        波动 — 近期 HV / 历史 HV
  Volume_Surge    成交量 — 近 20bar 均量 / 历史均量
  OBV_Slope       成交量 — OBV 10bar 斜率（归一化）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Factor, FactorMeta


# ── 辅助函数 ──────────────────────────────────────────────────────────────────

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_g = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_l = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ── 因子实现 ──────────────────────────────────────────────────────────────────

class RSI14(Factor):
    meta = FactorMeta("RSI_14", "momentum", "RSI(14) 动量强弱，50以上偏多，50以下偏空", 20)

    def compute(self, df: pd.DataFrame) -> pd.Series:
        return _rsi_series(df["close"], 14)

    def signal(self, df: pd.DataFrame) -> pd.Series:
        rsi = self.compute(df)
        sig = pd.Series(0, index=rsi.index, dtype=int)
        sig[rsi > 55] = 1
        sig[rsi < 45] = -1
        return sig.where(rsi.notna(), other=0)


class MACDHist(Factor):
    meta = FactorMeta("MACD_Hist", "momentum", "MACD 柱（差离值），正值多头动能，负值空头动能", 35)

    def compute(self, df: pd.DataFrame) -> pd.Series:
        macd = _ema(df["close"], 12) - _ema(df["close"], 26)
        signal = _ema(macd, 9)
        return macd - signal


class Momentum21(Factor):
    meta = FactorMeta("Momentum_21", "momentum", "21 bar 价格涨幅 %（约一个月）", 25)

    def compute(self, df: pd.DataFrame) -> pd.Series:
        return df["close"].pct_change(21) * 100


class Momentum63(Factor):
    meta = FactorMeta("Momentum_63", "momentum", "63 bar 价格涨幅 %（约一个季度）", 70)

    def compute(self, df: pd.DataFrame) -> pd.Series:
        return df["close"].pct_change(63) * 100


class MACross5_20(Factor):
    meta = FactorMeta("MA_Cross_5_20", "trend", "EMA5/EMA20 差值（归一化为价格 %）", 25)

    def compute(self, df: pd.DataFrame) -> pd.Series:
        diff = _ema(df["close"], 5) - _ema(df["close"], 20)
        return diff / df["close"] * 100


class MACross20_60(Factor):
    meta = FactorMeta("MA_Cross_20_60", "trend", "EMA20/EMA60 差值（归一化为价格 %）", 65)

    def compute(self, df: pd.DataFrame) -> pd.Series:
        diff = _ema(df["close"], 20) - _ema(df["close"], 60)
        return diff / df["close"] * 100


class BBPosition(Factor):
    meta = FactorMeta("BB_Position", "trend", "布林带内位置 0=下轨 1=上轨", 22)

    def compute(self, df: pd.DataFrame) -> pd.Series:
        mid = df["close"].rolling(20).mean()
        std = df["close"].rolling(20).std()
        upper = mid + 2 * std
        lower = mid - 2 * std
        band = upper - lower
        return ((df["close"] - lower) / band).clip(0, 1)


class StochK(Factor):
    meta = FactorMeta("Stoch_K", "trend", "Stochastic %K(14)，衡量价格在区间内的位置", 16)

    def compute(self, df: pd.DataFrame) -> pd.Series:
        low14 = df["low"].rolling(14).min()
        high14 = df["high"].rolling(14).max()
        rng = high14 - low14
        return ((df["close"] - low14) / rng * 100).where(rng > 0)


class ATRRatio(Factor):
    meta = FactorMeta("ATR_Ratio", "volatility", "近期ATR / 历史ATR，<1=低波，>1=高波", 40)

    def compute(self, df: pd.DataFrame) -> pd.Series:
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ], axis=1).max(axis=1)
        atr_short = tr.rolling(10).mean()
        atr_long = tr.rolling(30).mean()
        return atr_short / atr_long.replace(0, np.nan)

    def signal(self, df: pd.DataFrame) -> pd.Series:
        ratio = self.compute(df)
        sig = pd.Series(0, index=ratio.index, dtype=int)
        sig[ratio < 0.85] = 1    # 低波 = 环境好，偏多
        sig[ratio > 1.30] = -1   # 高波 = 风险升，偏空
        return sig.where(ratio.notna(), other=0)


class HVRatio(Factor):
    meta = FactorMeta("HV_Ratio", "volatility", "近期历史波动率 / 长期历史波动率", 50)

    def compute(self, df: pd.DataFrame) -> pd.Series:
        rets = df["close"].pct_change()
        hv_short = rets.rolling(20).std()
        hv_long = rets.rolling(40).std()
        return hv_short / hv_long.replace(0, np.nan)


class VolumeSurge(Factor):
    meta = FactorMeta("Volume_Surge", "volume", "近20bar均量 / 历史均量，>1=量能放大", 45)

    def compute(self, df: pd.DataFrame) -> pd.Series:
        if "volume" not in df.columns:
            return pd.Series(np.nan, index=df.index)
        vol = df["volume"].replace(0, np.nan)
        short_avg = vol.rolling(20).mean()
        long_avg = vol.rolling(40).mean()
        return short_avg / long_avg.replace(0, np.nan)


class OBVSlope(Factor):
    meta = FactorMeta("OBV_Slope", "volume", "OBV 10bar 斜率（归一化），衡量资金流向趋势", 15)

    def compute(self, df: pd.DataFrame) -> pd.Series:
        if "volume" not in df.columns:
            return pd.Series(np.nan, index=df.index)
        direction = np.sign(df["close"].diff())
        obv = (direction * df["volume"]).fillna(0).cumsum()
        slope = obv.diff(10) / (df["close"] * 10).replace(0, np.nan)
        return slope
