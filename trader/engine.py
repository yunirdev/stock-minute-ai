"""
trader/engine.py
=================
单一回测/模拟引擎 —— 整个项目唯一的成交与盈亏计算入口。

设计目标(取代旧的两套引擎:exploration 向量化引擎 + runtime 事件驱动引擎):
  - 输入是一个带 strat_signal / strat_exec_px 列的 DataFrame。
    任何"信号来源"(现在的 TA 策略、未来的 AI 模型)只要产出这两列,
    就能喂进同一个 simulate()。这就是 SignalSource 接口。
  - 默认"下一根开盘成交",消除前视偏差(用当根收盘决策、却在当根收盘成交是不现实的)。
  - 自包含的精简账本(现金 + 持仓),不碰 DuckDB,所以足够快(UI 可反复调用)。
    持久化(写 trade.duckdb)由实盘层单独负责,不在引擎里。
  - 风控熔断可配置,默认关闭(回测看原始表现);启用时按真实日界重置
    —— 修掉旧引擎"相对初始本金跌 3% 就永久停手"的 bug。

账本数学与旧 Portfolio 一致:
  BUY  : cash -= qty*px + fee ; 加权平均建仓
  SELL : cash += qty*px - fee ; 已实现盈亏 = (px - avg_entry) * qty
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

DEFAULT_CAPITAL = 10_000.0


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    bar_index: int
    time: object
    side: str            # "BUY" | "SELL"
    qty: float
    price: float
    fee: float
    realized_pnl: float  # 仅平仓腿非零
    equity_after: float
    ret: float = 0.0     # 单笔回合收益率(未杠杆),仅平仓腿非零


@dataclass
class SimResult:
    final_equity: float
    cash: float
    position_qty: float
    avg_entry: float
    initial_capital: float
    equity_curve: pd.Series          # index = timestamp_utc, value = 每根 mark-to-market 权益
    trades: List[Trade] = field(default_factory=list)

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def n_round_trips(self) -> int:
        """完整平仓次数(有 realized_pnl 的腿)。"""
        return sum(1 for t in self.trades if t.realized_pnl != 0.0 or t.side == "SELL")

    @property
    def total_return(self) -> float:
        if self.initial_capital <= 0:
            return 0.0
        return self.final_equity / self.initial_capital - 1.0

    @property
    def wins(self) -> int:
        return sum(1 for t in self.trades if t.realized_pnl > 0)

    @property
    def closed_trades(self) -> int:
        return sum(1 for t in self.trades if t.realized_pnl != 0.0)

    @property
    def win_rate(self) -> float:
        c = self.closed_trades
        return (self.wins / c) if c > 0 else float("nan")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _finite_px(*candidates: float) -> Optional[float]:
    """返回第一个有限且为正的价格,全部无效则 None。集中处理价格校验。"""
    for px in candidates:
        try:
            v = float(px)
        except (TypeError, ValueError):
            continue
        if np.isfinite(v) and v > 0:
            return v
    return None


def _day_key(ts) -> object:
    try:
        return pd.Timestamp(ts).tz_convert("UTC").date()
    except Exception:
        try:
            return pd.Timestamp(ts).date()
        except Exception:
            return None


# ---------------------------------------------------------------------------
# The single simulation engine
# ---------------------------------------------------------------------------

def simulate(
    df: pd.DataFrame,
    *,
    capital: float = DEFAULT_CAPITAL,
    leverage: float = 1.0,
    allow_short: bool = False,
    fee_bps: float = 0.0,
    max_position_pct: float = 1.0,
    fill: str = "next_open",        # "next_open"(诚实,默认) | "close"(对照旧引擎)
    risk_halt: bool = False,        # 是否启用日内回撤熔断
    daily_dd_limit: float = 0.03,   # 熔断阈值(按真实日界重置)
    signal_col: str = "strat_signal",
    exec_col: str = "strat_exec_px",
) -> SimResult:
    """对单标的、单信号序列做逐根回测。

    df 需含列: timestamp_utc, open, high, low, close, strat_signal, strat_exec_px。
    strat_signal: +1 买入/做多, -1 卖出/做空, 0 持有。
    返回 SimResult(权益曲线 + 成交明细)。
    """
    capital = float(capital)
    n = len(df)
    if n == 0:
        empty = pd.Series(dtype=float)
        return SimResult(capital, capital, 0.0, 0.0, capital, empty, [])

    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=float)
    open_ = pd.to_numeric(df["open"], errors="coerce").to_numpy(dtype=float)
    ts = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
    sig = pd.to_numeric(df[signal_col], errors="coerce").fillna(0).astype(int).to_numpy() \
        if signal_col in df.columns else np.zeros(n, dtype=int)
    exec_px = pd.to_numeric(df[exec_col], errors="coerce").to_numpy(dtype=float) \
        if exec_col in df.columns else np.full(n, np.nan)

    L = max(float(leverage), 1.0)
    fee_rate = max(float(fee_bps), 0.0) / 10_000.0
    use_next_open = (fill != "close")

    cash = capital
    pos_qty = 0.0          # 带符号: >0 多, <0 空
    avg_entry = 0.0
    realized_total = 0.0
    trades: List[Trade] = []
    equity_curve = np.empty(n, dtype=float)

    # 风控熔断状态(按日重置)
    halted = False
    day_start_equity: Optional[float] = None
    cur_day: object = None

    pending: Optional[int] = None   # 上一根决定、待本根开盘执行的信号方向

    def _equity_at(price: float) -> float:
        return cash + pos_qty * price

    def _execute(direction: int, price: float, bar_idx: int) -> None:
        nonlocal cash, pos_qty, avg_entry, realized_total
        px = _finite_px(price)
        if px is None:
            return
        # 目标仓位规模: max_position_pct * equity / price * leverage(整股)
        equity_now = _equity_at(px)
        target_qty = int(max(equity_now, 0.0) * max_position_pct / px * L)
        if target_qty < 1 and pos_qty == 0:
            return

        if direction > 0:
            if pos_qty < 0:                      # 先回补空头
                _close_position(px, bar_idx)
            if pos_qty == 0 and target_qty >= 1:  # 再开多
                _open_position(target_qty, px, bar_idx, side="BUY")
        elif direction < 0:
            if pos_qty > 0:                      # 先平多头
                _close_position(px, bar_idx)
            if pos_qty == 0 and allow_short and target_qty >= 1:  # 再开空
                _open_position(target_qty, px, bar_idx, side="SELL")

    def _open_position(qty: int, px: float, bar_idx: int, side: str) -> None:
        nonlocal cash, pos_qty, avg_entry
        fee = qty * px * fee_rate
        signed = qty if side == "BUY" else -qty
        cash -= signed * px        # 买:现金减少; 卖空:现金增加
        cash -= fee
        pos_qty = float(signed)
        avg_entry = px
        trades.append(Trade(bar_idx, ts.iloc[bar_idx], side, float(qty), px, fee,
                            0.0, _equity_at(px)))

    def _close_position(px: float, bar_idx: int) -> None:
        nonlocal cash, pos_qty, avg_entry, realized_total
        qty = abs(pos_qty)
        if qty < 1:
            pos_qty = 0.0
            return
        fee = qty * px * fee_rate
        side = "SELL" if pos_qty > 0 else "BUY"   # 平多=卖, 平空=买
        pnl = (px - avg_entry) * pos_qty          # pos_qty 带符号,空头时 (px-entry)*(-qty) 正确
        ret = (px / avg_entry - 1.0) * (1.0 if pos_qty > 0 else -1.0) if avg_entry > 0 else 0.0
        cash += pos_qty * px                       # 平多:现金增加; 平空:现金减少(回补)
        cash -= fee
        realized_total += pnl
        trades.append(Trade(bar_idx, ts.iloc[bar_idx], side, float(qty), px, fee,
                            float(pnl), _equity_at(px), float(ret)))
        pos_qty = 0.0
        avg_entry = 0.0

    for i in range(n):
        # 1. 执行上一根决定的挂单(下一根开盘成交)
        if pending is not None:
            fill_px = _finite_px(open_[i] if use_next_open else exec_px[i], close[i])
            if fill_px is not None:
                _execute(pending, fill_px, i)
            pending = None

        # 2. 本根 mark-to-market 权益
        bar_close = _finite_px(close[i]) or (avg_entry if pos_qty else 0.0)
        equity = _equity_at(bar_close)
        equity_curve[i] = equity

        # 3. 风控熔断(可选,按真实日界重置)
        if risk_halt:
            d = _day_key(ts.iloc[i])
            if d != cur_day:
                cur_day = d
                day_start_equity = equity
                halted = False
            if day_start_equity and day_start_equity > 0:
                if (equity - day_start_equity) / day_start_equity <= -daily_dd_limit:
                    halted = True

        # 4. 读取本根信号,登记到下一根执行
        s = int(sig[i])
        if s != 0 and not halted:
            if fill == "close":
                # 对照模式:当根收盘/exec_px 直接成交
                px = _finite_px(exec_px[i], close[i])
                if px is not None:
                    _execute(s, px, i)
                    equity_curve[i] = _equity_at(_finite_px(close[i]) or px)
            else:
                pending = s

    final_px = _finite_px(close[-1]) or avg_entry or 0.0
    final_equity = _equity_at(final_px)
    curve = pd.Series(equity_curve, index=ts, name="equity")
    return SimResult(
        final_equity=float(final_equity),
        cash=float(cash),
        position_qty=float(pos_qty),
        avg_entry=float(avg_entry),
        initial_capital=capital,
        equity_curve=curve,
        trades=trades,
    )
