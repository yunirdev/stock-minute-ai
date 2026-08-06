"""做空支持——之前只有 risk_engine 的 allow_short 一道闸门，真正的持仓记账
（Portfolio）和持仓计划链（PositionPlanFillProjector）都是长仓专用设计：
Portfolio 在开新空单/穿仓反向时会直接把仓位记录丢掉，PositionPlanFillProjector
在没有已有仓位时收到 SELL 成交会直接抛异常崩掉。这里补全两边的多空对称逻辑，
以及 plan.py 的开平仓动作判定、risk_engine 的 OPEN 防重复检查。
"""
from datetime import datetime, timedelta, timezone

import pytest

from trader.config import RiskConfig, TradingConfig
from trader.models import (
    Candidate,
    Bar,
    Fill,
    Position,
    PositionPlanStatus,
    RiskVerdict,
    Side,
    TradePlan,
)
from trader.plan import ATRPlanner
from trader.portfolio import Portfolio
from trader.position_plans import PositionPlanFillProjector, PositionPlanStore
from trader.risk_engine import RiskEngine

NOW = datetime(2026, 8, 6, 15, 0, tzinfo=timezone.utc)


def _fill(*, order_id, side, cumulative, price, symbol="AAPL", minute=0) -> Fill:
    return Fill(
        order_id=order_id, intent_id=f"intent-{order_id}", symbol=symbol,
        side=side, filled_qty=cumulative, avg_price=price,
        fill_time=NOW + timedelta(minutes=minute),
    )


# ---------------------------------------------------------------------------
# Portfolio：开新仓/穿仓反向不再丢记录
# ---------------------------------------------------------------------------

def test_portfolio_opens_short_from_flat(tmp_path):
    portfolio = Portfolio(TradingConfig(initial_capital=10_000.0, db_path=str(tmp_path / "trade.duckdb")))
    portfolio.apply_fill(_fill(order_id="s1", side=Side.SELL, cumulative=10, price=100.0))

    pos = portfolio.positions["AAPL"]
    assert pos.qty == -10.0
    assert pos.avg_entry_px == 100.0
    # 卖空收到现金，跟平多收到现金是同一套公式。
    assert portfolio.cash == 11_000.0


def test_portfolio_adds_to_existing_short(tmp_path):
    portfolio = Portfolio(TradingConfig(initial_capital=10_000.0, db_path=str(tmp_path / "trade.duckdb")))
    portfolio.apply_fill(_fill(order_id="s1", side=Side.SELL, cumulative=10, price=100.0))
    portfolio.apply_fill(_fill(order_id="s2", side=Side.SELL, cumulative=10, price=110.0, minute=1))

    pos = portfolio.positions["AAPL"]
    assert pos.qty == -20.0
    assert pos.avg_entry_px == pytest.approx(105.0)  # (100*10 + 110*10) / 20


def test_portfolio_partial_cover_realizes_pnl_and_keeps_remainder(tmp_path):
    portfolio = Portfolio(TradingConfig(initial_capital=10_000.0, db_path=str(tmp_path / "trade.duckdb")))
    portfolio.apply_fill(_fill(order_id="s1", side=Side.SELL, cumulative=10, price=100.0))
    # 空头在价格跌到 90 时回补 4 股——盈利 (100-90)*4 = 40。
    portfolio.apply_fill(_fill(order_id="b1", side=Side.BUY, cumulative=4, price=90.0, minute=1))

    pos = portfolio.positions["AAPL"]
    assert pos.qty == -6.0
    assert pos.avg_entry_px == 100.0  # 剩余部分成本价不变
    assert pos.realized_pnl == pytest.approx(40.0)
    assert portfolio.realized_pnl == pytest.approx(40.0)


def test_portfolio_full_cover_closes_short(tmp_path):
    portfolio = Portfolio(TradingConfig(initial_capital=10_000.0, db_path=str(tmp_path / "trade.duckdb")))
    portfolio.apply_fill(_fill(order_id="s1", side=Side.SELL, cumulative=10, price=100.0))
    portfolio.apply_fill(_fill(order_id="b1", side=Side.BUY, cumulative=10, price=90.0, minute=1))

    assert "AAPL" not in portfolio.positions
    assert portfolio.realized_pnl == pytest.approx(100.0)  # (100-90)*10


def test_portfolio_reversal_crossing_zero_opens_opposite_position(tmp_path):
    """空头 10 股被一笔 15 股的买单回补——前 10 股结清空头盈亏，剩下 5 股
    是全新的多头仓位，成本价是这笔成交价，不是原来空头的成本价。原来的
    实现会在 qty 变成正数之前就因为 `qty<=0` 判断把记录删掉，穿仓之后的
    多头仓位凭空消失。"""
    portfolio = Portfolio(TradingConfig(initial_capital=10_000.0, db_path=str(tmp_path / "trade.duckdb")))
    portfolio.apply_fill(_fill(order_id="s1", side=Side.SELL, cumulative=10, price=100.0))
    portfolio.apply_fill(_fill(order_id="b1", side=Side.BUY, cumulative=15, price=90.0, minute=1))

    pos = portfolio.positions["AAPL"]
    assert pos.qty == 5.0
    assert pos.avg_entry_px == 90.0
    # 新仓位是刚开的，自己的 realized_pnl 从 0 起算（跟"平仓后重新开仓"是
    # 同一套语义）；100 记在账户级别的累计已实现盈亏上，不是这张新仓位上。
    assert pos.realized_pnl == 0.0
    assert portfolio.realized_pnl == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# PositionPlanFillProjector：开空单不再崩，加空/回补走对称路径
# ---------------------------------------------------------------------------

def _short_trade_plan() -> TradePlan:
    return TradePlan(
        plan_id="trade-plan-short", symbol="AAPL", side=Side.SELL, action="OPEN",
        entry_price=100, stop_loss=105, take_profit=90, qty=10,
    )


def test_projector_opens_short_position_plan(tmp_path):
    store = PositionPlanStore(tmp_path / "trade.duckdb")
    projector = PositionPlanFillProjector(store)

    plan = projector.apply(
        fill=_fill(order_id="s1", side=Side.SELL, cumulative=10, price=100.0),
        applied_delta=10,
        trade_plan=_short_trade_plan(),
    )

    assert plan.side == Side.SELL
    assert plan.open_quantity == 10
    assert plan.status == PositionPlanStatus.ACTIVE


def test_projector_adds_to_short_then_covers_it(tmp_path):
    store = PositionPlanStore(tmp_path / "trade.duckdb")
    projector = PositionPlanFillProjector(store)
    projector.apply(
        fill=_fill(order_id="s1", side=Side.SELL, cumulative=10, price=100.0),
        applied_delta=10,
        trade_plan=_short_trade_plan(),
    )
    added = projector.apply(
        fill=_fill(order_id="s2", side=Side.SELL, cumulative=5, price=98.0, minute=1),
        applied_delta=5,
        trade_plan=None,
    )
    assert added.open_quantity == 15
    assert added.status == PositionPlanStatus.ACTIVE

    covered = projector.apply(
        fill=_fill(order_id="b1", side=Side.BUY, cumulative=15, price=90.0, minute=2),
        applied_delta=15,
        trade_plan=None,
    )
    assert covered.open_quantity == 0
    assert covered.status == PositionPlanStatus.CLOSED


def test_projector_rejects_side_mismatch_on_open(tmp_path):
    """trade_plan 说要开多（side=BUY），但成交回报是 SELL——两者对不上，
    不能悄悄按 SELL 开一张空单出来，得原地报错。"""
    store = PositionPlanStore(tmp_path / "trade.duckdb")
    projector = PositionPlanFillProjector(store)
    long_plan = TradePlan(
        plan_id="trade-plan-long", symbol="AAPL", side=Side.BUY, action="OPEN",
        entry_price=100, stop_loss=95, take_profit=115, qty=10,
    )
    with pytest.raises(ValueError, match="POSITION_PLAN_INITIAL_FILL_SIDE_INVALID"):
        projector.apply(
            fill=_fill(order_id="s1", side=Side.SELL, cumulative=10, price=100.0),
            applied_delta=10,
            trade_plan=long_plan,
        )


# ---------------------------------------------------------------------------
# plan.py：开平仓动作判定不再把"已有空头"和"空仓"混成同一个 OPEN
# ---------------------------------------------------------------------------

def _bar(close: float) -> Bar:
    return Bar(symbol="AAPL", timestamp=NOW, open=close, high=close + 1,
               low=close - 1, close=close, volume=1000)


def _candidate() -> Candidate:
    return Candidate(symbol="AAPL", score=80.0, rank=1, reasons={}, as_of=NOW)


def test_sell_signal_adds_to_existing_short_not_reopens():
    planner = ATRPlanner()
    plan = planner.make_plan(
        _candidate(), _bar(95.0), side=Side.SELL, current_qty=-10.0,
    )
    assert plan.action == "ADD"


def test_sell_signal_reduces_existing_long():
    planner = ATRPlanner()
    plan = planner.make_plan(
        _candidate(), _bar(95.0), side=Side.SELL, current_qty=10.0,
    )
    assert plan.action == "REDUCE"


def test_buy_signal_reduces_existing_short():
    planner = ATRPlanner()
    plan = planner.make_plan(
        _candidate(), _bar(95.0), side=Side.BUY, current_qty=-10.0,
    )
    assert plan.action == "REDUCE"


def test_buy_signal_opens_from_flat():
    planner = ATRPlanner()
    plan = planner.make_plan(
        _candidate(), _bar(95.0), side=Side.BUY, current_qty=0.0,
    )
    assert plan.action == "OPEN"


# ---------------------------------------------------------------------------
# risk_engine：OPEN 防重复现在也拦已有空头
# ---------------------------------------------------------------------------

def test_risk_engine_rejects_open_when_already_short():
    config = TradingConfig(risk=RiskConfig(allow_short=True))
    engine = RiskEngine(config)
    plan = TradePlan(
        plan_id="p1", symbol="AAPL", side=Side.SELL, action="OPEN",
        entry_price=100, stop_loss=105, take_profit=90, qty=5,
    )
    verdict = engine.evaluate_plan(
        plan, 100_000.0,
        {"AAPL": Position(symbol="AAPL", qty=-10.0, avg_entry_px=100.0)},
    )
    assert not verdict.approved
    assert "已有持仓" in verdict.reason
