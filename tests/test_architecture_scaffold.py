from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from trader.config import RiskConfig, TradingConfig
from trader.models import Fill, Signal, Side, new_id, utc_now
from trader.order_manager import OrderManager
from trader.portfolio import Portfolio
from trader.strategies.registry import build_default_registry
from trader.strategy_core import compute_signals


def _sample_df(n: int = 140) -> pd.DataFrame:
    rows = []
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(n):
        close = 100.0 + i * 0.1
        rows.append(
            {
                "timestamp_utc": base + pd.Timedelta(minutes=i),
                "open": close - 0.2,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": 1000 + i,
            }
        )
    return pd.DataFrame(rows)


def test_default_strategy_registry_matches_existing_compute_signals() -> None:
    df = _sample_df()
    strategy = "MACD零轴战法"
    registry = build_default_registry()

    expected = compute_signals(df, strategy)
    actual = registry.compute(strategy, df)

    pd.testing.assert_frame_equal(
        actual[["strat_signal", "strat_exec_px"]],
        expected[["strat_signal", "strat_exec_px"]],
    )


def test_pending_order_without_optional_guards_does_not_crash() -> None:
    manager = OrderManager(max_bars_alive=3)
    signal = Signal(
        signal_id=new_id(),
        symbol="AAPL",
        strategy="test",
        side=Side.BUY,
        exec_price=10.0,
        timeframe="5m",
        signal_time=datetime.now(timezone.utc),
        bar_close=12.0,
    )

    fill_now = manager.maybe_enqueue(
        signal=signal,
        bar_open=12.0,
        bar_high=13.0,
        bar_low=11.0,
        bar_close=12.0,
    )
    filled = []

    assert fill_now is False
    manager.check_pending(
        symbol="AAPL",
        bar_high=12.5,
        bar_low=11.5,
        bar_close=12.0,
        on_fill=filled.append,
    )
    assert filled == []


def test_exploration_strategy_wrapper_uses_shared_core() -> None:
    from app.exploration import _build_strategy_signals

    df = _sample_df()
    strategy = "RSI震荡战法(60买40卖)"

    expected = compute_signals(df, strategy)
    actual = _build_strategy_signals(df, strategy)

    pd.testing.assert_frame_equal(
        actual[["strat_signal", "strat_exec_px"]],
        expected[["strat_signal", "strat_exec_px"]],
    )


def test_portfolio_equity_includes_position_market_value(tmp_path) -> None:
    portfolio = Portfolio(TradingConfig(initial_capital=10_000.0, db_path=str(tmp_path / "trade.duckdb")))
    portfolio.apply_fill(
        Fill(
            order_id="paper-1",
            intent_id="intent-1",
            symbol="AAPL",
            side=Side.BUY,
            filled_qty=10,
            avg_price=100.0,
            fill_time=utc_now(),
        )
    )

    assert portfolio.cash == 9_000.0
    assert portfolio.get_equity({"AAPL": 100.0}) == 10_000.0
    assert portfolio.get_equity({"AAPL": 110.0}) == 10_100.0


def test_scheduler_direct_execution_fills_portfolio(tmp_path) -> None:
    """The live path now routes signal → risk → fill via direct calls (no event
    bus). A risk-approved BUY must land a real position in the portfolio."""
    from trader.scheduler import Scheduler

    cfg = TradingConfig(
        symbols=["AAPL"],
        strategies=["全仓买入并持有"],
        initial_capital=10_000.0,
        order_type="MKT",
        data_feed_type="alpaca",   # no network at construction time
        db_path=str(tmp_path / "trade.duckdb"),
        risk=RiskConfig(max_position_pct=1.0, max_trade_risk_pct=1.0),
    )
    sched = Scheduler(cfg)

    signal = Signal(
        signal_id=new_id(), symbol="AAPL", strategy="test", side=Side.BUY,
        exec_price=100.0, timeframe="5m",
        signal_time=datetime.now(timezone.utc), bar_close=100.0,
    )
    sched._execute(signal, equity=10_000.0, prices={"AAPL": 100.0})

    pos = sched.portfolio.positions.get("AAPL")
    assert pos is not None and pos.qty > 0
    assert sched.portfolio.cash < 10_000.0
