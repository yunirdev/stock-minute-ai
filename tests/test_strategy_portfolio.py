from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from trader.config import TradingConfig
from trader.models import Fill, Side, utc_now
from trader.portfolio import Portfolio
from trader.strategies.registry import build_default_registry
from trader.strategy_core import STRATEGY_OPTIONS, compute_signals


def _sample_df(n: int = 140) -> pd.DataFrame:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(n):
        close = 100.0 + i * 0.1
        rows.append({"timestamp_utc": base + pd.Timedelta(minutes=i), "open": close - 0.2, "high": close + 0.5, "low": close - 0.5, "close": close, "volume": 1000 + i})
    return pd.DataFrame(rows)


def _strategy_containing(text: str) -> str:
    return next(name for name in STRATEGY_OPTIONS if text in name)


def test_default_strategy_registry_matches_existing_compute_signals() -> None:
    df = _sample_df()
    strategy = _strategy_containing("MACD")
    actual = build_default_registry().compute(strategy, df)
    expected = compute_signals(df, strategy)
    pd.testing.assert_frame_equal(actual[["strat_signal", "strat_exec_px"]], expected[["strat_signal", "strat_exec_px"]])


def test_portfolio_equity_includes_position_market_value(tmp_path) -> None:
    portfolio = Portfolio(TradingConfig(initial_capital=10_000.0, db_path=str(tmp_path / "trade.duckdb")))
    portfolio.apply_fill(Fill(order_id="paper-1", intent_id="intent-1", symbol="AAPL", side=Side.BUY, filled_qty=10, avg_price=100.0, fill_time=utc_now()))
    assert portfolio.cash == 9_000.0
    assert portfolio.get_equity({"AAPL": 100.0}) == 10_000.0
    assert portfolio.get_equity({"AAPL": 110.0}) == 10_100.0
