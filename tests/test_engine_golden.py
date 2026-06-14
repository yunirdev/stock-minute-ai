"""
Golden / characterization test for the unified engine (trader/engine.simulate).

This is the lasting regression guard: it pins simulate()'s output on deterministic
synthetic data so future changes can't silently alter backtest results.

History: the project previously had two divergent engines (a vectorized one in
exploration.py and an event-driven one in runtime/). They disagreed by up to ~39%
of capital on identical data. Both were replaced by trader/engine.simulate(); the
UI backtest now calls it through thin adapters in app/exploration.py.

Hermetic: seeded random walk, no dependency on data/bars cache files.
"""
import logging

import numpy as np
import pandas as pd
import pytest

logging.disable(logging.CRITICAL)

CAPITAL = 10_000.0


def make_synthetic_bars(n=600, seed=42):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0002, 0.01, n)
    close = 100.0 * np.cumprod(1.0 + rets)
    high = close * (1.0 + np.abs(rng.normal(0, 0.004, n)))
    low = close * (1.0 - np.abs(rng.normal(0, 0.004, n)))
    open_ = np.concatenate([[100.0], close[:-1]])
    vol = rng.integers(1_000, 100_000, n).astype(float)
    ts = pd.date_range("2024-01-01", periods=n, freq="30min", tz="UTC")
    return pd.DataFrame({
        "timestamp_utc": ts, "timestamp": ts,
        "open": open_, "high": high, "low": low, "close": close, "volume": vol,
    })


# (final_equity, closed_trades) pinned from trader.engine.simulate() — default
# "next_open" honest fill, no transaction costs, risk halt off.
GOLDEN_ENGINE = {
    "5/20均线金叉死叉": (9664.8651, 3),
    "RSI震荡战法(60买40卖)": (8438.4683, 14),
    "MACD零轴战法": (8549.3053, 13),
    "布林带突破(上下轨)": (8753.9997, 10),
}


@pytest.mark.parametrize("strategy", list(GOLDEN_ENGINE.keys()))
def test_engine_golden(strategy):
    from trader.strategy_core import compute_signals, DEFAULT_STRATEGY_PARAMS
    from trader.engine import simulate

    df = make_synthetic_bars()
    df_sig = compute_signals(df.copy(), strategy, **DEFAULT_STRATEGY_PARAMS)
    res = simulate(df_sig, capital=CAPITAL)

    exp_eq, exp_trades = GOLDEN_ENGINE[strategy]
    assert res.final_equity == pytest.approx(exp_eq, rel=1e-5), f"{strategy}: equity drift"
    assert res.closed_trades == exp_trades, f"{strategy}: trade count drift"


def test_engine_basic_invariants():
    """Cash accounting sanity: flat strategy never trades and equity == capital."""
    from trader.engine import simulate

    df = make_synthetic_bars()
    df["strat_signal"] = 0
    df["strat_exec_px"] = np.nan
    res = simulate(df, capital=CAPITAL)
    assert res.n_trades == 0
    assert res.final_equity == pytest.approx(CAPITAL)
    assert len(res.equity_curve) == len(df)


def test_engine_next_open_vs_close():
    """The two fill models should both run and generally differ (honest vs optimistic)."""
    from trader.strategy_core import compute_signals, DEFAULT_STRATEGY_PARAMS
    from trader.engine import simulate

    df = make_synthetic_bars()
    d = compute_signals(df.copy(), "5/20均线金叉死叉", **DEFAULT_STRATEGY_PARAMS)
    r_open = simulate(d, capital=CAPITAL, fill="next_open")
    r_close = simulate(d, capital=CAPITAL, fill="close")
    assert r_open.final_equity > 0 and r_close.final_equity > 0
