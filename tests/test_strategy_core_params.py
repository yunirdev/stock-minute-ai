import numpy as np
import pandas as pd

from trader.strategy_core import (
    DEFAULT_STRATEGY_PARAMS,
    STRATEGY_OPTIONS,
    STRATEGY_PARAM_SPECS,
    compute_signals,
)


def _synthetic_bars(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    base = 100 + np.cumsum(rng.normal(0, 0.6, n))
    high = base + rng.uniform(0.1, 0.5, n)
    low = base - rng.uniform(0.1, 0.5, n)
    open_ = base + rng.uniform(-0.2, 0.2, n)
    close = base + rng.uniform(-0.2, 0.2, n)
    return pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.integers(1000, 5000, n),
        }
    )


def test_grid_pct_override_changes_signals_for_small_grid():
    df = _synthetic_bars()
    default_sig = compute_signals(df.copy(), "半仓小网格(5%间距)")
    wide_sig = compute_signals(df.copy(), "半仓小网格(5%间距)", grid_pct=0.20)
    assert not default_sig["strat_signal"].equals(wide_sig["strat_signal"])


def test_grid_pct_defaults_match_hardcoded_historical_behavior():
    df = _synthetic_bars()
    small_default = compute_signals(df.copy(), "半仓小网格(5%间距)")
    small_explicit = compute_signals(df.copy(), "半仓小网格(5%间距)", grid_pct=0.05)
    assert small_default["strat_signal"].equals(small_explicit["strat_signal"])

    large_default = compute_signals(df.copy(), "半仓大网格(10%间距)")
    large_explicit = compute_signals(df.copy(), "半仓大网格(10%间距)", grid_pct=0.10)
    assert large_default["strat_signal"].equals(large_explicit["strat_signal"])


def test_strategy_param_specs_only_reference_known_strategies():
    assert set(STRATEGY_PARAM_SPECS) <= set(STRATEGY_OPTIONS)


def test_strategy_param_specs_keys_are_recognized_kwargs():
    known_extra_keys = {"grid_pct"}
    for strategy, spec in STRATEGY_PARAM_SPECS.items():
        for _label, key, _default, _kind in spec:
            assert key in DEFAULT_STRATEGY_PARAMS or key in known_extra_keys, (
                f"{strategy}: unrecognized param key {key}"
            )
