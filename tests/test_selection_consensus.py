from datetime import datetime, timezone

import pandas as pd

from trader import selection
from trader.selection import ConsensusSelector

NOW = datetime(2026, 7, 28, 14, 0, tzinfo=timezone.utc)


def _bars(n: int = 60) -> pd.DataFrame:
    return pd.DataFrame({"close": [100.0] * n})


def test_errored_strategy_is_excluded_from_denominator_not_counted_as_neutral(
    monkeypatch,
):
    """A strategy that throws must not lower the score the same way a
    strategy that computed and genuinely voted neutral would.
    """
    monkeypatch.setattr(selection, "get_bars", lambda symbol, timeframe: _bars())

    def compute_signals(df, strategy, **kwargs):
        if strategy == "broken_strategy":
            raise ValueError("bad data")
        out = df.copy()
        out["strat_signal"] = 1  # bullish
        return out

    monkeypatch.setattr(selection, "compute_signals", compute_signals)

    selector = ConsensusSelector(strategies=["broken_strategy", "good_strategy"])
    candidates = selector.select(["AAPL"], "5m", NOW)

    assert len(candidates) == 1
    cand = candidates[0]
    # Only the one working, bullish strategy should count — 1/1 = 100%,
    # not 1/2 = 50% (which is what happens if the errored one is silently
    # counted as a neutral vote in the denominator).
    assert cand.score == 100.0
    assert cand.reasons["total_strategies"] == 1
    assert cand.reasons["errored_strategies"] == ["broken_strategy"]


def test_all_strategies_erroring_falls_back_to_neutral_score_not_zero_division(
    monkeypatch,
):
    monkeypatch.setattr(selection, "get_bars", lambda symbol, timeframe: _bars())

    def compute_signals(df, strategy, **kwargs):
        raise ValueError("bad data")

    monkeypatch.setattr(selection, "compute_signals", compute_signals)

    selector = ConsensusSelector(strategies=["a", "b"])
    candidates = selector.select(["AAPL"], "5m", NOW)

    assert len(candidates) == 1
    cand = candidates[0]
    assert cand.score == 50.0
    assert cand.reasons["total_strategies"] == 0
    assert cand.reasons["errored_strategies"] == ["a", "b"]


def test_genuinely_neutral_strategy_still_counts_in_denominator(monkeypatch):
    monkeypatch.setattr(selection, "get_bars", lambda symbol, timeframe: _bars())

    def compute_signals(df, strategy, **kwargs):
        out = df.copy()
        out["strat_signal"] = 0 if strategy == "neutral_strategy" else 1
        return out

    monkeypatch.setattr(selection, "compute_signals", compute_signals)

    selector = ConsensusSelector(strategies=["neutral_strategy", "bull_strategy"])
    candidates = selector.select(["AAPL"], "5m", NOW)

    cand = candidates[0]
    # Both strategies ran successfully (one just voted neutral), so both
    # count in the denominator: 1 bull / 2 total = 50%.
    assert cand.score == 50.0
    assert cand.reasons["total_strategies"] == 2
    assert cand.reasons["errored_strategies"] == []
