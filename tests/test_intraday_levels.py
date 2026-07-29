from __future__ import annotations

import pandas as pd


def test_compute_intraday_levels_uses_regular_session_or15_and_vwap():
    from trader.intraday_levels import compute_intraday_levels

    base = pd.Timestamp("2026-06-18 13:30:00", tz="UTC")
    rows = []
    for idx in range(20):
        rows.append({
            "timestamp_utc": base + pd.Timedelta(minutes=idx),
            "high": 100 + idx * 0.2,
            "low": 99 + idx * 0.1,
            "close": 99.5 + idx * 0.25,
            "volume": 1000 + idx * 10,
        })
    df = pd.DataFrame(rows)

    levels = compute_intraday_levels("SPY", df, open_range_minutes=15)

    assert levels is not None
    assert levels.symbol == "SPY"
    assert levels.open_range_high == 102.8
    assert levels.open_range_low == 99.0
    assert levels.last_price == 104.25
    assert levels.vwap is not None
    assert levels.state == "bull_breakout"


def test_intraday_followup_handles_missing_bars():
    from trader.intraday_levels import build_intraday_followup, compute_intraday_levels

    levels = compute_intraday_levels("QQQ", pd.DataFrame())

    assert levels is None
    assert "暂不可用" in build_intraday_followup(levels)
