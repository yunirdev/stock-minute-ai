from __future__ import annotations

import pandas as pd


def _sample_intraday_bars():
    base = pd.Timestamp("2026-06-18 13:30:00", tz="UTC")
    return pd.DataFrame([
        {
            "timestamp_utc": base + pd.Timedelta(minutes=i),
            "open": 100.0 + i * 0.1,
            "high": 100.3 + i * 0.2,
            "low": 99.8 + i * 0.1,
            "close": 100.0 + i * 0.25,
            "volume": 1000 + i * 10,
        }
        for i in range(20)
    ])


def test_manual_intraday_push_message_uses_intraday_levels(monkeypatch):
    import trader.manual_push as mp

    monkeypatch.setattr(mp, "_load_bars", lambda symbol: _sample_intraday_bars())

    messages = mp.build_intraday_levels_messages(["SPY"])

    assert len(messages) == 1
    assert "OR/VWAP" in messages[0].title
    assert "SPY" in messages[0].body
    assert "OR15" in messages[0].body
    assert "VWAP" in messages[0].body


def _record_call(db_path, bias="偏多"):
    """写入一条当期的晨报方向记录，模拟"今天早上确实这么说过"。"""
    from datetime import datetime, timezone

    from trader.brief_review import BriefCallStore
    from trader.report_period import resolve_daily_period

    period = resolve_daily_period(datetime.now(timezone.utc))
    BriefCallStore(db_path).record(trading_date=period.label, bias=bias)
    return period.label


def test_manual_direction_review_message_handles_missing_bars(monkeypatch, tmp_path):
    import trader.manual_push as mp

    db = str(tmp_path / "calls.duckdb")
    _record_call(db)
    monkeypatch.setattr(mp, "_load_bars", lambda symbol: pd.DataFrame())

    message = mp.build_direction_review_message(["QQQ"], db_path=db)

    assert "晨报方向复盘" in message.title
    assert "QQQ" in message.body
    assert "暂无法评分" in message.body


def test_manual_direction_review_message_scores_the_recorded_call(monkeypatch, tmp_path):
    """方向来自当天记录，不再由使用者从下拉框里挑。"""
    import trader.manual_push as mp

    db = str(tmp_path / "calls.duckdb")
    _record_call(db, bias="偏多")
    monkeypatch.setattr(mp, "_load_bars", lambda symbol: _sample_intraday_bars())

    message = mp.build_direction_review_message(["SPY"], db_path=db)

    assert "方向判断有效" in message.body
    assert "早上的判断：偏多" in message.body


def test_direction_review_refuses_to_score_without_a_recorded_call(tmp_path):
    """没有晨报记录就说没有，而不是默认"中性"糊弄过去——那等于凭空
    编一个判断再给它打分。"""
    import trader.manual_push as mp

    message = mp.build_direction_review_message(
        ["SPY"], db_path=str(tmp_path / "empty.duckdb")
    )
    assert "没有记录到晨报方向判断" in message.body
    assert "无法复盘" in message.body
