from __future__ import annotations


def test_daily_candidates_combine_three_logic_layers(monkeypatch):
    import trader.daily_candidates as dc

    monkeypatch.setattr(dc, "_load_ai_scores", lambda _path: {"NVDA": 82.0, "AAPL": 58.0})
    monkeypatch.setattr(
        dc,
        "_tactical_score",
        lambda symbol, timeframe: {
            "score": 78.0 if symbol == "NVDA" else 52.0,
            "reasons": ["策略共识偏多"],
            "risks": [],
        },
    )

    rows = dc.build_daily_candidates(["NVDA", "AAPL"], ai_db_path=":memory:", limit=3)

    assert rows[0].symbol == "NVDA"
    assert rows[0].status in {"ENTRY_READY", "WAIT_BREAKOUT"}
    assert rows[0].data_confidence == "高"
    assert "AI 综合分 82.0" in rows[0].reasons
    assert "策略共识偏多" in rows[0].reasons


def test_daily_candidates_marks_missing_data_as_lower_confidence(monkeypatch):
    import trader.daily_candidates as dc

    monkeypatch.setattr(dc, "_load_ai_scores", lambda _path: {})
    monkeypatch.setattr(dc, "_tactical_score", lambda symbol, timeframe: None)

    rows = dc.build_daily_candidates(["XYZ"], ai_db_path=None, include_anchors=False)

    assert rows[0].symbol == "XYZ"
    assert rows[0].data_confidence == "低"
    assert "缺少最新 AI 综合分" in rows[0].risk_flags
    assert "缺少本地 K 线或技术共识" in rows[0].risk_flags


def test_daily_candidates_hard_risk_caps_action_status(monkeypatch):
    import trader.daily_candidates as dc

    monkeypatch.setattr(dc, "_load_ai_scores", lambda _path: {})
    monkeypatch.setattr(dc, "_tactical_score", lambda symbol, timeframe: None)

    rows = dc.build_daily_candidates(["NVDA"], ai_db_path=None, include_anchors=False)

    assert rows[0].symbol == "NVDA"
    assert rows[0].status == "BENCH"


def test_daily_candidates_save_load_and_symbol_filter(tmp_path):
    import trader.daily_candidates as dc

    path = tmp_path / "daily.json"
    candidates = [
        dc.DailyCandidate(
            symbol="NVDA",
            rank=1,
            score=80.0,
            status="ENTRY_READY",
            source_quality_score=70.0,
            ai_score=82.0,
            tactical_score=78.0,
            data_confidence="高",
        ),
        dc.DailyCandidate(
            symbol="SPY",
            rank=2,
            score=70.0,
            status="MARKET_ANCHOR",
            source_quality_score=62.0,
            ai_score=None,
            tactical_score=70.0,
            data_confidence="中",
        ),
    ]

    dc.save_daily_candidates(candidates, path)

    loaded = dc.load_daily_candidates(path)
    assert [row.symbol for row in loaded] == ["NVDA", "SPY"]
    assert dc.daily_candidate_symbols(path) == ["NVDA"]
