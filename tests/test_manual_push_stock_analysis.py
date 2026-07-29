from trader import manual_push


def test_build_stock_analysis_message_uses_todays_research_conclusion(monkeypatch):
    monkeypatch.setattr(
        manual_push,
        "daily_research_monitor",
        lambda db_path: {
            "run": {"run_id": "run-1", "trading_date": "2026-07-27"},
            "items": [
                {
                    "symbol": "AAPL",
                    "status": "COMPLETED",
                    "screening_status": "BENCH",
                    "recommendation": "BUY",
                    "ai_score": 85.0,
                    "screening_score": 70.0,
                    "confidence": "HIGH",
                    "thesis": "强劲盈利增长",
                    "risks": ["估值偏高"],
                    "error_code": "",
                }
            ],
        },
    )
    msg = manual_push.build_stock_analysis_message("aapl")
    assert "AAPL" in msg.title
    assert "BUY" in msg.body
    assert "强劲盈利增长" in msg.body
    assert "估值偏高" in msg.body
    assert msg.fields["标的"] == "AAPL"


def test_build_stock_analysis_message_missing_symbol_is_explicit(monkeypatch):
    monkeypatch.setattr(
        manual_push,
        "daily_research_monitor",
        lambda db_path: {"run": None, "items": []},
    )
    msg = manual_push.build_stock_analysis_message("TSLA")
    assert "TSLA" in msg.body
    assert "没有" in msg.body


def test_build_stock_analysis_message_requires_symbol():
    msg = manual_push.build_stock_analysis_message("   ")
    assert "未提供" in msg.body
