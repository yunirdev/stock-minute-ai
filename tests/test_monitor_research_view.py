from datetime import datetime, timezone

from trader.monitor_research_view import live_research_html


def test_live_research_html_renders_and_escapes():
    html = live_research_html(
        {
            "research": {
                "run": {
                    "trading_date": "2026-07-27",
                    "status": "COMPLETED",
                    "completed_symbols": 1,
                    "failed_symbols": 0,
                    "run_id": "r1",
                },
                "items": [
                    {
                        "rank": 1,
                        "symbol": "<AAPL>",
                        "status": "COMPLETED",
                        "screening_status": "BENCH",
                        "recommendation": "BUY",
                        "ai_score": 82,
                        "screening_score": 75,
                        "risks": ["<missing bars>"],
                    }
                ],
            },
            "runtime": {
                "session": "open",
                "tick_count": 2,
                "equity": 10_000,
                "updated_at": "2026-07-27T12:00:00+00:00",
                "candidates": [],
            },
        },
        now=datetime(2026, 7, 27, 12, 1, tzinfo=timezone.utc),
    )
    assert "2026-07-27" in html
    assert "&lt;AAPL&gt;" in html
    assert "<AAPL>" not in html
    assert "BENCH" in html
    assert "&lt;missing bars&gt;" in html
    assert "<missing bars>" not in html


def test_live_research_html_explains_empty_deep_candidate_failure():
    html = live_research_html(
        {
            "research": {
                "run": {
                    "trading_date": "2026-07-27",
                    "status": "FAILED",
                    "total_symbols": 0,
                    "completed_symbols": 0,
                    "failed_symbols": 0,
                    "run_id": "r2",
                    "error_code": "NO_ELIGIBLE_DEEP_CANDIDATES",
                },
                "items": [],
            },
            "runtime": None,
        }
    )

    assert "没有可进入深度研究的候选" in html
    assert "深度候选 0" in html


def test_live_research_html_marks_stale_runtime_as_stopped():
    html = live_research_html(
        {
            "research": {"run": None, "items": []},
            "runtime": {
                "session": "post",
                "tick_count": 1,
                "equity": 10_000,
                "updated_at": "2026-07-27T12:00:00+00:00",
                "candidates": [],
            },
        },
        now=datetime(2026, 7, 27, 12, 4, 1, tzinfo=timezone.utc),
        runtime_stale_after_seconds=180,
    )

    assert "已停止或心跳过期" in html
    assert "运行正常" not in html
