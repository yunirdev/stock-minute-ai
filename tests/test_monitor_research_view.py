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
                        "recommendation": "BUY",
                        "ai_score": 82,
                        "screening_score": 75,
                        "thesis": "<script>alert(1)</script>",
                        "risks": [],
                    }
                ],
            },
        },
    )
    assert "2026-07-27" in html
    assert "&lt;AAPL&gt;" in html
    assert "<AAPL>" not in html
    assert "BUY" in html
    assert "&lt;script&gt;" in html
    assert "<script>" not in html


def test_live_research_html_renders_multiline_briefing_with_br_not_escaped():
    """说明栏是 AI 写的多行简报——换行要变成 <br> 才能真的分行显示，不能
    被当成普通文本转义成看不见的 \\n。转义和加 <br> 的顺序错了会把我们自己
    加的标签也转义掉，这里专门测一下顺序对不对。"""
    html = live_research_html(
        {
            "research": {
                "run": {
                    "trading_date": "2026-07-27",
                    "status": "COMPLETED",
                    "completed_symbols": 1,
                    "failed_symbols": 0,
                    "run_id": "r1b",
                },
                "items": [
                    {
                        "rank": 1,
                        "symbol": "CEG",
                        "status": "COMPLETED",
                        "recommendation": "BUY",
                        "ai_score": 85.0,
                        "screening_score": 70.0,
                        "thesis": "**Action**: Buy\n\n**Entry Price**: 267.5\n\n**Stop Loss**: 235.0",
                        "risks": [],
                    }
                ],
            },
        },
    )
    assert "**Entry Price**: 267.5<br><br>**Stop Loss**: 235.0" in html
    assert "\\n" not in html


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
        }
    )

    assert "没有可进入深度研究的候选" in html
    assert "深度候选 0" in html


def test_live_research_html_shows_a_briefing_not_a_risk_dump():
    """已完成的标的，说明栏该是 AI 自己的结论摘要（thesis），不是罗列风险
    标签——"缺少最新 AI 综合分"这类每行都一样的提示不该出现在这里。"""
    html = live_research_html(
        {
            "research": {
                "run": {
                    "trading_date": "2026-07-27",
                    "status": "COMPLETED",
                    "total_symbols": 1,
                    "completed_symbols": 1,
                    "failed_symbols": 0,
                    "run_id": "r3",
                },
                "items": [
                    {
                        "rank": 1,
                        "symbol": "MSFT",
                        "status": "COMPLETED",
                        "recommendation": "BUY",
                        "ai_score": 85.0,
                        "screening_score": 70.0,
                        "thesis": "云业务增长强劲，管理层上调全年指引",
                        "risks": ["缺少最新 AI 综合分", "缺少当前环境的可靠 holdout 统计"],
                    }
                ],
            },
        }
    )

    assert "云业务增长强劲" in html
    assert "缺少最新 AI 综合分" not in html
    assert "缺少当前环境的可靠 holdout 统计" not in html


def test_live_research_html_pending_item_shows_short_status_not_placeholder_conclusion():
    html = live_research_html(
        {
            "research": {
                "run": {
                    "trading_date": "2026-07-27",
                    "status": "RUNNING",
                    "total_symbols": 1,
                    "completed_symbols": 0,
                    "failed_symbols": 0,
                    "run_id": "r4",
                },
                "items": [
                    {
                        "rank": 1,
                        "symbol": "GLD",
                        "status": "PENDING",
                        "recommendation": None,
                        "ai_score": None,
                        "screening_score": 37.5,
                        "risks": ["缺少最新 AI 综合分"],
                    }
                ],
            },
        }
    )

    assert "排队中" in html
    assert "缺少最新 AI 综合分" not in html


def test_live_research_html_translates_worker_failure_diagnosis():
    """原始 error_code 是给日志用的技术代码——说明栏是给人看的简报，
    要翻译成一句话，不能把 TRADINGAGENTS_WORKER_FAILED:... 这种代码直接甩出来。"""
    html = live_research_html(
        {
            "research": {
                "run": {
                    "trading_date": "2026-07-27",
                    "status": "COMPLETED_WITH_ERRORS",
                    "total_symbols": 1,
                    "completed_symbols": 0,
                    "failed_symbols": 1,
                    "run_id": "r5",
                },
                "items": [
                    {
                        "rank": 1,
                        "symbol": "NVDA",
                        "status": "FAILED",
                        "error_code": "TRADINGAGENTS_WORKER_FAILED:APIConnectionError:LLM_API_CONNECTION_UNAVAILABLE",
                    }
                ],
            },
        }
    )

    assert "连不上本地大模型服务" in html
    assert "TRADINGAGENTS_WORKER_FAILED" not in html


def test_live_research_html_translates_dead_snapshot_link_error():
    html = live_research_html(
        {
            "research": {
                "run": {
                    "trading_date": "2026-07-27",
                    "status": "COMPLETED_WITH_ERRORS",
                    "total_symbols": 1,
                    "completed_symbols": 0,
                    "failed_symbols": 1,
                    "run_id": "r6",
                },
                "items": [
                    {
                        "rank": 1,
                        "symbol": "SOXL",
                        "status": "FAILED",
                        "error_code": "TRADINGAGENTS_SNAPSHOT_LINK_UNAVAILABLE",
                    }
                ],
            },
        }
    )

    assert "内部数据写入失败，本次未分析" in html
    assert "TRADINGAGENTS_SNAPSHOT_LINK_UNAVAILABLE" not in html


def test_research_table_only_has_three_columns():
    html = live_research_html(
        {
            "research": {
                "run": {
                    "trading_date": "2026-07-27",
                    "status": "COMPLETED",
                    "total_symbols": 1,
                    "completed_symbols": 1,
                    "failed_symbols": 0,
                    "run_id": "r7",
                },
                "items": [
                    {
                        "rank": 1,
                        "symbol": "MSFT",
                        "status": "COMPLETED",
                        "recommendation": "BUY",
                        "ai_score": 85.0,
                        "screening_score": 70.0,
                        "thesis": "test",
                        "complexity": "HIGH",
                    }
                ],
            },
        }
    )

    assert "<th>标的</th><th>AI 结论</th><th>说明</th>" in html
    assert "复杂度" not in html
    assert "初筛分" not in html
