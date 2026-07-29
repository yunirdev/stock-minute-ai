from datetime import datetime, timedelta, timezone

from trader.ai.safety import AIScoreSnapshot
from trader.models import Candidate
from trader.paper_decision import (
    PaperDecisionService,
    StrategyStatistics,
    StrategyStatisticsRepository,
)


def test_paper_decision_rejects_below_configured_ai_score():
    now = datetime(2026, 7, 27, tzinfo=timezone.utc)
    stats = StrategyStatistics(
        statistics_id="stats",
        symbol="AAPL",
        strategy="momentum",
        strategy_version="1",
        timeframe="5m",
        market_regime="bull_trend",
        out_of_sample_net_return=0.1,
        sharpe=1.2,
        max_drawdown=0.1,
        trade_count=40,
        win_rate=0.55,
        average_trade_return=0.01,
        fees=0.001,
        slippage=0.001,
        data_start=now - timedelta(days=180),
        data_end=now - timedelta(days=1),
        evaluated_at=now - timedelta(days=1),
        statistics_version="1",
    )
    candidate = Candidate(
        "AAPL",
        75,
        1,
        {"votes": {"momentum": 1}},
        now,
    )
    advisory = AIScoreSnapshot(
        "AAPL",
        64,
        now,
        run_id="research-1",
        provider="tradingagents",
        model="model",
        source="daily_research",
        generated_by="TradingAgentsAdapter",
        contributors=[{"agent_name": "tradingagents_graph"}],
        contributor_count=1,
        weight_coverage=1.0,
        has_llm=True,
    )
    decisions = PaperDecisionService(
        min_ai_score=65,
        ai_max_age_minutes=36 * 60,
        ai_min_contributors=1,
        ai_min_weight_coverage=1.0,
    ).decide(
        bars={"AAPL": [object()]},
        positions={},
        candidates=[candidate],
        strategy_statistics=StrategyStatisticsRepository([stats]),
        ai_advisories={"AAPL": advisory},
        market_regime="bull_trend",
        now=now,
        timeframe="5m",
    )
    assert decisions == []
