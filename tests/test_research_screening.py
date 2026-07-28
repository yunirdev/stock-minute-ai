from datetime import datetime, timedelta, timezone

from trader.daily_candidates import DailyCandidate
from trader.paper_decision import StrategyStatistics, StrategyStatisticsRepository
from trader.research_screening import build_research_candidates


def _candidate(symbol: str, score: float) -> DailyCandidate:
    return DailyCandidate(
        symbol=symbol,
        rank=1,
        score=score,
        status="WATCH",
        source_quality_score=60,
        ai_score=None,
        tactical_score=score,
        data_confidence="高",
    )


def _stats(symbol: str, net_return: float) -> StrategyStatistics:
    now = datetime(2026, 7, 27, tzinfo=timezone.utc)
    return StrategyStatistics(
        statistics_id=f"s-{symbol}",
        symbol=symbol,
        strategy="momentum",
        strategy_version="1",
        timeframe="5m",
        market_regime="bull_trend",
        out_of_sample_net_return=net_return,
        sharpe=1.5,
        max_drawdown=0.1,
        trade_count=50,
        win_rate=0.6,
        average_trade_return=0.01,
        fees=0.001,
        slippage=0.001,
        data_start=now - timedelta(days=180),
        data_end=now - timedelta(days=1),
        evaluated_at=now - timedelta(days=1),
        statistics_version="1",
    )


def test_research_screening_prioritizes_reliable_holdout(monkeypatch):
    import trader.research_screening as module

    captured = {}

    def build_candidates(*args, **kwargs):
        captured.update(kwargs)
        return [_candidate("AAPL", 60), _candidate("MSFT", 65)]

    monkeypatch.setattr(module, "build_daily_candidates", build_candidates)
    repository = StrategyStatisticsRepository([_stats("AAPL", 0.2)])
    monkeypatch.setattr(
        module.StrategyStatisticsRepository,
        "from_json",
        lambda path: repository,
    )

    now = datetime(2026, 7, 27, tzinfo=timezone.utc)
    rows = build_research_candidates(
        ["AAPL", "MSFT"],
        timeframe="5m",
        strategy_statistics_path="ignored.json",
        market_regime="bull_trend",
        now=now,
    )

    assert captured["now"] == now
    assert rows[0].symbol == "AAPL"
    assert "Holdout" in rows[0].reasons[-1]
    assert rows[1].data_confidence == "中"
    assert "缺少当前环境" in rows[1].risk_flags[-1]
