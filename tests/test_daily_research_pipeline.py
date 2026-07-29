from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from trader.daily_candidates import DailyCandidate
from trader.daily_research import (
    DailyResearchService,
    DailyResearchStore,
    ResearchAnalysis,
)
from trader.models import Candidate
from trader.paper_decision import (
    PaperDecisionService,
    StrategyStatistics,
    StrategyStatisticsRepository,
)


class _Analyzer:
    provider = "fake-tradingagents"
    model = "model"

    def analyze(self, symbol: str, trading_date: str):
        as_of = (
            datetime.fromisoformat(trading_date)
            .replace(tzinfo=timezone.utc)
            .isoformat()
        )
        return ResearchAnalysis(
            recommendation="BUY",
            score=84,
            confidence=0.84,
            thesis="positive thesis",
            provider=self.provider,
            model=self.model,
            source_manifest=(
                {
                    "source": "fake-tradingagents",
                    "status": "OK",
                    "as_of": as_of,
                    "fetched_at": as_of,
                    "quality_score": 1.0,
                    "coverage": ["test"],
                    "payload_version": "test:v1",
                    "failure_code": "",
                    "metadata": {},
                },
            ),
        )


def test_frozen_daily_research_reaches_paper_decision(tmp_path, monkeypatch):
    import trader.daily_research as module

    now = datetime.now(timezone.utc)
    trading_date = now.astimezone(ZoneInfo('America/New_York')).date().isoformat()
    monkeypatch.setattr(
        module,
        "build_daily_candidates",
        lambda *args, **kwargs: [
            DailyCandidate(
                symbol="AAPL",
                rank=1,
                score=80,
                status="ENTRY_READY",
                source_quality_score=70,
                ai_score=None,
                tactical_score=80,
                data_confidence="高",
            )
        ],
    )
    store = DailyResearchStore(str(tmp_path / "ai.duckdb"))
    DailyResearchService(store, _Analyzer()).run(
        ["AAPL"],
        trading_date=trading_date,
        now=now,
    )
    snapshots = store.score_snapshots(now, max_age_hours=36)
    stats = StrategyStatistics(
        statistics_id="stats-1",
        symbol="AAPL",
        strategy="momentum",
        strategy_version="1",
        timeframe="5m",
        market_regime="bull_trend",
        out_of_sample_net_return=0.12,
        sharpe=1.4,
        max_drawdown=0.1,
        trade_count=50,
        win_rate=0.58,
        average_trade_return=0.01,
        fees=0.001,
        slippage=0.001,
        data_start=now - timedelta(days=180),
        data_end=now - timedelta(days=1),
        evaluated_at=now - timedelta(days=1),
        statistics_version="1",
    )
    candidate = Candidate(
        symbol="AAPL",
        score=80,
        rank=1,
        reasons={"votes": {"momentum": 1}},
        as_of=now,
    )

    decisions = PaperDecisionService(
        ai_max_age_minutes=36 * 60,
        ai_min_contributors=1,
        ai_min_weight_coverage=1.0,
    ).decide(
        bars={"AAPL": [object()]},
        positions={},
        candidates=[candidate],
        strategy_statistics=StrategyStatisticsRepository([stats]),
        ai_advisories=snapshots,
        market_regime="bull_trend",
        now=now,
        timeframe="5m",
        universe_version="u1",
        data_version="d1",
    )

    assert len(decisions) == 1
    assert decisions[0].ai_advisory_run_id == snapshots["AAPL"].run_id
    assert decisions[0].evidence["ai"]["provider"] == "fake-tradingagents"
