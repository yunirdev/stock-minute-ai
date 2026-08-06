import json
from datetime import datetime, timedelta, timezone

import duckdb
import pandas as pd

from trader.daily_candidates import DailyCandidate
from trader.daily_research import (
    DailyResearchService,
    DailyResearchStore,
    ResearchAnalysis,
)
from trader.paper_decision import StrategyStatistics
from trader.research_snapshot_quality import (
    generate_snapshot_quality_report,
    save_snapshot_quality_report,
)


class _Analyzer:
    provider = "quality-test"
    model = "quality-v1"

    def describe(self):
        return {"provider": self.provider, "model": self.model}

    def analyze(self, symbol, trading_date, *, complexity: str = ""):
        return ResearchAnalysis(
            recommendation="BUY",
            score=80.0,
            confidence=0.8,
            thesis=f"{symbol} {trading_date}",
            provider=self.provider,
            model=self.model,
        )


def _captured_inputs(now: datetime):
    timestamps = pd.date_range(
        now - timedelta(minutes=5 * 50),
        periods=50,
        freq="5min",
        tz="UTC",
    )
    bars = pd.DataFrame(
        {
            "symbol": ["AAPL"] * 50,
            "timestamp_utc": timestamps,
            "open": [100.0] * 50,
            "high": [101.0] * 50,
            "low": [99.0] * 50,
            "close": [100.5] * 50,
            "volume": [1_000.0] * 50,
        }
    )
    statistics = StrategyStatistics(
        statistics_id=f"stat-{now.date().isoformat()}",
        symbol="AAPL",
        strategy="trend",
        strategy_version="trend-v1",
        timeframe="5m",
        market_regime="bull",
        out_of_sample_net_return=0.10,
        sharpe=1.0,
        max_drawdown=0.08,
        trade_count=40,
        win_rate=0.60,
        average_trade_return=0.01,
        fees=10.0,
        slippage=0.0005,
        data_start=now - timedelta(days=60),
        data_end=now - timedelta(days=1),
        evaluated_at=now - timedelta(hours=1),
        statistics_version="holdout-v1",
    )
    candidate = DailyCandidate(
        symbol="AAPL",
        rank=1,
        score=80.0,
        status="ENTRY_READY",
        source_quality_score=70.0,
        ai_score=None,
        tactical_score=80.0,
        data_confidence="高",
        reasons=["complete inputs"],
        risk_flags=[],
        as_of=(now - timedelta(minutes=1)).isoformat(),
    )
    return bars, statistics, candidate


def _seed_days(db_path, monkeypatch, count=10):
    import trader.daily_research as module

    def build(*_args, **kwargs):
        now = kwargs["now"]
        bars, statistics, candidate = _captured_inputs(now)
        kwargs["input_capture"]["bars"] = {"AAPL": bars}
        kwargs["input_capture"]["strategy_statistics"] = (statistics,)
        return [candidate]

    monkeypatch.setattr(module, "build_daily_candidates", build)
    store = DailyResearchStore(str(db_path))
    service = DailyResearchService(store, _Analyzer())
    trading_days = [
        "2026-07-13",
        "2026-07-14",
        "2026-07-15",
        "2026-07-16",
        "2026-07-17",
        "2026-07-20",
        "2026-07-21",
        "2026-07-22",
        "2026-07-23",
        "2026-07-24",
    ][:count]
    for trading_date in trading_days:
        now = datetime.fromisoformat(
            f"{trading_date}T12:00:00+00:00"
        )
        service.run(
            ["AAPL"],
            trading_date=trading_date,
            timeframe="5m",
            strategy_statistics_path="stats.json",
            market_regime="bull",
            now=now,
        )
    return store


def test_ten_day_snapshot_quality_report_passes_and_is_auditable(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "ai.duckdb"
    _seed_days(db_path, monkeypatch)

    report = generate_snapshot_quality_report(
        db_path,
        required_days=10,
        min_coverage=1.0,
        generated_at=datetime(2026, 7, 25, tzinfo=timezone.utc),
    )

    assert report["passed"]
    assert report["observed_trading_days"] == 10
    assert report["expected_snapshots"] == 10
    assert report["snapshot_coverage"] == 1.0
    assert report["acceptable_quality_coverage"] == 1.0
    assert report["comparison_coverage"] == 1.0
    assert report["critical_field_coverage"] == 1.0
    assert report["unclassified_differences"] == 0
    assert all(day["available_snapshots"] == 1 for day in report["daily"])

    assert save_snapshot_quality_report(db_path, report)
    assert not save_snapshot_quality_report(db_path, report)
    conn = duckdb.connect(str(db_path), read_only=True)
    saved = conn.execute(
        "SELECT payload FROM research_snapshot_quality_reports "
        "WHERE report_id=?",
        [report["report_id"]],
    ).fetchone()
    conn.close()
    assert json.loads(saved[0])["passed"]


def test_quality_report_fails_for_unclassified_difference(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "ai.duckdb"
    _seed_days(db_path, monkeypatch)
    conn = duckdb.connect(str(db_path))
    conn.execute(
        """
        UPDATE daily_research_snapshot_comparisons
        SET status='MISMATCH',
            classification='UNCLASSIFIED',
            differences_json=?
        WHERE run_id=(
            SELECT run_id FROM daily_research_runs
            ORDER BY trading_date LIMIT 1
        )
        """,
        [
            json.dumps(
                [
                    {
                        "field": "score",
                        "actual": 80,
                        "captured": 79,
                        "classification": "UNCLASSIFIED",
                    }
                ]
            )
        ],
    )
    conn.close()

    report = generate_snapshot_quality_report(
        db_path,
        required_days=10,
    )

    assert not report["passed"]
    assert report["total_differences"] == 1
    assert report["unclassified_differences"] == 1


def test_quality_report_requires_full_observation_window(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "ai.duckdb"
    _seed_days(db_path, monkeypatch, count=9)

    report = generate_snapshot_quality_report(
        db_path,
        required_days=10,
    )

    assert not report["passed"]
    assert report["observed_trading_days"] == 9
