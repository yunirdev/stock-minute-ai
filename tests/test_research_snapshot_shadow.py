from dataclasses import asdict
from datetime import datetime, timedelta, timezone

import pandas as pd

from trader.daily_candidates import DailyCandidate
from trader.daily_research import (
    DailyResearchService,
    DailyResearchStore,
    ResearchAnalysis,
)
from trader.models import ResearchQuality, ResearchSourceStatus
from trader.paper_decision import StrategyStatistics
from trader.research_snapshot import ResearchSnapshotStore
from trader.research_screening import build_research_candidates


NOW = datetime(2026, 7, 27, 12, tzinfo=timezone.utc)


class _Analyzer:
    provider = "shadow-test"
    model = "model-v1"

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


def _bars() -> pd.DataFrame:
    timestamps = pd.date_range(
        NOW - timedelta(minutes=5 * 50),
        periods=50,
        freq="5min",
        tz="UTC",
    )
    return pd.DataFrame(
        {
            "symbol": ["AAPL"] * 50,
            "timestamp_utc": timestamps,
            "open": [100.0 + index for index in range(50)],
            "high": [101.0 + index for index in range(50)],
            "low": [99.0 + index for index in range(50)],
            "close": [100.5 + index for index in range(50)],
            "volume": [1_000.0 + index for index in range(50)],
        }
    )


def _statistics() -> StrategyStatistics:
    return StrategyStatistics(
        statistics_id="stat-aapl",
        symbol="AAPL",
        strategy="trend",
        strategy_version="trend-v1",
        timeframe="5m",
        market_regime="bull",
        out_of_sample_net_return=0.12,
        sharpe=1.2,
        max_drawdown=0.08,
        trade_count=40,
        win_rate=0.60,
        average_trade_return=0.01,
        fees=10.0,
        slippage=0.0005,
        data_start=NOW - timedelta(days=60),
        data_end=NOW - timedelta(days=1),
        evaluated_at=NOW - timedelta(hours=1),
        statistics_version="holdout-v1",
    )


def _candidate() -> DailyCandidate:
    return DailyCandidate(
        symbol="AAPL",
        rank=1,
        score=81.5,
        status="ENTRY_READY",
        source_quality_score=70.0,
        ai_score=None,
        tactical_score=82.0,
        data_confidence="高",
        reasons=["actual reason"],
        risk_flags=[],
        as_of=(NOW - timedelta(minutes=1)).isoformat(),
    )


def test_daily_research_shadow_snapshot_matches_actual_inputs(
    tmp_path,
    monkeypatch,
):
    import trader.daily_research as module

    bars = _bars()
    statistics = _statistics()
    candidate = _candidate()

    def build(*_args, **kwargs):
        capture = kwargs["input_capture"]
        capture["bars"] = {"AAPL": bars.copy()}
        capture["strategy_statistics"] = (statistics,)
        return [candidate]

    monkeypatch.setattr(module, "build_daily_candidates", build)
    db_path = tmp_path / "ai.duckdb"
    store = DailyResearchStore(str(db_path))
    service = DailyResearchService(store, _Analyzer())

    run = service.run(
        ["AAPL"],
        trading_date="2026-07-27",
        timeframe="5m",
        strategy_statistics_path="conf/strategy_statistics.json",
        now=NOW,
    )

    link = store.snapshot_links(run.run_id)[0]
    assert link["status"] == "WRITTEN"
    snapshot_store = ResearchSnapshotStore(db_path)
    snapshot = snapshot_store.get(link["snapshot_id"])
    assert snapshot is not None
    assert snapshot.run_id == run.run_id
    assert snapshot.quality == ResearchQuality.GOOD
    assert snapshot.payload["candidate"] == asdict(candidate)

    captured_rows = snapshot.payload["bars"]["rows"]
    assert len(captured_rows) == len(bars)
    assert captured_rows[0]["timestamp_utc"] == (
        bars.iloc[0]["timestamp_utc"].isoformat()
    )
    for field in ("open", "high", "low", "close", "volume"):
        assert captured_rows[-1][field] == bars.iloc[-1][field]

    captured_statistics = snapshot.payload["strategy_statistics"][0]
    expected_statistics = asdict(statistics)
    for field in ("data_start", "data_end", "evaluated_at"):
        expected_statistics[field] = expected_statistics[field].isoformat()
    assert captured_statistics == expected_statistics
    assert {
        entry.source: entry.status
        for entry in snapshot.source_manifest
    } == {
        "local_bar_cache": ResearchSourceStatus.OK,
        "strategy_statistics": ResearchSourceStatus.OK,
        "deterministic_screening": ResearchSourceStatus.OK,
    }
    assert (
        snapshot_store.replay_for_run(run.run_id, "AAPL")
        == snapshot
    )


def test_screening_capture_receives_exact_bar_and_statistics_objects(
    monkeypatch,
):
    import trader.daily_candidates as daily_candidates
    import trader.research_screening as screening

    bars = _bars()
    statistics = _statistics()
    monkeypatch.setattr(
        daily_candidates,
        "_load_bars",
        lambda _symbol, _timeframe: bars.copy(),
    )
    monkeypatch.setattr(
        daily_candidates,
        "_consensus_score",
        lambda _symbol, _timeframe: 70.0,
    )
    monkeypatch.setattr(
        screening.StrategyStatisticsRepository,
        "from_json",
        lambda _path: screening.StrategyStatisticsRepository([statistics]),
    )
    capture = {}

    rows = build_research_candidates(
        ["AAPL"],
        timeframe="5m",
        strategy_statistics_path="statistics.json",
        market_regime="bull",
        now=NOW,
        input_capture=capture,
    )

    assert rows[0].symbol == "AAPL"
    pd.testing.assert_frame_equal(capture["bars"]["AAPL"], bars)
    assert capture["strategy_statistics"] == (statistics,)


def test_shadow_snapshot_marks_missing_sources_without_changing_research(
    tmp_path,
    monkeypatch,
):
    import trader.daily_research as module

    monkeypatch.setattr(
        module,
        "build_daily_candidates",
        lambda *args, **kwargs: [_candidate()],
    )
    db_path = tmp_path / "ai.duckdb"
    store = DailyResearchStore(str(db_path))

    run = DailyResearchService(store, _Analyzer()).run(
        ["AAPL"],
        trading_date="2026-07-27",
        now=NOW,
    )

    assert run.status == "COMPLETED"
    link = store.snapshot_links(run.run_id)[0]
    assert link["status"] == "WRITTEN"
    snapshot = ResearchSnapshotStore(db_path).get(link["snapshot_id"])
    assert snapshot is not None
    assert snapshot.quality == ResearchQuality.PARTIAL
    manifest = {entry.source: entry for entry in snapshot.source_manifest}
    assert manifest["local_bar_cache"].status == ResearchSourceStatus.MISSING
    assert manifest["local_bar_cache"].failure_code == "BAR_CACHE_EMPTY"
    assert (
        manifest["strategy_statistics"].status
        == ResearchSourceStatus.MISSING
    )
    assert (
        manifest["strategy_statistics"].failure_code
        == "STRATEGY_STATISTICS_PATH_MISSING"
    )


def test_snapshot_write_failure_blocks_unreplayable_deep_research(
    tmp_path,
    monkeypatch,
):
    import trader.daily_research as module

    monkeypatch.setattr(
        module,
        "build_daily_candidates",
        lambda *args, **kwargs: [_candidate()],
    )

    class _FailingSnapshotStore:
        def save(self, _snapshot):
            raise OSError("snapshot disk unavailable")

    store = DailyResearchStore(str(tmp_path / "ai.duckdb"))
    service = DailyResearchService(
        store,
        _Analyzer(),
        snapshot_store=_FailingSnapshotStore(),
    )

    run = service.run(
        ["AAPL"],
        trading_date="2026-07-27",
        now=NOW,
    )

    assert run.status == "FAILED"
    assert run.error_code == "TRADINGAGENTS_SNAPSHOT_LINK_UNAVAILABLE"
    assert store.items(run.run_id)[0].status == "FAILED"
    link = store.snapshot_links(run.run_id)[0]
    assert link["status"] == "FAILED"
    assert link["snapshot_id"] == ""
    assert link["error_code"] == "IO_ERROR:OSError"
