from datetime import datetime, timedelta, timezone

from trader.daily_candidates import DailyCandidate, build_daily_candidates
from trader.daily_research import (
    DailyResearchItem,
    DailyResearchRun,
    DailyResearchService,
    DailyResearchStore,
    ResearchAnalysis,
    in_daily_run_window,
    research_target_date,
)


class _Analyzer:
    provider = "fake-tradingagents"
    model = "test-model"

    def describe(self):
        return {"provider": self.provider, "model": self.model}

    def analyze(self, symbol: str, trading_date: str):
        if symbol == "FAIL":
            raise RuntimeError("MODEL_UNAVAILABLE")
        return ResearchAnalysis(
            recommendation="BUY" if symbol == "AAPL" else "HOLD",
            score=82 if symbol == "AAPL" else 58,
            confidence=0.82,
            thesis=f"{symbol} thesis for {trading_date}",
            risks=["test risk"],
            provider=self.provider,
            model=self.model,
            raw={"symbol": symbol},
            source_manifest=(
                {
                    "source": "fake-tradingagents",
                    "status": "OK",
                    "as_of": (
                        datetime.fromisoformat(trading_date)
                        .replace(tzinfo=timezone.utc)
                        .isoformat()
                    ),
                    "fetched_at": (
                        datetime.fromisoformat(trading_date)
                        .replace(tzinfo=timezone.utc)
                        .isoformat()
                    ),
                    "quality_score": 1.0,
                    "coverage": ["test"],
                    "payload_version": "test:v1",
                    "failure_code": "",
                    "metadata": {},
                },
            ),
        )


def _candidates():
    return [
        DailyCandidate(
            symbol="AAPL",
            rank=1,
            score=78,
            status="ENTRY_READY",
            source_quality_score=70,
            ai_score=None,
            tactical_score=80,
            data_confidence="高",
        ),
        DailyCandidate(
            symbol="MSFT",
            rank=2,
            score=70,
            status="WAIT_BREAKOUT",
            source_quality_score=70,
            ai_score=None,
            tactical_score=70,
            data_confidence="高",
        ),
        DailyCandidate(
            symbol="FAIL",
            rank=3,
            score=69,
            status="WATCH",
            source_quality_score=60,
            ai_score=None,
            tactical_score=69,
            data_confidence="中",
        ),
    ]


def test_daily_research_persists_one_batch_and_snapshots(tmp_path, monkeypatch):
    import trader.daily_research as module

    monkeypatch.setattr(module, "build_daily_candidates", lambda *args, **kwargs: _candidates())
    store = DailyResearchStore(str(tmp_path / "ai.duckdb"))
    service = DailyResearchService(store, _Analyzer())
    now = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)

    run = service.run(
        ["AAPL", "MSFT", "FAIL"],
        trading_date="2026-07-27",
        deep_limit=2,
        now=now,
    )

    assert run.status == "COMPLETED"
    assert run.completed_symbols == 2
    assert run.failed_symbols == 0
    assert [item.status for item in store.items(run.run_id)] == [
        "COMPLETED",
        "COMPLETED",
        "SCREENED",
    ]
    snapshots = store.score_snapshots(now)
    assert snapshots["AAPL"].score == 82
    assert snapshots["AAPL"].source == "daily_research"
    assert snapshots["AAPL"].run_id == run.run_id


def test_daily_research_records_partial_failure(tmp_path, monkeypatch):
    import trader.daily_research as module

    monkeypatch.setattr(module, "build_daily_candidates", lambda *args, **kwargs: _candidates())
    store = DailyResearchStore(str(tmp_path / "ai.duckdb"))
    service = DailyResearchService(store, _Analyzer())
    now = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)

    run = service.run(
        ["AAPL", "MSFT", "FAIL"],
        trading_date="2026-07-27",
        deep_limit=3,
        now=now,
    )

    assert run.status == "COMPLETED_WITH_ERRORS"
    assert run.completed_symbols == 2
    assert run.failed_symbols == 1
    failed = [item for item in store.items(run.run_id) if item.status == "FAILED"]
    assert failed[0].symbol == "FAIL"
    assert failed[0].error_code == "MODEL_UNAVAILABLE"


def test_daily_research_is_idempotent_without_force(tmp_path, monkeypatch):
    import trader.daily_research as module

    monkeypatch.setattr(module, "build_daily_candidates", lambda *args, **kwargs: _candidates())
    store = DailyResearchStore(str(tmp_path / "ai.duckdb"))
    service = DailyResearchService(store, _Analyzer())
    now = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)
    first = service.run(["AAPL"], trading_date="2026-07-27", now=now)
    second = service.run(["AAPL"], trading_date="2026-07-27", now=now)
    assert second.run_id == first.run_id


def test_daily_schedule_uses_current_date_premarket_and_next_date_postclose():
    premarket = datetime(2026, 7, 27, 12, 30, tzinfo=timezone.utc)
    postclose = datetime(2026, 7, 27, 21, 0, tzinfo=timezone.utc)
    friday_postclose = datetime(2026, 7, 31, 21, 0, tzinfo=timezone.utc)

    assert in_daily_run_window(premarket)
    assert research_target_date(premarket) == "2026-07-27"
    assert research_target_date(postclose) == "2026-07-28"
    assert research_target_date(friday_postclose) == "2026-08-03"


def test_daily_candidate_as_of_uses_the_injected_batch_clock(monkeypatch):
    import trader.daily_candidates as module

    now = datetime(2026, 7, 27, 5, 10, 21, tzinfo=timezone.utc)
    monkeypatch.setattr(
        module,
        "_tactical_score",
        lambda *args, **kwargs: None,
    )

    rows = build_daily_candidates(
        ["AAPL"],
        include_anchors=False,
        now=now,
    )

    assert rows[0].as_of == now.isoformat()


def test_real_screening_clock_builds_a_replayable_snapshot(tmp_path):
    now = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)
    store = DailyResearchStore(str(tmp_path / "ai.duckdb"))

    run = DailyResearchService(store, _Analyzer()).run(
        ["AAPL"],
        trading_date="2026-07-27",
        deep_limit=1,
        now=now,
    )

    assert run.status == "COMPLETED"
    link = store.snapshot_links(run.run_id)[0]
    assert link["status"] == "WRITTEN"
    assert link["error_code"] == ""


def _running_record(now: datetime) -> DailyResearchRun:
    return DailyResearchRun(
        run_id="research-interrupted",
        trading_date="2026-07-27",
        status="RUNNING",
        universe_version="universe",
        data_cutoff=now,
        timeframe="5m",
        screen_limit=2,
        deep_limit=2,
        provider="tradingagents",
        model="model",
        total_symbols=2,
        completed_symbols=0,
        failed_symbols=0,
        started_at=now,
        config_version="config",
    )


def test_stale_research_run_recovery_is_timeout_bounded_and_terminal(tmp_path):
    started = datetime(2026, 7, 27, 5, 0, tzinfo=timezone.utc)
    store = DailyResearchStore(str(tmp_path / "ai.duckdb"))
    run = _running_record(started)
    store.start_run(run)
    for rank, (symbol, status) in enumerate(
        (("AAPL", "RUNNING"), ("MSFT", "PENDING")),
        start=1,
    ):
        store.save_item(
            DailyResearchItem(
                run_id=run.run_id,
                trading_date=run.trading_date,
                symbol=symbol,
                rank=rank,
                screening_score=70,
                screening_status="WATCH",
                status=status,
                created_at=started,
            )
        )

    assert store.recover_stale_runs(
        now=started + timedelta(seconds=119),
        stale_after_seconds=120,
    ) == []
    assert store.recover_stale_runs(
        now=started + timedelta(seconds=120),
        stale_after_seconds=120,
    ) == [run.run_id]

    recovered = store.latest_run(run.trading_date)
    assert recovered is not None
    assert recovered.status == "FAILED"
    assert recovered.error_code == "DAILY_RESEARCH_INTERRUPTED"
    assert recovered.failed_symbols == 2
    assert {
        (item.status, item.error_code)
        for item in store.items(run.run_id)
    } == {("FAILED", "DAILY_RESEARCH_INTERRUPTED")}


def test_failed_records_never_persist_an_empty_error_code(tmp_path):
    now = datetime(2026, 7, 27, 5, 0, tzinfo=timezone.utc)
    store = DailyResearchStore(str(tmp_path / "ai.duckdb"))
    run = _running_record(now)
    store.start_run(run)
    store.save_item(
        DailyResearchItem(
            run_id=run.run_id,
            trading_date=run.trading_date,
            symbol="AAPL",
            rank=1,
            screening_score=70,
            screening_status="WATCH",
            status="FAILED",
            created_at=now,
            completed_at=now,
        )
    )
    store.finish_run(
        run.run_id,
        status="FAILED",
        completed=0,
        failed=1,
        at=now,
    )

    latest = store.latest_run(run.trading_date)
    assert latest is not None
    assert latest.error_code == "DAILY_RESEARCH_FAILED_UNCLASSIFIED"
    assert (
        store.items(run.run_id)[0].error_code
        == "DAILY_RESEARCH_ITEM_FAILED_UNCLASSIFIED"
    )



def test_deep_candidates_prioritize_verified_then_use_low_confidence_fallback():
    low = DailyCandidate(
        symbol="AAPL", rank=1, score=80, status="BENCH",
        source_quality_score=70, ai_score=None, tactical_score=None,
        data_confidence="低",
    )
    verified = DailyCandidate(
        symbol="MSFT", rank=2, score=70, status="WATCH",
        source_quality_score=70, ai_score=None, tactical_score=70,
        data_confidence="中",
    )
    avoid = DailyCandidate(
        symbol="NVDA", rank=3, score=30, status="AVOID_NOW",
        source_quality_score=70, ai_score=None, tactical_score=20,
        data_confidence="中",
    )

    selected = DailyResearchService._deep_candidates(
        [low, verified, avoid], deep_limit=2
    )

    assert [candidate.symbol for candidate in selected] == ["MSFT", "AAPL"]


def test_low_confidence_candidate_still_runs_deep_research(tmp_path, monkeypatch):
    import trader.daily_research as module

    candidate = DailyCandidate(
        symbol="AAPL",
        rank=1,
        score=70,
        status="BENCH",
        source_quality_score=70,
        ai_score=None,
        tactical_score=None,
        data_confidence="低",
        risk_flags=["缺少本地 K 线或技术共识"],
    )
    monkeypatch.setattr(
        module,
        "build_daily_candidates",
        lambda *args, **kwargs: [candidate],
    )
    store = DailyResearchStore(str(tmp_path / "ai.duckdb"))
    service = DailyResearchService(store, _Analyzer())

    run = service.run(
        ["AAPL"],
        trading_date="2026-07-27",
        now=datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc),
    )

    assert run.status == "COMPLETED"
    item = store.items(run.run_id)[0]
    assert item.status == "COMPLETED"
    assert "缺少本地 K 线或技术共识" in item.risks
    assert any("TradingAgents 独立获取市场资料" in risk for risk in item.risks)


def test_no_eligible_deep_candidates_records_explicit_error(tmp_path, monkeypatch):
    import trader.daily_research as module

    anchor = DailyCandidate(
        symbol="SPY",
        rank=1,
        score=62,
        status="MARKET_ANCHOR",
        source_quality_score=62,
        ai_score=None,
        tactical_score=None,
        data_confidence="低",
    )
    monkeypatch.setattr(
        module,
        "build_daily_candidates",
        lambda *args, **kwargs: [anchor],
    )
    store = DailyResearchStore(str(tmp_path / "ai.duckdb"))
    service = DailyResearchService(store, _Analyzer())

    run = service.run(
        ["SPY"],
        trading_date="2026-07-27",
        now=datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc),
    )

    assert run.status == "FAILED"
    assert run.error_code == "NO_ELIGIBLE_DEEP_CANDIDATES"
