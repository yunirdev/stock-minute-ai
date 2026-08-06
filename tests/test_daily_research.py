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

    def analyze(self, symbol: str, trading_date: str, *, complexity: str = ""):
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


def test_briefing_thesis_prefers_trader_plan_over_bare_decision():
    """decision 经常就是一个裁决词（比如字符串 "Overweight"），没有任何
    解释——真正的入场价/止损/仓位和推理在 state.trader_investment_plan
    里，说明栏的简报该取这个，不是裸的 decision。"""
    from trader.daily_research import _briefing_thesis

    decision = "Overweight"
    state = {
        "trader_investment_plan": (
            "**Action**: Buy\n**Entry Price**: 267.5\n**Stop Loss**: 235.0"
        ),
        "final_trade_decision": "**Rating**: Overweight\n**Price Target**: 310.0",
        "investment_plan": "x" * 3000,  # 完整辩论记录，太长，不该被选中
    }
    assert _briefing_thesis(decision, state) == state["trader_investment_plan"]


def test_briefing_thesis_falls_back_to_final_decision_then_bare_decision():
    from trader.daily_research import _briefing_thesis

    assert (
        _briefing_thesis("Overweight", {"final_trade_decision": "**Rating**: Overweight"})
        == "**Rating**: Overweight"
    )
    assert _briefing_thesis("Overweight", {}) == "Overweight"
    assert _briefing_thesis("Overweight", None) == "Overweight"


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

    # 现在每个候选都会被分析（不再是"前 deep_limit 名深挖、剩下的完全不
    # 分析"），FAIL 也会真的跑一次、真的失败，所以整体状态是
    # COMPLETED_WITH_ERRORS 而不是 COMPLETED。
    assert run.status == "COMPLETED_WITH_ERRORS"
    assert run.completed_symbols == 2
    assert run.failed_symbols == 1
    assert [item.status for item in store.items(run.run_id)] == [
        "COMPLETED",
        "COMPLETED",
        "FAILED",
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
    """research_target_date() 的"收盘后属于第二天"这条边界规则本身没变——
    即使批次触发窗口收窄成只剩盘前，一个批次到底算今天还是明天的研究，仍然
    要看它是不是在 close_hour_et:close_minute_et 之后跑的（手动 force 补跑
    收盘后也可能发生）。"""
    premarket = datetime(2026, 7, 27, 12, 30, tzinfo=timezone.utc)
    postclose = datetime(2026, 7, 27, 21, 0, tzinfo=timezone.utc)
    friday_postclose = datetime(2026, 7, 31, 21, 0, tzinfo=timezone.utc)

    assert in_daily_run_window(premarket)
    assert research_target_date(premarket) == "2026-07-27"
    assert research_target_date(postclose) == "2026-07-28"
    assert research_target_date(friday_postclose) == "2026-08-03"


def test_daily_run_window_is_premarket_only_starting_530_et():
    """收盘后窗口已经删掉——`start_if_due()` 一旦当天有一条 run（不论自动
    触发还是手动 force）就不会再跑第二次，实际观察下来收盘后那次几乎总是
    先发生，盘前窗口从没真正执行过，等于全天用一份隔了一夜的结论。现在只
    留盘前，5:30 ET 起（比原来 6:00 提前半小时），离真正开盘/下单最近。"""
    et_529 = datetime(2026, 7, 27, 9, 29, tzinfo=timezone.utc)   # 5:29 ET（夏令时 UTC-4）
    et_530 = datetime(2026, 7, 27, 9, 30, tzinfo=timezone.utc)   # 5:30 ET
    et_915 = datetime(2026, 7, 27, 13, 15, tzinfo=timezone.utc)  # 9:15 ET
    et_916 = datetime(2026, 7, 27, 13, 16, tzinfo=timezone.utc)  # 9:16 ET
    old_postclose = datetime(2026, 7, 27, 21, 0, tzinfo=timezone.utc)  # 17:00 ET，原来的收盘后窗口

    assert not in_daily_run_window(et_529)
    assert in_daily_run_window(et_530)
    assert in_daily_run_window(et_915)
    assert not in_daily_run_window(et_916)
    assert not in_daily_run_window(old_postclose)


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



def test_every_candidate_gets_analyzed_and_tagged_with_complexity(tmp_path, monkeypatch):
    """确保所有标的都有分析到——不再是"前 deep_limit 名深挖、剩下的完全不
    分析"的硬截断。排名只决定 complexity 档位（top3=HIGH/4-7=MEDIUM/8+=LIGHT），
    不再决定要不要分析，AVOID_NOW/MARKET_ANCHOR 也不例外。"""
    import trader.daily_research as module

    candidates = [
        DailyCandidate(
            symbol=f"SYM{i}", rank=i, score=90 - i,
            status="AVOID_NOW" if i >= 8 else "WATCH",
            source_quality_score=70, ai_score=None, tactical_score=None,
            data_confidence="中",
        )
        for i in range(1, 10)
    ]
    monkeypatch.setattr(
        module, "build_daily_candidates", lambda *args, **kwargs: candidates
    )
    store = DailyResearchStore(str(tmp_path / "ai.duckdb"))
    service = DailyResearchService(store, _Analyzer())

    run = service.run(
        [c.symbol for c in candidates],
        trading_date="2026-07-27",
        now=datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc),
    )

    assert run.status == "COMPLETED"
    assert run.total_symbols == len(candidates)
    items = {item.symbol: item for item in store.items(run.run_id)}
    assert len(items) == len(candidates)
    assert all(item.status == "COMPLETED" for item in items.values())
    assert items["SYM1"].complexity == module.COMPLEXITY_HIGH
    assert items["SYM3"].complexity == module.COMPLEXITY_HIGH
    assert items["SYM4"].complexity == module.COMPLEXITY_MEDIUM
    assert items["SYM7"].complexity == module.COMPLEXITY_MEDIUM
    assert items["SYM8"].complexity == module.COMPLEXITY_LIGHT
    assert items["SYM9"].complexity == module.COMPLEXITY_LIGHT


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


def test_market_anchor_candidate_still_gets_analyzed(tmp_path, monkeypatch):
    """以前 MARKET_ANCHOR/AVOID_NOW 会被 _deep_candidates 整个排除、跑出
    NO_ELIGIBLE_DEEP_CANDIDATES——现在"确保所有标的都有分析到"，这类候选
    也会拿到（轻量级）分析，不会被直接跳过。"""
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

    assert run.status == "COMPLETED"
    item = store.items(run.run_id)[0]
    assert item.status == "COMPLETED"
    assert item.complexity == module.COMPLEXITY_HIGH  # rank 1
