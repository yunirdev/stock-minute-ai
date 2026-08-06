import json
import subprocess
from datetime import datetime, timedelta, timezone

import duckdb
import pytest

from trader.daily_candidates import DailyCandidate
from trader.daily_research import (
    DailyResearchService,
    DailyResearchStore,
    ResearchAnalysis,
    TradingAgentsAdapter,
    TradingAgentsInvocation,
)
from trader.research_snapshot import ResearchSnapshotStore
from trader.tradingagents_worker import (
    _source_manifest,
    _validated_invocation,
)

NOW = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)


def _manifest():
    return (
        {
            "source": "tradingagents.market",
            "status": "OK",
            "as_of": NOW.isoformat(),
            "fetched_at": (NOW + timedelta(seconds=1)).isoformat(),
            "quality_score": 1.0,
            "coverage": ["ohlcv", "technical_indicators"],
            "payload_version": "tradingagents-source:v1",
            "failure_code": "",
            "metadata": {"configured_vendors": ["yfinance"]},
        },
    )


def _candidate():
    return DailyCandidate(
        symbol="AAPL",
        rank=1,
        score=80,
        status="ENTRY_READY",
        source_quality_score=80,
        ai_score=None,
        tactical_score=80,
        data_confidence="高",
    )


class ContextAnalyzer:
    provider = "fake-tradingagents"
    model = "model-v1"

    def __init__(self):
        self.invocations = []

    def describe(self):
        return {"provider": self.provider, "model": self.model}

    def analyze_with_context(self, symbol, trading_date, invocation, *, complexity: str = ""):
        self.invocations.append(invocation)
        return ResearchAnalysis(
            recommendation="BUY",
            score=85,
            confidence=0.8,
            thesis=f"{symbol}:{trading_date}",
            provider=self.provider,
            model=self.model,
            raw={"invocation": invocation.to_dict()},
            source_manifest=_manifest(),
        )


def _invocation():
    return TradingAgentsInvocation(
        invocation_id="ta-invocation-test",
        run_id="research-run",
        snapshot_id="snapshot-1",
        snapshot_content_hash="a" * 64,
        data_version="screening-inputs:v1:abc",
        model_version="model-config-v1",
        symbol="AAPL",
        trading_date="2026-07-27",
        data_cutoff=NOW,
        requested_at=NOW,
    )


def _worker_success(payload, *, invocation=None, started_at=None):
    returned = dict(invocation or payload["invocation"])
    started = started_at or datetime.fromisoformat(
        returned["requested_at"]
    )
    result = {
        "ok": True,
        "worker_contract_version": 1,
        "invocation": returned,
        "worker_started_at": started.isoformat(),
        "worker_completed_at": (
            started + timedelta(seconds=1)
        ).isoformat(),
        "decision": {
            "recommendation": "BUY",
            "confidence": 0.8,
        },
        "state": {"market_report": "positive"},
        "source_manifest": list(_manifest()),
    }
    return (
        "__TRADINGAGENTS_RESULT__="
        + json.dumps(result, separators=(",", ":"))
        + "\n"
    )


def test_daily_research_persists_snapshot_model_and_data_linkage(
    tmp_path,
    monkeypatch,
):
    import trader.daily_research as module

    monkeypatch.setattr(
        module,
        "build_daily_candidates",
        lambda *args, **kwargs: [_candidate()],
    )
    analyzer = ContextAnalyzer()
    store = DailyResearchStore(str(tmp_path / "ai.duckdb"))

    run = DailyResearchService(store, analyzer).run(
        ["AAPL"],
        trading_date="2026-07-27",
        now=NOW,
    )

    item = store.items(run.run_id)[0]
    invocation = analyzer.invocations[0]
    assert run.status == "COMPLETED"
    assert item.snapshot_id == invocation.snapshot_id
    assert item.data_version == invocation.data_version
    assert item.model_version == run.config_version
    assert item.invocation_id == invocation.invocation_id
    assert invocation.run_id == run.run_id
    assert invocation.snapshot_content_hash
    assert item.raw["invocation"] == invocation.to_dict()
    assert item.ta_snapshot_id.startswith("ta-snapshot-")
    output_snapshot = ResearchSnapshotStore(
        store.db_path
    ).replay(item.ta_snapshot_id)
    assert output_snapshot.run_id == run.run_id
    assert output_snapshot.payload_version == "tradingagents-output:v1"
    assert output_snapshot.source_manifest[0].source == (
        "tradingagents.market"
    )
    assert store.score_snapshots(NOW)["AAPL"].run_id == run.run_id


def test_daily_research_store_migrates_legacy_item_schema(tmp_path):
    path = tmp_path / "legacy.duckdb"
    conn = duckdb.connect(str(path))
    try:
        conn.execute(
            """
            CREATE TABLE daily_research_items (
                run_id VARCHAR,
                trading_date VARCHAR,
                symbol VARCHAR,
                rank INTEGER,
                screening_score DOUBLE,
                screening_status VARCHAR,
                status VARCHAR,
                recommendation VARCHAR,
                ai_score DOUBLE,
                confidence DOUBLE,
                thesis VARCHAR,
                risks_json VARCHAR,
                provider VARCHAR,
                model VARCHAR,
                error_code VARCHAR,
                created_at TIMESTAMPTZ,
                completed_at TIMESTAMPTZ,
                raw_json VARCHAR,
                PRIMARY KEY(run_id, symbol)
            )
            """
        )
    finally:
        conn.close()

    DailyResearchStore(str(path))
    conn = duckdb.connect(str(path), read_only=True)
    try:
        columns = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info('daily_research_items')"
            ).fetchall()
        }
    finally:
        conn.close()

    assert {
        "snapshot_id",
        "data_version",
        "model_version",
        "invocation_id",
    }.issubset(columns)


def test_subprocess_contract_is_echoed_and_validated(monkeypatch, tmp_path):
    import trader.daily_research as module

    python_path = tmp_path / "python.exe"
    python_path.touch()
    captured = {}

    def run(command, **kwargs):
        payload = json.loads(kwargs["input"])
        captured["payload"] = payload
        return subprocess.CompletedProcess(
            command,
            0,
            _worker_success(payload),
            "",
        )

    monkeypatch.setattr(module.subprocess, "run", run)
    result = TradingAgentsAdapter(
        python_executable=str(python_path),
        deep_model="model",
    ).analyze_with_context(
        "AAPL",
        "2026-07-27",
        _invocation(),
    )

    assert captured["payload"]["worker_contract_version"] == 1
    assert captured["payload"]["invocation"] == _invocation().to_dict()
    assert result.raw["invocation"] == _invocation().to_dict()
    assert result.raw["worker_contract_version"] == 1


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (
            lambda value: {**value, "snapshot_id": "wrong"},
            "TRADINGAGENTS_WORKER_INVOCATION_MISMATCH",
        ),
        (
            lambda value: value,
            "TRADINGAGENTS_WORKER_OUTPUT_STALE",
        ),
    ],
)
def test_subprocess_rejects_mismatched_or_stale_output(
    monkeypatch,
    tmp_path,
    mutation,
    error,
):
    import trader.daily_research as module

    python_path = tmp_path / "python.exe"
    python_path.touch()

    def run(command, **kwargs):
        payload = json.loads(kwargs["input"])
        invocation = mutation(payload["invocation"])
        started_at = None
        if error.endswith("STALE"):
            started_at = NOW - timedelta(seconds=1)
        return subprocess.CompletedProcess(
            command,
            0,
            _worker_success(
                payload,
                invocation=invocation,
                started_at=started_at,
            ),
            "",
        )

    monkeypatch.setattr(module.subprocess, "run", run)
    with pytest.raises(RuntimeError, match=error):
        TradingAgentsAdapter(
            python_executable=str(python_path)
        ).analyze_with_context(
            "AAPL",
            "2026-07-27",
            _invocation(),
        )


def test_subprocess_timeout_and_crash_are_explicit(monkeypatch, tmp_path):
    import trader.daily_research as module

    python_path = tmp_path / "python.exe"
    python_path.touch()
    adapter = TradingAgentsAdapter(
        python_executable=str(python_path),
        timeout_seconds=1,
    )

    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, 1)

    monkeypatch.setattr(module.subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match="TRADINGAGENTS_TIMEOUT"):
        adapter.analyze_with_context(
            "AAPL",
            "2026-07-27",
            _invocation(),
        )

    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command,
            1,
            "worker crashed",
            "",
        ),
    )
    with pytest.raises(RuntimeError, match="TRADINGAGENTS_WORKER_NO_RESULT"):
        adapter.analyze_with_context(
            "AAPL",
            "2026-07-27",
            _invocation(),
        )


def test_worker_rejects_invalid_invocation_contract():
    payload = {
        "worker_contract_version": 1,
        "invocation": _invocation().to_dict(),
    }

    assert _validated_invocation(
        payload,
        "AAPL",
        "2026-07-27",
    ) == _invocation().to_dict()
    with pytest.raises(ValueError, match="INVOCATION_SYMBOL_MISMATCH"):
        _validated_invocation(
            payload,
            "MSFT",
            "2026-07-27",
        )


def test_worker_registers_configured_external_sources_and_failures():
    config = {
        "data_vendors": {
            "core_stock_apis": "alpaca",
            "technical_indicators": "yfinance",
            "fundamental_data": "sec",
            "news_data": "finnhub",
            "macro_data": "fred",
            "prediction_markets": "polymarket",
        }
    }
    state = {
        "market_report": "market",
        "fundamentals_report": "fundamentals",
        "news_report": "news",
        "sentiment_report": "",
    }

    manifest = _source_manifest(
        config,
        state,
        _invocation().to_dict(),
        NOW + timedelta(seconds=1),
    )

    assert [item["source"] for item in manifest] == [
        "tradingagents.market",
        "tradingagents.fundamentals",
        "tradingagents.news",
        "tradingagents.sentiment",
    ]
    assert manifest[0]["metadata"]["configured_vendors"] == [
        "alpaca",
        "yfinance",
    ]
    assert manifest[2]["metadata"]["configured_vendors"] == [
        "finnhub",
        "fred",
        "polymarket",
    ]
    assert manifest[3]["status"] == "FAILED"
    assert manifest[3]["quality_score"] == 0.0
    assert manifest[3]["failure_code"] == "ANALYST_REPORT_MISSING"


def test_rerunning_same_day_research_serves_the_latest_run_not_nothing(
    tmp_path,
    monkeypatch,
):
    """手动重跑当天研究（"运行今日研究"按钮，force=True）是正常操作，不是数据
    损坏——两条同日 COMPLETED 记录曾经被当成"无法判断，直接返回空"，代价是
    runtime 的选股/AI 门槛在这种完全合法的场景下拿不到任何数据，一整天选不
    出候选（这正是 2026-08-05 那次事故的根因之一）。重跑之后应该采信最新
    一次的结果，不是把两次都作废。"""
    import trader.daily_research as module

    monkeypatch.setattr(
        module,
        "build_daily_candidates",
        lambda *args, **kwargs: [_candidate()],
    )
    store = DailyResearchStore(str(tmp_path / "ai.duckdb"))
    service = DailyResearchService(store, ContextAnalyzer())
    service.run(
        ["AAPL"],
        trading_date="2026-07-27",
        now=NOW,
    )
    second_run = service.run(
        ["AAPL"],
        trading_date="2026-07-27",
        now=NOW + timedelta(seconds=1),
        force=True,
    )

    snapshots = store.score_snapshots(NOW + timedelta(seconds=1))
    assert snapshots["AAPL"].run_id == second_run.run_id


def test_wrong_date_or_model_contract_is_not_consumed(
    tmp_path,
    monkeypatch,
):
    import trader.daily_research as module

    monkeypatch.setattr(
        module,
        "build_daily_candidates",
        lambda *args, **kwargs: [_candidate()],
    )
    store = DailyResearchStore(str(tmp_path / "ai.duckdb"))
    run = DailyResearchService(store, ContextAnalyzer()).run(
        ["AAPL"],
        trading_date="2026-07-27",
        now=NOW,
    )
    conn = duckdb.connect(store.db_path)
    try:
        conn.execute(
            "UPDATE daily_research_items SET model_version='wrong' "
            "WHERE run_id=?",
            [run.run_id],
        )
    finally:
        conn.close()

    assert store.score_snapshots(NOW) == {}
