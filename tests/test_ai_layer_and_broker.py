"""Regression tests for the AI decision layer and the Alpaca broker adapter.

This layer had zero test coverage while being the sole source of both stock
selection and trade direction. Every test here pins a bug that was found and
fixed by auditing this layer — they exist so those specific failure modes
cannot come back silently.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import duckdb
import pytest

from trader.ai.manager import (
    AgentManager,
    get_composite_scores_from_db,
    get_score_snapshots_from_db,
)
from trader.models import Advisory, OrderIntent, Side

NOW = datetime(2026, 7, 30, 16, 0, tzinfo=timezone.utc)


# ═══════════════════════════════════════════════════════════════════════════
# Alpaca broker: order quantity handling
# ═══════════════════════════════════════════════════════════════════════════


def _intent(qty: float, side: Side = Side.SELL) -> OrderIntent:
    return OrderIntent(
        intent_id="i1",
        signal_id="s1",
        symbol="AAPL",
        side=side,
        qty=qty,
        order_type="LMT",
        limit_price=100.0,
    )


class _FakeTradingClient:
    """Captures the request instead of hitting Alpaca."""

    def __init__(self) -> None:
        self.submitted = []

    def submit_order(self, req):
        self.submitted.append(req)
        return type("O", (), {"id": "broker-1"})()


def _broker_with_fake_client():
    from trader.broker.alpaca import AlpacaBroker

    broker = AlpacaBroker.__new__(AlpacaBroker)
    broker._client = _FakeTradingClient()
    broker._paper = True
    return broker


@pytest.mark.parametrize(
    ("plan_qty", "expected_submitted"),
    [
        (5.0, 5),      # whole share passes through
        (2.5, 2),      # fractional floors DOWN, never up
        (2.9, 2),      # still floors, no rounding to 3
    ],
)
def test_order_qty_floors_and_never_inflates_exposure(plan_qty, expected_submitted):
    broker = _broker_with_fake_client()

    broker.place_order(_intent(plan_qty))

    assert broker._client.submitted[0].qty == expected_submitted


@pytest.mark.parametrize("plan_qty", [0.4, 0.5, 0.99])
def test_sub_one_share_order_is_rejected_not_rounded_up_to_one(plan_qty):
    """The old code did max(int(qty), 1), turning a 0.5-share close into a
    1-share SELL — overselling the position and potentially opening a naked
    short. Refusing loudly is the only safe behavior for a limit order."""
    broker = _broker_with_fake_client()

    with pytest.raises(ValueError, match="ORDER_QTY_BELOW_ONE_SHARE"):
        broker.place_order(_intent(plan_qty))

    assert broker._client.submitted == []


def test_live_broker_still_refuses_to_submit():
    broker = _broker_with_fake_client()
    broker._paper = False

    with pytest.raises(RuntimeError, match="LIVE_ORDER_SUBMISSION_DISABLED"):
        broker.place_order(_intent(5.0))


def test_unknown_broker_status_maps_to_submitted_and_is_logged(caplog):
    """Unknown status defaults to SUBMITTED (keep polling rather than assume
    the order is done) — but it must be logged, otherwise a new terminal
    status from Alpaca would strand orders in _open_orders forever."""
    from trader.broker.alpaca import _map_status
    from trader.models import OrderStatus

    with caplog.at_level("WARNING"):
        assert _map_status("some_future_status") == OrderStatus.SUBMITTED

    assert any("未知订单状态" in r.getMessage() for r in caplog.records)


def test_known_terminal_statuses_map_correctly():
    from trader.broker.alpaca import _map_status
    from trader.models import OrderStatus

    assert _map_status("filled") == OrderStatus.FILLED
    assert _map_status("partially_filled") == OrderStatus.PARTIAL
    assert _map_status("rejected") == OrderStatus.REJECTED
    assert _map_status("canceled") == OrderStatus.CANCELLED
    assert _map_status("expired") == OrderStatus.CANCELLED


# ═══════════════════════════════════════════════════════════════════════════
# Composite score: must never fabricate a neutral 50
# ═══════════════════════════════════════════════════════════════════════════


def _seed_advisories(db_path: str, rows: list[dict]) -> None:
    manager = AgentManager(agents=[])
    manager._init_db(db_path)
    con = duckdb.connect(db_path)
    for i, row in enumerate(rows):
        con.execute(
            "INSERT INTO ai_advisories (advisory_id, kind, agent, payload_json, "
            "confidence, created_at, run_id, provider, model, is_stub, source, "
            "generated_by, schema_version, created_at_utc, is_fallback) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                f"adv-{i}", row["kind"], row["kind"],
                json.dumps({"symbol": row["symbol"], row["field"]: row["score"]}),
                0.8, row.get("created_at", NOW),
                row.get("run_id", "run-1"), row.get("provider", "ollama"),
                "m1", row.get("is_stub", False), "agent_manager", row["kind"], 1,
                row.get("created_at", NOW), row.get("is_fallback", False),
            ],
        )
    con.commit()
    con.close()


def test_symbol_without_any_weighted_agent_gets_no_score_not_a_fake_50(tmp_path):
    """bull_bear_debate is deliberately excluded from _AGENT_WEIGHTS. A symbol
    that only has that advisory used to receive a fabricated composite of
    50.0 — indistinguishable from a genuinely computed neutral score, and it
    fed straight into the research candidate shortlist."""
    db = str(tmp_path / "ai.duckdb")
    _seed_advisories(db, [
        {"symbol": "FAKE", "kind": "bull_bear_debate", "field": "final_score", "score": 88},
    ])

    scores = get_composite_scores_from_db(db)

    assert "FAKE" not in scores


def test_weighted_composite_is_computed_from_contributing_agents(tmp_path):
    db = str(tmp_path / "ai.duckdb")
    # macro weight .25, technical weight .05 → normalized over .30
    _seed_advisories(db, [
        {"symbol": "AAPL", "kind": "macro", "field": "macro_score", "score": 80},
        {"symbol": "AAPL", "kind": "technical", "field": "technical_score", "score": 20},
    ])

    scores = get_composite_scores_from_db(db)

    expected = (80 * 0.25 + 20 * 0.05) / 0.30
    assert scores["AAPL"] == pytest.approx(round(expected, 1))


def test_score_snapshot_rejects_stub_only_symbol(tmp_path):
    """Stub advisories carry is_stub=True so the safety gate can reject them;
    the snapshot must surface that rather than laundering it into a score."""
    db = str(tmp_path / "ai.duckdb")
    _seed_advisories(db, [
        {"symbol": "STUBBY", "kind": "macro", "field": "macro_score",
         "score": 50, "is_stub": True, "provider": "stub"},
    ])

    snapshots = get_score_snapshots_from_db(db)

    assert snapshots["STUBBY"].is_stub is True

    from trader.ai.safety import AIScorePolicy, AIScoreValidator

    result = AIScoreValidator(
        AIScorePolicy(min_ai_score=0, max_age_minutes=60 * 24 * 3650),
        lambda: NOW,
    ).validate(snapshots["STUBBY"])
    assert not result.valid
    assert result.reason_code == "AI_SCORE_STUB"


# ═══════════════════════════════════════════════════════════════════════════
# Provenance must fail closed when the LLM client is unknown
# ═══════════════════════════════════════════════════════════════════════════


def test_missing_llm_client_is_recorded_as_stub_so_safety_gate_rejects(tmp_path):
    """With self._client = None the old code wrote provider="nonetype" and
    is_stub=False — the AI safety gate only checks that provider is non-empty
    and is_stub is False, so an advisory of unknown origin passed the gate
    wearing a fake provider name."""
    db = str(tmp_path / "ai.duckdb")
    manager = AgentManager(agents=[])       # no client at all
    manager._init_db(db)
    manager._write_advisories(
        [Advisory(
            advisory_id="a1", kind="macro", agent="macro",
            payload={"symbol": "AAPL", "macro_score": 90},
            confidence=0.9, model="", created_at=NOW,
        )],
        db,
    )

    con = duckdb.connect(db, read_only=True)
    provider, is_stub = con.execute(
        "SELECT provider, is_stub FROM ai_advisories WHERE advisory_id='a1'"
    ).fetchone()
    con.close()

    assert provider == "unknown"
    assert is_stub is True

    snapshots = get_score_snapshots_from_db(db)
    from trader.ai.safety import AIScorePolicy, AIScoreValidator

    result = AIScoreValidator(
        AIScorePolicy(min_ai_score=0, max_age_minutes=60 * 24 * 3650),
        lambda: NOW,
    ).validate(snapshots["AAPL"])
    assert not result.valid


def test_algo_agents_stay_deterministic_even_without_client(tmp_path):
    """Algorithmic agents never call an LLM, so a missing client must not
    mark their output as stub — that would wrongly block valid quant data."""
    db = str(tmp_path / "ai.duckdb")
    manager = AgentManager(agents=[])
    manager._init_db(db)
    manager._write_advisories(
        [Advisory(
            advisory_id="q1", kind="quant", agent="quant",
            payload={"symbol": "AAPL", "quant_score": 70},
            confidence=0.9, model="", created_at=NOW,
        )],
        db,
    )

    con = duckdb.connect(db, read_only=True)
    provider, is_stub = con.execute(
        "SELECT provider, is_stub FROM ai_advisories WHERE advisory_id='q1'"
    ).fetchone()
    con.close()

    assert provider == "deterministic"
    assert is_stub is False


# ═══════════════════════════════════════════════════════════════════════════
# Timezone migration
# ═══════════════════════════════════════════════════════════════════════════


def test_naive_timestamp_tables_migrate_preserving_all_columns(tmp_path):
    """Legacy DBs stored tz-aware datetimes in naive TIMESTAMP columns, which
    DuckDB silently converts to local wall-clock time. Migration must convert
    the type AND keep every existing column value (an earlier version of this
    migration copied only the 6 base columns and blanked provenance)."""
    db = str(tmp_path / "legacy.duckdb")
    con = duckdb.connect(db)
    con.execute(
        "CREATE TABLE ai_advisories (advisory_id VARCHAR PRIMARY KEY, kind VARCHAR, "
        "agent VARCHAR, payload_json VARCHAR, confidence FLOAT, created_at TIMESTAMP, "
        "run_id VARCHAR, provider VARCHAR)"
    )
    con.execute(
        "INSERT INTO ai_advisories VALUES "
        "('a1','macro','macro','{}',0.5,TIMESTAMP '2026-07-30 09:00:00','run-9','ollama')"
    )
    con.commit()
    con.close()

    AgentManager(agents=[])._init_db(db)

    con = duckdb.connect(db, read_only=True)
    dtype = con.execute(
        "SELECT data_type FROM information_schema.columns "
        "WHERE table_name='ai_advisories' AND column_name='created_at'"
    ).fetchone()[0]
    run_id, provider = con.execute(
        "SELECT run_id, provider FROM ai_advisories WHERE advisory_id='a1'"
    ).fetchone()
    leftover = con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_name LIKE '%legacy_naive%'"
    ).fetchall()
    con.close()

    assert dtype == "TIMESTAMP WITH TIME ZONE"
    assert (run_id, provider) == ("run-9", "ollama")   # provenance survived
    assert leftover == []                               # temp table cleaned up


def test_init_db_is_idempotent(tmp_path):
    db = str(tmp_path / "ai.duckdb")
    _seed_advisories(db, [
        {"symbol": "AAPL", "kind": "macro", "field": "macro_score", "score": 80},
    ])

    manager = AgentManager(agents=[])
    manager._init_db(db)
    manager._init_db(db)

    con = duckdb.connect(db, read_only=True)
    count = con.execute("SELECT COUNT(*) FROM ai_advisories").fetchone()[0]
    con.close()
    assert count == 1


# ═══════════════════════════════════════════════════════════════════════════
# Ollama model selection
# ═══════════════════════════════════════════════════════════════════════════


def test_exact_model_match_wins_over_prefix_match():
    """Prefix matching means a request for qwen2.5:14b can silently land on
    qwen2.5:0.5b. An exact match must always be preferred."""
    from trader.ai.client import _pick_ollama_model

    available = [
        {"name": "qwen2.5:0.5b", "caps": []},
        {"name": "qwen2.5:14b", "caps": []},
    ]

    assert _pick_ollama_model(available, "qwen2.5:14b") == "qwen2.5:14b"


def test_prefix_fallback_still_works_when_exact_model_absent():
    from trader.ai.client import _pick_ollama_model

    available = [{"name": "qwen2.5:0.5b", "caps": []}]

    assert _pick_ollama_model(available, "qwen2.5:14b") == "qwen2.5:0.5b"


# ═══════════════════════════════════════════════════════════════════════════
# Degraded (fallback) agent output must be flagged, not laundered into a score
# ═══════════════════════════════════════════════════════════════════════════


class _DeadLLM:
    """An LLM client that is completely unavailable."""

    _model = "dead"

    def json_chat(self, system, user, model="", temperature=0.1):
        return {}

    def chat(self, *a, **k):
        return ""


def _ctx(score: float = 70.0):
    from trader.models import AgentContext, Candidate

    return AgentContext(
        candidates=[Candidate(symbol="AAPL", score=score, rank=1, reasons={"votes": {}})],
        plans=[], news=[], positions={}, equity=100_000.0,
        as_of=NOW, extra={},
    )


def test_llm_agent_marks_output_as_fallback_when_llm_is_down():
    """_llm_json tags its result with _is_fallback, but every agent builds a
    fresh payload dict and used to drop that flag — so Advisory.is_fallback was
    always False and hardcoded 50s counted as genuine LLM analysis."""
    from trader.ai.agents.macro import MacroAgent

    advisories = MacroAgent(client=_DeadLLM()).run(_ctx())

    assert advisories[0].payload["macro_score"] == 50
    assert advisories[0].is_fallback is True


def test_bull_bear_fallback_verdict_is_flagged():
    """With no LLM the judge fallback emits BUY purely from cand.score >= 65 —
    that must never be indistinguishable from a real debate verdict."""
    from trader.ai.agents.bull_bear import BullBearDebate

    advisories = BullBearDebate(client=_DeadLLM(), min_score=55, max_symbols=1).run(_ctx())

    assert advisories[0].payload["verdict"] == "BUY"    # fallback still fires
    assert advisories[0].is_fallback is True            # but it is labelled


def test_bull_bear_confidence_is_clamped_to_unit_range():
    """LLMs sometimes return confidence as 0-100; unclamped it becomes a 9500
    'score' downstream in AgentManager._run_one."""
    from trader.ai.agents.bull_bear import BullBearDebate

    class Overconfident(_DeadLLM):
        def json_chat(self, system, user, model="", temperature=0.1):
            return {"verdict": "BUY", "final_score": 80, "confidence": 95}

    advisories = BullBearDebate(client=Overconfident(), min_score=55, max_symbols=1).run(_ctx())

    assert 0.0 <= advisories[0].confidence <= 1.0


def test_algorithmic_agent_with_no_data_is_flagged_not_scored_50():
    """quant's _composite_score seeds at 50.0 and returns it untouched when
    every data source fails — 'nothing computed' must not look like 'neutral'."""
    from unittest.mock import patch

    from trader.ai.agents.quant import QuantAgent

    with patch("trader.ai.agents.quant._fetch_daily", return_value=None), \
         patch("trader.data_cache.get_bars", return_value=None):
        advisories = QuantAgent().run(_ctx())

    assert advisories[0].payload["quant_score"] == 50.0
    assert advisories[0].payload["factors_used"] == []
    assert advisories[0].is_fallback is True


def test_all_fallback_symbol_produces_no_snapshot_at_all(tmp_path):
    """End-to-end consequence: if every contributing agent degraded, the symbol
    must drop out entirely rather than surface a confident-looking 50."""
    db = str(tmp_path / "ai.duckdb")
    _seed_advisories(db, [
        {"symbol": "DEGRADED", "kind": "macro", "field": "macro_score",
         "score": 50, "is_fallback": True},
        {"symbol": "DEGRADED", "kind": "technical", "field": "technical_score",
         "score": 50, "is_fallback": True},
    ])

    snapshots = get_score_snapshots_from_db(db)

    assert "DEGRADED" not in snapshots


def test_partially_degraded_symbol_scores_only_on_healthy_agents(tmp_path):
    db = str(tmp_path / "ai.duckdb")
    _seed_advisories(db, [
        {"symbol": "AAPL", "kind": "macro", "field": "macro_score",
         "score": 50, "is_fallback": True},          # degraded, must be ignored
        {"symbol": "AAPL", "kind": "technical", "field": "technical_score",
         "score": 90, "is_fallback": False},         # healthy, must drive score
    ])

    snap = get_score_snapshots_from_db(db)["AAPL"]

    assert snap.score == 90.0            # not diluted by the fallback 50
    assert snap.contributor_count == 1
    assert snap.fallback_count == 1


# ═══════════════════════════════════════════════════════════════════════════
# Allocator must emit whole-share quantities (limit orders require them)
# ═══════════════════════════════════════════════════════════════════════════


def _tp(symbol="AAPL", side=Side.BUY, action="OPEN", entry=200.0):
    from trader.models import TradePlan

    return TradePlan(
        plan_id=f"p-{symbol}", symbol=symbol, side=side, action=action,
        entry_price=entry, stop_loss=entry * 0.95, take_profit=entry * 1.15,
        confidence=0.9,
    )


def _allocator():
    from trader.allocator import EqualWeightAllocator

    return EqualWeightAllocator(max_position_pct=0.2)


def test_allocated_qty_is_always_whole_shares():
    """Risk approves plan.qty and the broker submits it — if the allocator
    emits fractions the broker floors them, so risk validates a quantity that
    is never actually submitted."""
    out = _allocator().allocate([_tp()], equity=100_000.0, positions={})

    assert out[0].qty == int(out[0].qty)


def test_position_too_small_to_close_is_dropped_not_sent_to_broker():
    """A 0.5-share position cannot be closed with a limit order. Dropping it
    at allocation time keeps it out of the broker, where a raised error would
    count toward the consecutive-failure circuit breaker."""
    from trader.models import Position

    positions = {"TSLA": Position(symbol="TSLA", qty=0.5, avg_entry_px=300.0)}
    plan = _tp("TSLA", side=Side.SELL, action="CLOSE", entry=300.0)

    out = _allocator().allocate([plan], equity=100_000.0, positions=positions)

    assert out == []


def test_fractional_position_closes_the_whole_share_portion():
    from trader.models import Position

    positions = {"AAPL": Position(symbol="AAPL", qty=2.5, avg_entry_px=150.0)}
    plan = _tp("AAPL", side=Side.SELL, action="REDUCE")

    out = _allocator().allocate([plan], equity=100_000.0, positions=positions)

    assert out[0].qty == 2.0   # floor, never 3 (would oversell into a short)


def test_allocation_below_one_share_is_skipped():
    """Small account + very expensive stock — must not become a 1-share order
    far exceeding the intended position size."""
    plan = _tp("BRKA", entry=700_000.0)

    out = _allocator().allocate([plan], equity=10_000.0, positions={})

    assert out == []


def test_closing_plans_are_never_truncated_by_buying_power():
    """Closes release exposure; they must not compete with new opens for
    buying power or be cut by the max-open-plans limit."""
    from trader.allocator import EqualWeightAllocator
    from trader.models import Position

    positions = {"AAPL": Position(symbol="AAPL", qty=10.0, avg_entry_px=150.0)}
    close_plan = _tp("AAPL", side=Side.SELL, action="REDUCE")
    open_plan = _tp("MSFT", entry=100.0)

    out = EqualWeightAllocator(max_position_pct=1.0, max_open_plans=1).allocate(
        [close_plan, open_plan], equity=1_000.0, positions=positions,
    )

    assert any(p.symbol == "AAPL" and p.qty == 10.0 for p in out)


def test_stub_client_returns_neutral_score_and_is_identifiable():
    """The stub exists for offline runs, but its output must be traceable as
    stub so it never silently becomes a real trading signal."""
    from trader.ai.client import StubLLMClient

    client = StubLLMClient()

    assert client.json_chat("s", "u")["score"] == 50
    assert type(client).__name__ == "StubLLMClient"
