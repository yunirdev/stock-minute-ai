from datetime import datetime, timedelta, timezone
import threading
import time

import duckdb
import pytest

from trader.operations_observability import (
    ActionJobResult,
    AsyncButtonActionRunner,
    BUTTON_ACTIONS,
    ButtonActionAuditStore,
    button_contract_manifest,
    explain_order,
    queue_view,
    render_order_explanation_html,
)

NOW = datetime(2026, 7, 27, 20, 0, tzinfo=timezone.utc)


def _wait_for_terminal(store, action_run_id: str) -> dict:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        row = store.get(action_run_id)
        if row and row["state"] != "BUSY":
            return row
        time.sleep(0.01)
    raise AssertionError("async UI action did not reach a terminal state")


def test_all_17_button_actions_have_unique_complete_contracts():
    manifest = button_contract_manifest()
    assert len(BUTTON_ACTIONS) == 17
    assert len({item["action_id"] for item in manifest}) == 17
    assert all(item["label"] and item["category"] for item in manifest)
    assert sum(item["external_send"] for item in manifest) == 2
    assert not any(
        item["action_id"].startswith("agent.")
        or item["action_id"] == "pool.send_decision"
        for item in manifest
    )


@pytest.mark.parametrize("state", ["SUCCESS", "EMPTY", "ERROR"])
def test_every_action_supports_busy_terminal_and_duplicate_click(tmp_path, state):
    store = ButtonActionAuditStore(tmp_path / "trade.duckdb")
    for index, contract in enumerate(BUTTON_ACTIONS):
        request_id = f"{state}-{index}"
        busy = store.begin(contract.action_id, request_id, now=NOW)
        duplicate_busy = store.begin(contract.action_id, request_id, now=NOW)
        assert busy == duplicate_busy
        assert busy["state"] == "BUSY"
        terminal = store.finish(
            busy["action_run_id"],
            state=state,
            result_ref=f"result-{index}" if state == "SUCCESS" else "",
            user_message={
                "SUCCESS": "完成",
                "EMPTY": "没有可用数据",
                "ERROR": "操作失败，请查看审计记录",
            }[state],
            now=NOW + timedelta(seconds=1),
        )
        assert terminal["state"] == state
        assert store.begin(contract.action_id, request_id, now=NOW) == terminal


def test_async_runner_keeps_busy_and_deduplicates_until_worker_finishes(tmp_path):
    store = ButtonActionAuditStore(tmp_path / "trade.duckdb")
    runner = AsyncButtonActionRunner(store)
    entered = threading.Event()
    release = threading.Event()
    calls = 0

    def work():
        nonlocal calls
        calls += 1
        entered.set()
        assert release.wait(timeout=3)
        return {"result": "ready"}

    busy = runner.start("overview.run_research", work)
    assert entered.wait(timeout=3)
    duplicate = runner.start("overview.run_research", work)

    assert busy["state"] == "BUSY"
    assert duplicate == busy
    assert store.get(busy["action_run_id"])["state"] == "BUSY"
    assert calls == 1

    release.set()
    terminal = _wait_for_terminal(store, busy["action_run_id"])
    assert terminal["state"] == "SUCCESS"
    assert terminal["result_ref"].startswith("ui-job:overview.run_research:")


@pytest.mark.parametrize(
    ("work", "expected_state"),
    [
        (lambda: False, "EMPTY"),
        (lambda: [], "EMPTY"),
        (lambda: (_ for _ in ()).throw(RuntimeError("boom")), "ERROR"),
    ],
)
def test_async_runner_records_real_worker_outcome(tmp_path, work, expected_state):
    store = ButtonActionAuditStore(tmp_path / "trade.duckdb")
    runner = AsyncButtonActionRunner(store)

    busy = runner.start("maintenance.run", work)
    terminal = _wait_for_terminal(store, busy["action_run_id"])

    assert terminal["state"] == expected_state
    if expected_state == "ERROR":
        assert terminal["user_message"] == "操作失败: RuntimeError"


def test_async_runner_accepts_explicit_terminal_result(tmp_path):
    store = ButtonActionAuditStore(tmp_path / "trade.duckdb")
    runner = AsyncButtonActionRunner(store)

    busy = runner.start(
        "overview.run_research",
        lambda: ActionJobResult(state="ERROR", user_message="研究失败: TEST"),
    )
    terminal = _wait_for_terminal(store, busy["action_run_id"])

    assert terminal["state"] == "ERROR"
    assert terminal["user_message"] == "研究失败: TEST"


def test_async_runner_serializes_different_actions_instead_of_racing(tmp_path):
    """Two distinct actions queued close together must never run concurrently.

    Before the shared-queue rewrite each action got its own thread, so two
    different action_ids triggered around the same time could execute in
    parallel and contend for the same DuckDB file.
    """
    store = ButtonActionAuditStore(tmp_path / "trade.duckdb")
    runner = AsyncButtonActionRunner(store)
    active_concurrently = 0
    max_concurrent = 0
    lock = threading.Lock()

    def make_work():
        def work():
            nonlocal active_concurrently, max_concurrent
            with lock:
                active_concurrently += 1
                max_concurrent = max(max_concurrent, active_concurrently)
            time.sleep(0.05)
            with lock:
                active_concurrently -= 1
            return {"ok": True}

        return work

    first = runner.start("overview.run_research", make_work())
    second = runner.start("maintenance.run", make_work())
    third = runner.start("discord.send", make_work())

    _wait_for_terminal(store, first["action_run_id"])
    _wait_for_terminal(store, second["action_run_id"])
    _wait_for_terminal(store, third["action_run_id"])

    assert max_concurrent == 1


def test_queue_snapshot_reports_running_and_queued_tasks(tmp_path):
    store = ButtonActionAuditStore(tmp_path / "trade.duckdb")
    runner = AsyncButtonActionRunner(store)
    entered = threading.Event()
    release = threading.Event()

    def blocking_work():
        entered.set()
        assert release.wait(timeout=3)
        return {"ok": True}

    first = runner.start("overview.run_research", blocking_work)
    assert entered.wait(timeout=3)
    second = runner.start("maintenance.run", lambda: {"ok": True})

    snapshot = runner.queue_snapshot()
    assert snapshot["current"]["run_id"] == first["action_run_id"]
    assert snapshot["current"]["action_id"] == "overview.run_research"
    assert len(snapshot["queued"]) == 1
    assert snapshot["queued"][0]["run_id"] == second["action_run_id"]
    assert snapshot["queued"][0]["action_id"] == "maintenance.run"

    release.set()
    _wait_for_terminal(store, first["action_run_id"])
    _wait_for_terminal(store, second["action_run_id"])

    final_snapshot = runner.queue_snapshot()
    assert final_snapshot["current"] is None
    assert final_snapshot["queued"] == []


def test_store_recent_filters_by_category_and_action_id(tmp_path):
    store = ButtonActionAuditStore(tmp_path / "trade.duckdb")
    a = store.begin("discord.send", "req-a", now=NOW)
    store.finish(
        a["action_run_id"], state="SUCCESS", result_ref="r-a",
        user_message="ok", now=NOW + timedelta(seconds=2),
    )
    b = store.begin("maintenance.run", "req-b", now=NOW + timedelta(seconds=1))
    store.finish(
        b["action_run_id"], state="EMPTY", result_ref="",
        user_message="没有可用结果", now=NOW + timedelta(seconds=3),
    )

    discord_only = store.recent(category="discord")
    assert {row["action_id"] for row in discord_only} == {"discord.send"}
    assert discord_only[0]["label"] == "发送 Discord 简报"
    assert discord_only[0]["duration_seconds"] == pytest.approx(2.0)

    by_action = store.recent(action_id="maintenance.run")
    assert {row["action_id"] for row in by_action} == {"maintenance.run"}

    everything = store.recent()
    assert {row["action_id"] for row in everything} == {"discord.send", "maintenance.run"}


def test_average_duration_seconds_uses_completed_history_only(tmp_path):
    store = ButtonActionAuditStore(tmp_path / "trade.duckdb")
    assert store.average_duration_seconds("maintenance.run") is None

    for i in range(3):
        run = store.begin("maintenance.run", f"req-{i}", now=NOW)
        store.finish(
            run["action_run_id"], state="SUCCESS", result_ref="r",
            user_message="ok", now=NOW + timedelta(seconds=10),
        )
    still_busy = store.begin("maintenance.run", "req-busy", now=NOW)

    avg = store.average_duration_seconds("maintenance.run")
    assert avg == pytest.approx(10.0)
    assert still_busy["state"] == "BUSY"  # unfinished run must not skew the average


def test_queue_view_computes_cumulative_eta(tmp_path):
    store = ButtonActionAuditStore(tmp_path / "trade.duckdb")
    runner = AsyncButtonActionRunner(store)
    entered = threading.Event()
    release = threading.Event()

    def blocking_work():
        entered.set()
        assert release.wait(timeout=3)
        return {"ok": True}

    first = runner.start("overview.run_research", blocking_work)
    assert entered.wait(timeout=3)
    second = runner.start("maintenance.run", lambda: {"ok": True})

    view = queue_view(runner, store, default_duration=5.0)
    assert len(view["items"]) == 2
    running, queued = view["items"]
    assert running["status"] == "RUNNING"
    assert queued["status"] == "QUEUED"
    assert queued["eta_seconds"] > running["eta_seconds"]
    assert view["total_seconds"] == pytest.approx(queued["eta_seconds"])

    release.set()
    _wait_for_terminal(store, first["action_run_id"])
    _wait_for_terminal(store, second["action_run_id"])


def test_order_explanation_layers_versions_risk_order_and_fills(tmp_path):
    db_path = tmp_path / "trade.duckdb"
    connection = duckdb.connect(str(db_path))
    connection.execute("CREATE TABLE trade_plans(plan_id TEXT, created_at TIMESTAMPTZ)")
    connection.execute(
        """
        CREATE TABLE order_intents(
            intent_id TEXT, plan_id TEXT, decision_id TEXT,
            candidate_plan_id TEXT, final_plan_id TEXT,
            position_plan_id TEXT, evidence_refs_json TEXT,
            updated_at TIMESTAMPTZ
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE plan_risk_events(
            plan_id TEXT, decision_id TEXT, result TEXT, ts TIMESTAMPTZ
        )
        """
    )
    connection.execute(
        "CREATE TABLE fills(intent_id TEXT, fill_id TEXT, fill_time TIMESTAMPTZ)"
    )
    connection.execute("INSERT INTO trade_plans VALUES ('plan-1', ?)", [NOW])
    connection.execute(
        """
        INSERT INTO order_intents VALUES
        ('intent-1','plan-1','decision-1','candidate-1','final-1',
         'position-1','["snapshot-1","research-1"]',?)
        """,
        [NOW],
    )
    connection.execute(
        "INSERT INTO plan_risk_events VALUES ('plan-1','decision-1','PASS',?)",
        [NOW],
    )
    connection.execute(
        "INSERT INTO fills VALUES ('intent-1','fill-1',?)",
        [NOW],
    )
    connection.close()

    explanation = explain_order(db_path, "plan-1")
    assert explanation["status"] == "SUCCESS"
    assert explanation["sections"]["sources_and_snapshots"] == [
        "snapshot-1",
        "research-1",
    ]
    assert explanation["sections"]["plan_versions"]["final_plan_id"] == "final-1"
    assert explanation["sections"]["risk"][0]["result"] == "PASS"
    assert explanation["sections"]["fills"][0]["fill_id"] == "fill-1"
    rendered = render_order_explanation_html(explanation)
    assert "来源与快照" in rendered
    assert "candidate-1" in rendered


def test_order_explanation_layers_position_lifecycle_when_present(tmp_path):
    db_path = tmp_path / "trade.duckdb"
    connection = duckdb.connect(str(db_path))
    connection.execute("CREATE TABLE trade_plans(plan_id TEXT, created_at TIMESTAMPTZ)")
    connection.execute(
        """
        CREATE TABLE order_intents(
            intent_id TEXT, plan_id TEXT, decision_id TEXT,
            candidate_plan_id TEXT, final_plan_id TEXT,
            position_plan_id TEXT, evidence_refs_json TEXT,
            updated_at TIMESTAMPTZ
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE plan_risk_events(
            plan_id TEXT, decision_id TEXT, result TEXT, ts TIMESTAMPTZ
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE position_plan_heads(
            position_plan_id TEXT, symbol TEXT, side TEXT,
            source_trade_plan_id TEXT, initial_fill_id TEXT,
            initial_entry_price DOUBLE, initial_quantity DOUBLE,
            current_version INTEGER, current_version_id TEXT, status TEXT,
            created_at TIMESTAMPTZ, updated_at TIMESTAMPTZ
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE position_plan_versions(
            version_id TEXT, position_plan_id TEXT, version INTEGER,
            parent_version_id TEXT, symbol TEXT, side TEXT, status TEXT,
            source_trade_plan_id TEXT, initial_fill_id TEXT,
            initial_entry_price DOUBLE, initial_quantity DOUBLE,
            open_quantity DOUBLE, average_entry_price DOUBLE,
            stop_loss DOUBLE, take_profit DOUBLE,
            invalidation_rules_json TEXT, change_reason TEXT, created_at TIMESTAMPTZ
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE invalidation_events(
            event_id TEXT, dedupe_key TEXT, position_plan_id TEXT,
            position_plan_version_id TEXT, symbol TEXT, event_type TEXT,
            source TEXT, source_event_id TEXT, rule_id TEXT,
            as_of TIMESTAMPTZ, observed_at TIMESTAMPTZ, facts_json TEXT,
            evidence_refs_json TEXT, recorded_at TIMESTAMPTZ
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE position_adjustments(
            adjustment_id TEXT, event_id TEXT, position_plan_id TEXT,
            from_version_id TEXT, to_version_id TEXT, action TEXT, status TEXT,
            quantity DOUBLE, limit_price DOUBLE,
            previous_stop_loss DOUBLE, new_stop_loss DOUBLE,
            order_plan_id TEXT, order_intent_id TEXT, order_idempotency_key TEXT,
            created_at TIMESTAMPTZ, updated_at TIMESTAMPTZ
        )
        """
    )
    connection.execute("INSERT INTO trade_plans VALUES ('plan-1', ?)", [NOW])
    connection.execute(
        """
        INSERT INTO order_intents VALUES
        ('intent-1','plan-1','decision-1','candidate-1','final-1',
         'position-1','["snapshot-1"]',?)
        """,
        [NOW],
    )
    connection.execute(
        """
        INSERT INTO position_plan_heads VALUES
        ('position-1','AAPL','LONG','plan-1','fill-1',100.0,10.0,
         1,'version-1','ACTIVE',?,?)
        """,
        [NOW, NOW],
    )
    connection.execute(
        """
        INSERT INTO position_plan_versions VALUES
        ('version-1','position-1',1,NULL,'AAPL','LONG','ACTIVE','plan-1','fill-1',
         100.0,10.0,10.0,100.0,95.0,110.0,'{}','initial fill',?)
        """,
        [NOW],
    )
    connection.execute(
        """
        INSERT INTO invalidation_events VALUES
        ('event-1','dedupe-1','position-1','version-1','AAPL','PRICE_STOP',
         'broker','src-1','rule-1',?,?,'{}','[]',?)
        """,
        [NOW, NOW, NOW],
    )
    connection.execute(
        """
        INSERT INTO position_adjustments VALUES
        ('adj-1','event-1','position-1','version-1','version-2','EXIT','FILLED',
         10.0,94.5,95.0,NULL,'order-plan-1','intent-2','idem-1',?,?)
        """,
        [NOW, NOW],
    )
    connection.close()

    explanation = explain_order(db_path, "plan-1")
    assert explanation["status"] == "SUCCESS"
    assert explanation["sections"]["position_plan"]["symbol"] == "AAPL"
    assert explanation["sections"]["position_plan"]["stop_loss"] == 95.0
    assert explanation["sections"]["invalidation_events"][0]["event_type"] == "PRICE_STOP"
    assert explanation["sections"]["adjustments"][0]["action"] == "EXIT"
    rendered = render_order_explanation_html(explanation)
    assert "持仓计划版本" in rendered
    assert "失效事件" in rendered
    assert "自动调整" in rendered


def test_order_explanation_empty_error_and_html_escaping(tmp_path):
    assert explain_order(tmp_path / "missing.duckdb", "plan-1") == {
        "status": "EMPTY",
        "plan_id": "plan-1",
        "missing": ["audit_database"],
    }
    assert explain_order(tmp_path / "missing.duckdb", "") == {
        "status": "ERROR",
        "plan_id": "",
        "error_code": "PLAN_ID_REQUIRED",
    }
    rendered = render_order_explanation_html(
        {
            "status": "SUCCESS",
            "plan_id": "<script>alert(1)</script>",
            "sections": {"order": {"symbol": "<img src=x>"}},
        }
    )
    assert "<script>" not in rendered
    assert "<img src=x>" not in rendered
    assert "&lt;script&gt;" in rendered
