from datetime import datetime, timedelta, timezone

import duckdb
import pytest

from trader.operations_observability import (
    BUTTON_ACTIONS,
    ButtonActionAuditStore,
    button_contract_manifest,
    explain_order,
    render_order_explanation_html,
)

NOW = datetime(2026, 7, 27, 20, 0, tzinfo=timezone.utc)


def test_all_31_button_actions_have_unique_complete_contracts():
    manifest = button_contract_manifest()
    assert len(BUTTON_ACTIONS) == 31
    assert len({item["action_id"] for item in manifest}) == 31
    assert all(item["label"] and item["category"] for item in manifest)
    assert sum(item["external_send"] for item in manifest) == 4


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


def test_order_explanation_layers_versions_risk_order_and_fills(tmp_path):
    db_path = tmp_path / "trade.duckdb"
    connection = duckdb.connect(str(db_path))
    connection.execute(
        "CREATE TABLE trade_plans(plan_id TEXT, created_at TIMESTAMPTZ)"
    )
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
