import ast
import re
from pathlib import Path

from trader.operations_observability import BUTTON_ACTIONS

ROOT = Path(__file__).parents[1]
MONITOR = ROOT / "trader" / "monitor_nice.py"
ACCEPTANCE = ROOT / "docs" / "CLOSED_LOOP_ACCEPTANCE.md"
TASK_BOARD = ROOT / "docs" / "MIGRATION_TASK_BOARD.md"


def _button_calls():
    tree = ast.parse(MONITOR.read_text(encoding="utf-8"))
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "button"
    ]


def test_nicegui_button_inventory_matches_acceptance_contract():
    source = MONITOR.read_text(encoding="utf-8")
    contract = ACCEPTANCE.read_text(encoding="utf-8")
    calls = _button_calls()
    inline_handlers = sum(
        any(keyword.arg == "on_click" for keyword in call.keywords)
        for call in calls
    )
    assigned_handlers = len(re.findall(r"\.on_click\(", source))
    span_match = re.search(
        r'_SPAN_OPTS\s*=\s*\[(?P<values>[^\]]+)\]',
        source,
    )

    assert span_match is not None
    span_actions = len(re.findall(r'"[^"]+"', span_match["values"]))
    assert len(calls) == 26
    assert inline_handlers == 11
    assert assigned_handlers == 15
    assert inline_handlers + assigned_handlers == len(calls)
    assert len(calls) - 1 + span_actions == 31
    assert "31 个按钮动作" in contract
    assert "实盘下单请改" not in source
    assert "自动实盘不受支持" in source
    assert 'os.getenv("QUANT_HOST", "127.0.0.1")' in source
    assert "host=_host if _web else None" in source
    assert '@app.get("/api/ui-actions")' in source
    assert '@app.get("/api/order-explanation/{plan_id}")' in source
    assert "最新订单完整解释" in source
    static_action_ids = {
        action.action_id
        for action in BUTTON_ACTIONS
        if not action.action_id.startswith("overview.span_")
    }
    assert all(
        re.search(
            rf'_audited_callback\(\s*"{re.escape(action_id)}"',
            source,
        )
        for action_id in static_action_ids
    )
    assert 'f"overview.span_{_sl.lower()}"' in source


def test_closed_loop_stages_and_next_task_are_authoritative():
    contract = ACCEPTANCE.read_text(encoding="utf-8")
    board = TASK_BOARD.read_text(encoding="utf-8")

    for stage in (
        "数据获取",
        "市场分析",
        "交易计划",
        "交易执行",
        "风险控制",
        "复盘反思",
        "策略迭代",
        "用户解释",
    ):
        assert stage in contract
    assert "| D01 | DONE | C06 |" in board
    assert "| D02 | DONE | D01 |" in board
    assert "| D03 | DONE | D02 |" in board
    assert "| D04 | BLOCKED | D03 |" in board
    assert "| F05 | DONE | F04 |" in board
    assert "| F06 | DONE | F05 |" in board
    assert "| F07 | DONE | F06 |" in board
    assert "| G01 | DONE | F07 |" in board
    assert "| G02 | DONE | G01 |" in board
    assert "| G03 | DONE | G02 |" in board
    assert "| G04 | DONE | G03 |" in board
    assert "| H01 | DONE | G04 |" in board
    assert "| H02 | DONE | H01 |" in board
    assert "| H03 | DONE | H02 |" in board
    assert "| H04 | DONE | H03 |" in board
    assert "| H05 | DONE | H04 |" in board
    assert "| H06 | DONE | H05 |" in board
    assert "| I01 | BLOCKED | H06 |" in board
    assert "| I01-ACCEL | DONE | H06 |" in board
    assert "| I02 | DONE | I01-ACCEL |" in board
    assert "| I03-ARCH | DONE | I02 |" in board
    assert "| I03 | BLOCKED | I01 + I02 + I03-ARCH |" in board
    assert "下一项：I01" in contract
