import re
from pathlib import Path


MONITOR = Path(__file__).resolve().parents[1] / "trader" / "monitor_nice.py"


def test_primary_navigation_matches_platform_responsibilities() -> None:
    source = MONITOR.read_text(encoding="utf-8")

    expected = (
        '_nav_item("overview", "📊", "今日总览")',
        '_nav_item("selection", "🔭", "机会中心")',
        '_nav_item("activity", "🧾", "交易与持仓")',
        '_nav_item("research", "🔬", "研究验证")',
        '_nav_item("operations", "⚙️", "系统运营")',
    )
    for item in expected:
        assert item in source

    primary_navigation = source[
        source.index('_nav_group("工作台")') : source.index(
            'content = ui.element("div").classes("qa-content")'
        )
    ]
    assert primary_navigation.count("_nav_item(") == 5
    assert '"决策台"' not in primary_navigation
    assert '"风控"' not in primary_navigation
    assert '"维护"' not in primary_navigation


def test_operations_hub_retains_bounded_operational_views() -> None:
    source = MONITOR.read_text(encoding="utf-8")

    assert '"operations": _render_operations' in source
    assert 'system_link.on("click", lambda: _select("system"))' in source
    assert 'risk_link.on("click", lambda: _select("risk"))' in source
    assert 'maintenance_link.on("click", lambda: _select("maintenance"))' in source
    assert 'tasks_link.on("click", lambda: _select("tasks"))' in source
    assert "旧版自定义 Agent 已从主导航移除" in source


def test_task_center_is_registered_and_reachable() -> None:
    source = MONITOR.read_text(encoding="utf-8")

    assert '"tasks": _render_tasks' in source
    assert "def _render_tasks():" in source
    # Top bar badge reflects live queue state from anywhere in the app.
    assert "top_tasks.set_text(" in source
    assert 'top_tasks_stat.on("click", lambda: _select("tasks"))' in source
    # Primary sidebar stays at 5 items; Task Center is reached via the hub only.
    primary_navigation = source[
        source.index('_nav_group("工作台")') : source.index(
            'content = ui.element("div").classes("qa-content")'
        )
    ]
    assert primary_navigation.count("_nav_item(") == 5
    assert '"tasks"' not in primary_navigation


def test_legacy_agent_cockpit_is_not_reachable_from_nicegui() -> None:
    source = MONITOR.read_text(encoding="utf-8")

    assert '_select("cockpit")' not in source
    assert '"cockpit": _render_cockpit' not in source
    assert (
        "送到决策台"
        not in source[
            source.index("def _render_selection_pools()") : source.index(
                "def _render_risk()"
            )
        ]
    )


def test_background_actions_use_durable_async_runner() -> None:
    source = MONITOR.read_text(encoding="utf-8")
    action_ids = (
        "overview.refresh_regime",
        "overview.run_research",
        "discord.send",
        "discord.stock_analysis",
        "pool.full_scan",
        "pool.rebuild_all",
        "pool.long_term",
        "pool.decision",
        "maintenance.run",
    )

    for action_id in action_ids:
        assert re.search(
            rf'_audited_job_callback\(\s*"{re.escape(action_id)}"',
            source,
        )
    assert "Thread(" not in source


def test_auto_trade_control_is_session_scoped_and_defaults_off() -> None:
    source = MONITOR.read_text(encoding="utf-8")

    auto_trade_block = source[
        source.index(
            "# Execution authority is intentionally session-scoped"
        ) : source.index("score_in = _persist(")
    ]
    assert "value=False" in auto_trade_block
    assert "_persist(" not in auto_trade_block
    assert "sys_auto_trade" not in auto_trade_block
    assert "每次打开平台都默认关闭" in source
    assert "读取决策台的 AI 综合评分" not in source
