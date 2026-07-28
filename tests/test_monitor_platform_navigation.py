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
    assert "旧版自定义 Agent 已从主导航移除" in source
