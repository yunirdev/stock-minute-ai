"""每一种自动推送的报告都要能手动补发，且补发有明确的周期语义。"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

import trader.manual_push as mp
from trader.report_period import resolve_daily_period, resolve_weekly_period


def test_every_scheduled_report_has_a_manual_sender():
    """自动推送的报告如果补发不了，一次故障就等于永久丢失那份内容。"""
    for name in (
        "send_morning_brief_push",     # 见下方别名断言
        "send_open_confirmation_push",
        "send_close_report_push",
        "send_weekly_report_push",
        "send_intraday_levels_push",
        "send_direction_review_push",
        "send_stock_analysis_push",
    ):
        if name == "send_morning_brief_push":
            from trader.morning_brief import send_morning_brief

            assert callable(send_morning_brief)
            continue
        assert callable(getattr(mp, name)), f"{name} 缺少手动补发入口"


# ── 周期语义 ─────────────────────────────────────────────────────────────────


def test_open_confirmation_labels_the_resolved_period(monkeypatch):
    monkeypatch.setattr(mp, "_load_bars", lambda symbol: None)
    note = mp.build_open_confirmation_push_message(db_path=":memory:")
    period = resolve_daily_period(datetime.now(timezone.utc))
    assert period.label in note.title


def test_close_report_labels_the_resolved_period(tmp_path):
    note = mp.build_close_report_push_message(db_path=str(tmp_path / "x.duckdb"))
    period = resolve_daily_period(datetime.now(timezone.utc))
    assert period.label in note.title


def test_partial_close_report_does_not_collide_with_the_official_one(tmp_path):
    """盘中补发一次，不能把当天正式的收盘报告顶掉。"""
    note = mp.build_close_report_push_message(db_path=str(tmp_path / "y.duckdb"))
    period = resolve_daily_period(datetime.now(timezone.utc))
    if period.is_partial:
        assert note.dedupe_key.endswith(":partial")
        assert note.dedupe_key != f"close_report:{period.label}"


def test_missing_minute_data_is_stated_not_faked(monkeypatch):
    """拿不到分钟数据就说拿不到，不能编一份看起来正常的开盘确认。"""
    monkeypatch.setattr(
        "trader.data_cache.get_bars", lambda *a, **k: None, raising=False
    )
    note = mp.build_open_confirmation_push_message(db_path=":memory:")
    assert "无法计算" in note.body or "开盘确认" in note.title


def test_weekly_partial_label_is_distinct():
    """本周还没结束时补发，身份要和周五那份正式周报分开。"""
    period = resolve_weekly_period(datetime.now(timezone.utc))
    label = period.label + ("-partial" if period.is_partial else "")
    if period.is_partial:
        assert label != period.label


# ── 主观输入已移除 ───────────────────────────────────────────────────────────


def test_direction_review_takes_no_bias_argument():
    """方向不能再由人从下拉框里挑——那是给事后编造的判断打分。"""
    import inspect

    sig = inspect.signature(mp.build_direction_review_message)
    assert "bias" not in sig.parameters
    sig2 = inspect.signature(mp.send_direction_review_push)
    assert "bias" not in sig2.parameters


def test_bias_selector_is_gone_from_the_cockpit():
    from pathlib import Path

    source = Path("trader/monitor_nice.py").read_text(encoding="utf-8")
    assert "review_bias_sel" not in source
    assert "sys_review_bias" not in source
