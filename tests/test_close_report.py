"""收盘报告：复盘 + AI 研究合并，且复盘永远发得出去。

合并的代价是要等研究批次。这里的核心约束是：研究批次拖延或整批失败，绝不能
连累复盘——复盘只依赖本地账本，本来必定算得出来。
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from trader.close_report import (
    PendingCloseReport,
    build_close_report,
    build_direction_review_line,
    build_health_line,
    build_overnight_risk_line,
    build_unfilled_plans_line,
    research_wait_deadline,
)

NOW = datetime(2026, 8, 3, 20, 5, tzinfo=timezone.utc)
SESSION = {"open": 600.0, "close": 606.0, "high": 608.0, "low": 598.0}


# ── 合并与降级 ───────────────────────────────────────────────────────────────


def test_full_report_has_both_halves():
    note = build_close_report(
        trading_date="2026-08-03",
        review_body="盈亏 +100",
        research_body="🟢 NTNX BUY",
    )
    assert "今天怎么样" in note.body
    assert "明天做什么" in note.body
    assert "盈亏 +100" in note.body
    assert "NTNX" in note.body


def test_review_still_ships_when_research_is_missing():
    """研究批次失败不能把复盘一起拖没。"""
    note = build_close_report(
        trading_date="2026-08-03",
        review_body="盈亏 +100",
        research_missing_reason="研究批次到 17:00 ET 仍未完成",
    )
    assert "盈亏 +100" in note.body
    assert "17:00 ET 仍未完成" in note.body


def test_missing_research_says_why_not_just_omits_it():
    """"没有明日计划"是系统没跑出来，还是模型认为无机会？两者含义完全不同。"""
    note = build_close_report(
        trading_date="2026-08-03",
        review_body="x",
        research_missing_reason="今天没有研究批次记录",
    )
    assert "不代表模型认为明天无机会" in note.body


def test_one_report_per_day():
    note = build_close_report(trading_date="2026-08-03", review_body="x")
    assert note.dedupe_key == "close_report:2026-08-03"


# ── 方向复盘（闭环最后一环）──────────────────────────────────────────────────


def test_direction_review_scores_the_morning_call():
    line = build_direction_review_line(
        {"bias": "偏多，但只做回踩确认后的多头"}, SESSION
    )
    assert "早上判断" in line
    assert "晨报方向复盘" in line


def test_direction_review_is_explicit_when_no_call_was_recorded():
    line = build_direction_review_line(None, SESSION)
    assert "没有记录到" in line


def test_direction_review_is_explicit_without_market_data():
    line = build_direction_review_line({"bias": "中性"}, None)
    assert "无法评分" in line


def test_hedged_wording_is_not_scored_as_bullish():
    """重大事件日晨报会说"事件前观望，不抢多空"，子串匹配会先撞上"多"。"""
    from trader.brief_review import _normalize_bias

    assert _normalize_bias("事件前观望，不抢多空；事件后只跟随确认方向") == "neutral"
    assert _normalize_bias("中性震荡，先等开盘区间给方向") == "neutral"
    assert _normalize_bias("偏多，但只做回踩确认后的多头") == "bullish"
    assert _normalize_bias("偏空/防守，反弹不过关键位再考虑空头") == "bearish"


# ── 其余补充内容 ─────────────────────────────────────────────────────────────


def test_unfilled_plans_are_accounted_for():
    """INVALIDATED 不走盘中推送，但读者仍要知道计划最后怎么了。"""

    class _R:
        symbol = "SNOW"

        class state:
            value = "INVALIDATED"

    line = build_unfilled_plans_line([_R()])
    assert "SNOW" in line
    assert "到期未触发" in line


def test_no_unfilled_plans_produces_nothing():
    assert build_unfilled_plans_line([]) == ""


def test_overnight_risk_flags_earnings():
    class _P:
        def __init__(self, symbol):
            self.symbol = symbol

    line = build_overnight_risk_line([_P("VLO")], {"VLO": "本周四盘后"})
    assert "VLO" in line
    assert "财报" in line


def test_flat_overnight_is_stated_not_omitted():
    assert "空仓过夜" in build_overnight_risk_line([])


def test_health_line_confirms_the_engine_is_alive():
    """"没消息就是没事"只在系统确实活着时成立。"""
    line = build_health_line(tick_count=780, halted=False)
    assert "780" in line
    assert "无熔断" in line


def test_health_line_surfaces_a_halt():
    line = build_health_line(tick_count=10, halted=True, halt_reason="日内回撤 -5%")
    assert "日内回撤 -5%" in line


# ── 等待窗口 ─────────────────────────────────────────────────────────────────


def test_pending_report_expires_after_the_deadline():
    pending = PendingCloseReport(
        trading_date="2026-08-03",
        review_body="x",
        deadline=NOW + timedelta(minutes=55),
    )
    assert not pending.expired(NOW)
    assert not pending.expired(NOW + timedelta(minutes=54))
    assert pending.expired(NOW + timedelta(minutes=55))


def test_deadline_leaves_room_but_not_forever():
    deadline = research_wait_deadline(NOW)
    assert timedelta(minutes=30) <= deadline - NOW <= timedelta(hours=2)
