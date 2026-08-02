"""开盘确认：晨报给的关键触发位，开盘 15 分钟后回来对答案。"""
from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from trader.open_confirmation import (
    build_open_confirmation_message,
    build_open_range_line,
    build_premarket_drift_lines,
    check_index_triggers,
    decide_playbook,
    should_send_open_confirmation,
)

_ET = ZoneInfo("America/New_York")

# 字段名与 morning_brief._calc_index_technical 的输出一致
SPY = {"ma20": 745.69, "resistance": 748.90, "support": 737.68}


class _Levels:
    def __init__(self, last, lo=744.2, hi=747.3, vwap=745.8, stale=False):
        self.last_price = last
        self.open_range_low = lo
        self.open_range_high = hi
        self.open_range_minutes = 15
        self.vwap = vwap
        self.is_stale = stale


def _et(hour, minute, day=3):
    return datetime(2026, 8, day, hour, minute, tzinfo=_ET).astimezone(timezone.utc)


# ── 触发位对账 ───────────────────────────────────────────────────────────────


def test_checks_use_the_same_levels_the_brief_quoted():
    checks = check_index_triggers("SPY", SPY, 746.12)
    levels = {c.level for c in checks}
    assert levels == {745.69, 748.90, 737.68}


def test_each_condition_gets_a_verdict():
    checks = check_index_triggers("SPY", SPY, 746.12)
    by_label = {c.label: c.hit for c in checks}
    assert by_label["SPY 站回 20MA"] is True       # 746.12 > 745.69
    assert by_label["SPY 上破昨高"] is False       # 746.12 < 748.90
    assert by_label["SPY 跌破昨低"] is False       # 746.12 > 737.68


def test_rendered_check_shows_both_target_and_actual():
    line = check_index_triggers("SPY", SPY, 746.12)[0].render()
    assert "745.69" in line and "746.12" in line
    assert "✅" in line


# ── 剧本判定 ─────────────────────────────────────────────────────────────────


def test_bull_needs_both_ma_and_prior_high():
    assert decide_playbook(check_index_triggers("SPY", SPY, 750.0))[0] == "Bull"


def test_above_ma_alone_is_only_base():
    assert decide_playbook(check_index_triggers("SPY", SPY, 746.12))[0] == "Base"


def test_nothing_triggered_is_base():
    assert decide_playbook(check_index_triggers("SPY", SPY, 740.0))[0] == "Base"


def test_breaking_prior_low_forces_defence():
    assert decide_playbook(check_index_triggers("SPY", SPY, 730.0))[0] == "Bear/防守"


def test_defence_wins_over_a_bullish_reading_elsewhere():
    """一个指数跌破昨低就该转防守，哪怕另一个还站在均线上方。"""
    mixed = check_index_triggers("SPY", SPY, 730.0) + check_index_triggers(
        "QQQ", {"ma20": 700.0, "resistance": 710.0, "support": 690.0}, 715.0
    )
    assert decide_playbook(mixed)[0] == "Bear/防守"


def test_missing_technicals_do_not_fake_a_playbook():
    playbook, _ = decide_playbook([])
    assert playbook == "无法判定"


# ── 开盘区间 ─────────────────────────────────────────────────────────────────


def test_open_range_line_includes_vwap():
    line = build_open_range_line(_Levels(746.0))
    assert "744.20" in line and "747.30" in line and "745.80" in line


def test_stale_data_is_flagged():
    assert "过期" in build_open_range_line(_Levels(746.0, stale=True))


def test_absent_levels_say_so():
    assert "暂无" in build_open_range_line(None)


# ── 盘前 vs 实际 ─────────────────────────────────────────────────────────────


def test_premarket_decay_is_called_out():
    """盘前 +5% 开盘只剩 +3.2%，晨报把它列进重点观察的理由已经变了。"""
    lines = build_premarket_drift_lines({"VLO": 5.0}, {"VLO": 3.2})
    assert "异动减弱" in lines[0]
    assert "+5.00%" in lines[0] and "+3.20%" in lines[0]


def test_premarket_continuation_is_distinguished():
    assert "延续" in build_premarket_drift_lines({"VLO": 5.0}, {"VLO": 5.2})[0]


def test_premarket_strengthening_is_distinguished():
    assert "异动加强" in build_premarket_drift_lines({"VLO": 2.0}, {"VLO": 4.0})[0]


def test_a_deepening_decline_is_not_called_strengthening():
    """-1.70% 跌到 -2.40% 是跌势加深；"走强"会被读成股价上涨，恰好反了。"""
    line = build_premarket_drift_lines({"NTNX": -1.70}, {"NTNX": -2.40})[0]
    assert "异动加强" in line
    assert "走强" not in line


# ── 报告组装 ─────────────────────────────────────────────────────────────────


def test_message_states_the_playbook_and_the_morning_call():
    note = build_open_confirmation_message(
        trading_date="2026-08-03",
        index_levels={"SPY": _Levels(746.12)},
        technicals={"SPY": SPY},
        morning_bias="中性震荡，先等开盘区间给方向",
        now_et=datetime(2026, 8, 3, 9, 45, tzinfo=_ET),
    )
    assert "开盘确认" in note.title
    assert "中性震荡" in note.body
    assert "Base" in note.body
    assert note.fields["剧本"] == "Base"
    assert note.dedupe_key == "open_confirmation:2026-08-03"


def test_message_does_not_read_as_an_entry_order():
    note = build_open_confirmation_message(
        trading_date="2026-08-03",
        index_levels={"SPY": _Levels(750.0)},
        technicals={"SPY": SPY},
    )
    assert "不是新的入场指令" in note.body


# ── 调度 ─────────────────────────────────────────────────────────────────────


def test_fires_once_after_the_open_range_forms():
    assert should_send_open_confirmation(_et(9, 45), None)
    assert not should_send_open_confirmation(_et(9, 30), None)


def test_not_sent_twice_in_a_day():
    assert not should_send_open_confirmation(_et(9, 46), "2026-08-03")


def test_late_start_still_catches_up_within_the_window():
    """引擎晚启动或卡顿都不该让当天这份报告永远消失。

    晨报的 should_send_brief 至今是 hour == N 的精确匹配，有同样的脆弱性。
    """
    assert should_send_open_confirmation(_et(10, 30), None)


def test_too_late_is_no_longer_an_open_confirmation():
    assert not should_send_open_confirmation(_et(14, 0), None)


def test_weekend_is_skipped():
    # 2026-08-08 是周六
    assert not should_send_open_confirmation(_et(9, 45, day=8), None)
