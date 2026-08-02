"""手动补发时，报告必须说清自己覆盖的是哪一段时间。

在这之前所有手动按钮都只拿"此刻"去算：周日点一次"每日复盘"，得到的是一份
标着当天日期、没有任何成交的空报告，读起来像"那天什么都没做"，实际是"那天
根本不开市"。
"""
from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from trader.report_period import (
    is_trading_day,
    period_hours,
    previous_trading_day,
    resolve_daily_period,
    resolve_weekly_period,
)

_ET = ZoneInfo("America/New_York")


def _et(year, month, day, hour, minute=0):
    return datetime(year, month, day, hour, minute, tzinfo=_ET).astimezone(timezone.utc)


# 2026-08-03 周一 … 08-07 周五，08-08 周六，08-09 周日


# ── 日周期 ───────────────────────────────────────────────────────────────────


def test_intraday_reports_today_as_partial():
    """盘中补发 = 今天到此刻为止，且必须标明尚未结束。"""
    period = resolve_daily_period(_et(2026, 8, 4, 11, 30))
    assert period.label == "2026-08-04"
    assert period.is_partial
    assert "尚未结束" in period.note
    assert "11:30" in period.note


def test_after_the_close_today_is_complete():
    period = resolve_daily_period(_et(2026, 8, 4, 20, 30))
    assert period.label == "2026-08-04"
    assert not period.is_partial
    assert period.note == ""


def test_premarket_counts_as_the_new_day_started():
    """4:00 ET 盘前开始就算进入新交易日。"""
    period = resolve_daily_period(_et(2026, 8, 4, 5, 0))
    assert period.label == "2026-08-04"
    assert period.is_partial


def test_before_premarket_still_belongs_to_the_previous_day():
    """凌晨 2 点补发复盘，要的显然是昨天那一整天。"""
    period = resolve_daily_period(_et(2026, 8, 4, 2, 0))
    assert period.label == "2026-08-03"
    assert not period.is_partial


def test_weekend_falls_back_to_friday():
    """周日点"每日复盘"，该给周五的完整交易日，而不是一份空报告。"""
    period = resolve_daily_period(_et(2026, 8, 9, 14, 0))
    assert period.label == "2026-08-07"
    assert not period.is_partial


def test_saturday_falls_back_to_friday():
    assert resolve_daily_period(_et(2026, 8, 8, 10, 0)).label == "2026-08-07"


def test_holiday_is_skipped():
    """2026-07-04 是周六，独立日观察日顺延到 07-03 周五，所以 07-06 周一
    往前找应落到 07-02 周四。"""
    assert not is_trading_day(datetime(2026, 7, 3).date())
    assert previous_trading_day(datetime(2026, 7, 6).date()) == datetime(2026, 7, 2).date()


def test_period_covers_the_trading_day_not_the_calendar_day():
    period = resolve_daily_period(_et(2026, 8, 4, 20, 30))
    start_et = period.start.astimezone(_ET)
    end_et = period.end.astimezone(_ET)
    assert start_et.hour == 4      # 盘前开始
    assert end_et.hour == 20       # 盘后结束


# ── 周周期 ───────────────────────────────────────────────────────────────────


def test_midweek_reports_the_week_so_far():
    period = resolve_weekly_period(_et(2026, 8, 5, 12, 0))
    assert period.is_partial
    assert "本周尚未结束" in period.note
    assert period.label.endswith("W32")


def test_after_friday_close_the_week_is_complete():
    period = resolve_weekly_period(_et(2026, 8, 7, 21, 0))
    assert not period.is_partial
    assert period.note == ""


def test_weekend_still_reports_the_finished_week():
    period = resolve_weekly_period(_et(2026, 8, 9, 12, 0))
    assert not period.is_partial


def test_week_starts_on_monday():
    period = resolve_weekly_period(_et(2026, 8, 5, 12, 0))
    assert period.start.astimezone(_ET).date() == datetime(2026, 8, 3).date()


# ── 辅助 ─────────────────────────────────────────────────────────────────────


def test_period_hours_reflects_a_partial_window():
    """盘中补发的周报不能按 168 小时取数，否则会混进上周的成交。"""
    partial = resolve_weekly_period(_et(2026, 8, 4, 12, 0))
    full = resolve_weekly_period(_et(2026, 8, 7, 21, 0))
    assert period_hours(partial) < period_hours(full)


def test_period_hours_never_zero():
    period = resolve_daily_period(_et(2026, 8, 4, 4, 0))
    assert period_hours(period) >= 1
