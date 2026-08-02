"""报告周期解析 —— 手动补发时"这份报告说的是哪一段时间"。

自动推送的时点是确定的，周期不言自明；手动补发不是。同一个"发送复盘"按钮
在周日下午按下和在周二盘中按下，读者期待的显然不是同一份东西，而在此之前所
有手动按钮都只是拿"此刻"去算，算出什么发什么，报告里也不说自己覆盖的是哪一
段。于是周日点一次"每日复盘"，得到的是一份标着当天日期、实际上没有任何成交
的空报告——看起来像"那天什么都没做"，其实是"那天根本不开市"。

统一规则：

- **当前周期已经开始** → 报告这个周期从开始到此刻的内容，并标注尚未结束。
- **当前周期还没开始** → 报告上一个完整周期。

partial 与否必须写进报告，因为"今天到现在只成交 1 笔"和"今天全天只成交 1
笔"对读者是两件事。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from .market_calendar import market_holidays, session_at

_ET = ZoneInfo("America/New_York")

#: 美股盘前从 4:00 ET 开始，这也是"新的交易日已经开始"的分界。
_TRADING_DAY_START_HOUR = 4
#: 20:00 ET 盘后结束，此后当天就是一个完整周期了。
_TRADING_DAY_END_HOUR = 20


@dataclass(frozen=True)
class ReportPeriod:
    """一份报告覆盖的时间范围。"""

    label: str              # "2026-08-03" / "2026-W31"
    start: datetime         # UTC
    end: datetime           # UTC，partial 时就是"此刻"
    is_partial: bool
    kind: str = "daily"     # daily | weekly

    @property
    def note(self) -> str:
        """给读者的一句话说明，必须让人一眼看出数据截止到哪。"""
        if not self.is_partial:
            return ""
        local_end = self.end.astimezone(_ET)
        if self.kind == "weekly":
            return (
                f"⚠️ 本周尚未结束，以下统计只覆盖到 {local_end:%m/%d %H:%M} ET，"
                "不是完整一周的结果。"
            )
        return (
            f"⚠️ 本交易日尚未结束，以下内容只覆盖到 {local_end:%H:%M} ET，"
            "不是全天结果。"
        )


def is_trading_day(day: date) -> bool:
    return day.weekday() < 5 and day not in market_holidays(day.year)


def previous_trading_day(day: date) -> date:
    cursor = day - timedelta(days=1)
    # 最多回溯两周足以跨过任何连续假期
    for _ in range(14):
        if is_trading_day(cursor):
            return cursor
        cursor -= timedelta(days=1)
    return cursor


def _et_bounds(day: date) -> tuple[datetime, datetime]:
    start = datetime.combine(day, time(_TRADING_DAY_START_HOUR), tzinfo=_ET)
    end = datetime.combine(day, time(_TRADING_DAY_END_HOUR), tzinfo=_ET)
    return start.astimezone(timezone.utc), end.astimezone(timezone.utc)


def resolve_daily_period(now_utc: datetime) -> ReportPeriod:
    """解析"这份日报说的是哪个交易日"。

    今天开市且已过盘前开始时刻 → 今天（未到 20:00 则是 partial）；否则回退到
    上一个交易日，那是一个完整周期。
    """
    et = now_utc.astimezone(_ET)
    today = et.date()

    if is_trading_day(today) and et.hour >= _TRADING_DAY_START_HOUR:
        start, end = _et_bounds(today)
        finished = et.hour >= _TRADING_DAY_END_HOUR
        return ReportPeriod(
            label=today.isoformat(),
            start=start,
            end=end if finished else now_utc,
            is_partial=not finished,
            kind="daily",
        )

    # 走到这里只有两种情况：今天不开市（周末/假日），或今天开市但还没到盘前
    # 开始时刻。两者都该回退到"从今天往前数的第一个交易日"。
    previous = previous_trading_day(today)
    start, end = _et_bounds(previous)
    return ReportPeriod(
        label=previous.isoformat(),
        start=start,
        end=end,
        is_partial=False,
        kind="daily",
    )


def resolve_weekly_period(now_utc: datetime) -> ReportPeriod:
    """解析"这份周报说的是哪一周"。

    周五 20:00 ET 之后本周就完整了；在那之前手动补发，报告的是本周至今。
    """
    et = now_utc.astimezone(_ET)
    year, week, weekday = et.isocalendar()

    week_start_date = et.date() - timedelta(days=weekday - 1)
    start = datetime.combine(
        week_start_date, time(_TRADING_DAY_START_HOUR), tzinfo=_ET
    ).astimezone(timezone.utc)
    friday = week_start_date + timedelta(days=4)
    week_end = datetime.combine(
        friday, time(_TRADING_DAY_END_HOUR), tzinfo=_ET
    ).astimezone(timezone.utc)

    finished = now_utc >= week_end
    return ReportPeriod(
        label=f"{year}-W{week:02d}",
        start=start,
        end=week_end if finished else now_utc,
        is_partial=not finished,
        kind="weekly",
    )


def period_hours(period: ReportPeriod) -> int:
    """周期长度（小时），供按小时窗口取数的旧接口使用。"""
    delta = period.end - period.start
    return max(1, int(delta.total_seconds() // 3600))
