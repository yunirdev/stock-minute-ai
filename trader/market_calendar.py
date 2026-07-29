"""
US equity market session helper.

The runtime only needs a coarse full-day session gate:
pre / open / post / closed.  This module uses America/New_York so daylight
saving time is handled correctly, and includes the standard full-day
NYSE/Nasdaq holidays without requiring a network call.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Literal
from zoneinfo import ZoneInfo


SessionName = Literal["pre", "open", "post", "closed"]

_NY = ZoneInfo("America/New_York")


class SimpleMarketCalendar:
    """Implements the MarketCalendar protocol."""

    def session_now(self) -> SessionName:
        return session_at(datetime.now(timezone.utc))


def session_at(dt: datetime) -> SessionName:
    """Return the US equity session for a UTC or timezone-aware datetime."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    et = dt.astimezone(_NY)
    if et.weekday() >= 5 or et.date() in market_holidays(et.year):
        return "closed"

    minute = et.hour * 60 + et.minute
    if 4 * 60 <= minute < 9 * 60 + 30:
        return "pre"
    if 9 * 60 + 30 <= minute < 16 * 60:
        return "open"
    if 16 * 60 <= minute < 20 * 60:
        return "post"
    return "closed"


def market_holidays(year: int) -> set[date]:
    """Standard full-day NYSE/Nasdaq market holidays for one calendar year."""
    return {
        _observed(date(year, 1, 1)),
        _nth_weekday(year, 1, 0, 3),  # Martin Luther King Jr. Day
        _nth_weekday(year, 2, 0, 3),  # Presidents' Day
        _easter_date(year) - timedelta(days=2),  # Good Friday
        _last_weekday(year, 5, 0),  # Memorial Day
        _observed(date(year, 6, 19)),  # Juneteenth
        _observed(date(year, 7, 4)),  # Independence Day
        _nth_weekday(year, 9, 0, 1),  # Labor Day
        _nth_weekday(year, 11, 3, 4),  # Thanksgiving
        _observed(date(year, 12, 25)),  # Christmas
    }


def _observed(day: date) -> date:
    if day.weekday() == 5:
        return day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


def _nth_weekday(year: int, month: int, weekday: int, nth: int) -> date:
    cur = date(year, month, 1)
    while cur.weekday() != weekday:
        cur += timedelta(days=1)
    return cur + timedelta(days=7 * (nth - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    if month == 12:
        cur = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        cur = date(year, month + 1, 1) - timedelta(days=1)
    while cur.weekday() != weekday:
        cur -= timedelta(days=1)
    return cur


def _easter_date(year: int) -> date:
    """Gregorian Easter using the Meeus/Jones/Butcher algorithm."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month = (h + ell - 7 * m + 114) // 31
    day = ((h + ell - 7 * m + 114) % 31) + 1
    return date(year, month, day)


_calendar = SimpleMarketCalendar()


def session_now() -> SessionName:
    return _calendar.session_now()
