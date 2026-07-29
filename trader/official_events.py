from __future__ import annotations

import logging
import re
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime
from html import unescape
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; stock-minute-ai/1.0)",
    "Accept": "text/html,application/xhtml+xml",
}


@dataclass
class OfficialEvent:
    date: str
    time_str: str
    title: str
    impact: str
    category: str
    source: str
    note: str = ""


@dataclass
class OfficialSourceStatus:
    name: str
    state: str
    detail: str = ""


@dataclass
class OfficialEventsResult:
    events: list[OfficialEvent]
    sources: list[OfficialSourceStatus]


def get_official_events(
    from_s: str,
    to_s: str,
    fetcher: Optional[Callable[[str], str]] = None,
) -> OfficialEventsResult:
    fetcher = fetcher or _fetch_text
    events: list[OfficialEvent] = []
    sources: list[OfficialSourceStatus] = []

    source_jobs = [
        ("official_fed_fomc", _fetch_fed_fomc_events),
        ("official_bls_calendar", _fetch_bls_events),
        ("official_bea_calendar", _fetch_bea_events),
        ("official_eia_weekly", _fetch_eia_events),
    ]
    for source_name, fn in source_jobs:
        try:
            batch = fn(from_s, to_s, fetcher)
            events.extend(batch)
            sources.append(OfficialSourceStatus(source_name, "ok"))
        except Exception as exc:
            detail = exc.__class__.__name__
            logger.debug("%s 获取失败: %s", source_name, detail)
            sources.append(OfficialSourceStatus(source_name, "error", detail))

    return OfficialEventsResult(events=events, sources=sources)


def _fetch_text(url: str) -> str:
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=8) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _plain_lines(html: str) -> list[str]:
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", "\n", text)
    return [unescape(line).strip() for line in text.splitlines() if unescape(line).strip()]


def _date_range(from_s: str, to_s: str) -> tuple[date, date]:
    return datetime.strptime(from_s, "%Y-%m-%d").date(), datetime.strptime(to_s, "%Y-%m-%d").date()


def _months_between(start: date, end: date) -> list[tuple[int, int]]:
    cur = date(start.year, start.month, 1)
    out = []
    while cur <= end:
        out.append((cur.year, cur.month))
        if cur.month == 12:
            cur = date(cur.year + 1, 1, 1)
        else:
            cur = date(cur.year, cur.month + 1, 1)
    return out


def _fetch_fed_fomc_events(
    from_s: str,
    to_s: str,
    fetcher: Callable[[str], str],
) -> list[OfficialEvent]:
    start, end = _date_range(from_s, to_s)
    html = fetcher("https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm")
    lines = _plain_lines(html)
    month_lookup = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12,
    }

    events: list[OfficialEvent] = []
    current_year: Optional[int] = None
    current_month: Optional[int] = None
    for line in lines:
        year_match = re.match(r"^(\d{4}) FOMC Meetings$", line)
        if year_match:
            current_year = int(year_match.group(1))
            current_month = None
            continue
        if line in month_lookup:
            current_month = month_lookup[line]
            continue
        if current_year is None or current_month is None:
            continue
        date_match = re.match(r"^(\d{1,2})(?:-(\d{1,2}))?(\*)?$", line)
        if not date_match:
            continue
        end_day = int(date_match.group(2) or date_match.group(1))
        event_date = date(current_year, current_month, end_day)
        if start <= event_date <= end:
            projection = bool(date_match.group(3))
            note = "Fed 官方 FOMC 日历"
            if projection:
                note += "；含 Summary of Economic Projections"
            events.append(OfficialEvent(
                date=event_date.isoformat(),
                time_str="14:00 ET",
                title="FOMC 官方日历",
                impact="critical",
                category="fomc",
                source="official_fed_fomc",
                note=note,
            ))
    return events


def _fetch_bls_events(
    from_s: str,
    to_s: str,
    fetcher: Callable[[str], str],
) -> list[OfficialEvent]:
    start, end = _date_range(from_s, to_s)
    events: list[OfficialEvent] = []
    important = {
        "Consumer Price Index": ("CPI 官方发布", "critical", "通胀核心事件"),
        "Producer Price Index": ("PPI 官方发布", "high", "生产端通胀"),
        "Employment Situation": ("非农/就业官方发布", "critical", "就业核心事件"),
        "Job Openings and Labor Turnover Survey": ("JOLTS 官方发布", "high", "劳动力需求"),
        "U.S. Import and Export Price Indexes": ("进出口价格官方发布", "medium", "贸易价格/通胀参考"),
    }
    for year, month in _months_between(start, end):
        url = f"https://www.bls.gov/schedule/{year}/{month:02d}_sched.htm"
        lines = _plain_lines(fetcher(url))
        current_day: Optional[int] = None
        pending: list[str] = []
        for line in lines:
            if re.fullmatch(r"\d{1,2}", line):
                current_day = int(line)
                pending = []
                continue
            if current_day is None:
                continue
            if re.fullmatch(r"\d{1,2}:\d{2}\s+[AP]M", line):
                if pending:
                    title = pending[0]
                    if title in important:
                        event_date = date(year, month, current_day)
                        if start <= event_date <= end:
                            display, impact, note = important[title]
                            events.append(OfficialEvent(
                                date=event_date.isoformat(),
                                time_str=_am_pm_to_et(line),
                                title=display,
                                impact=impact,
                                category="economic",
                                source="official_bls_calendar",
                                note=f"BLS 官方日历；{note}",
                            ))
                pending = []
            else:
                pending.append(line)
    return events


def _fetch_bea_events(
    from_s: str,
    to_s: str,
    fetcher: Callable[[str], str],
) -> list[OfficialEvent]:
    start, end = _date_range(from_s, to_s)
    lines = _plain_lines(fetcher("https://www.bea.gov/news/schedule"))
    events: list[OfficialEvent] = []
    current_date: Optional[date] = None
    current_time = "08:30 ET"
    month_lookup = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12,
    }
    for line in lines:
        date_match = re.match(r"^([A-Z][a-z]+)\s+(\d{1,2})$", line)
        if date_match and date_match.group(1) in month_lookup:
            month = month_lookup[date_match.group(1)]
            year = start.year if month >= start.month else start.year + 1
            current_date = date(year, month, int(date_match.group(2)))
            continue
        if re.fullmatch(r"\d{1,2}:\d{2}\s+[AP]M", line):
            current_time = _am_pm_to_et(line)
            continue
        if current_date is None or not (start <= current_date <= end):
            continue
        lowered = line.lower()
        if "gdp" in lowered or "personal income and outlays" in lowered:
            impact = "high" if "gdp" in lowered else "critical"
            events.append(OfficialEvent(
                date=current_date.isoformat(),
                time_str=current_time,
                title=line,
                impact=impact,
                category="economic",
                source="official_bea_calendar",
                note="BEA 官方发布日程",
            ))
            current_date = None
    return events


def _fetch_eia_events(
    from_s: str,
    to_s: str,
    fetcher: Callable[[str], str],
) -> list[OfficialEvent]:
    start, end = _date_range(from_s, to_s)
    html = fetcher("https://www.eia.gov/petroleum/supply/weekly/")
    text = " ".join(_plain_lines(html))
    match = re.search(r"Next Release Date:\s+([A-Z][a-z]+)\s+(\d{1,2}),\s+(\d{4})", text)
    if not match:
        return []
    month_lookup = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12,
    }
    event_date = date(
        int(match.group(3)),
        month_lookup[match.group(1)],
        int(match.group(2)),
    )
    if not (start <= event_date <= end):
        return []
    return [OfficialEvent(
        date=event_date.isoformat(),
        time_str="10:30 ET",
        title="EIA 原油库存官方发布",
        impact="medium",
        category="economic",
        source="official_eia_weekly",
        note="EIA Weekly Petroleum Status Report 官方日程",
    )]


def _am_pm_to_et(value: str) -> str:
    parsed = datetime.strptime(value, "%I:%M %p")
    return parsed.strftime("%H:%M ET")
