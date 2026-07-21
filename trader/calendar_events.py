"""
trader/calendar_events.py
获取近期高波动市场事件：美联储会议 / CPI / NFP / 重点财报 等。

数据来源：
  - Nasdaq 公开经济日历（无 API Key，作为经济事件主源）
  - Finnhub API（仅用于财报日历；经济日历容易 403，已从默认链路移除）
  - yfinance（单只股票财报日期，作为补充）

使用方式：
    events = get_upcoming_events(symbols=["NVDA","AAPL"], days=7)
    for e in events:
        print(e.date, e.impact, e.title_cn)
"""
from __future__ import annotations

import json
import logging
import os
import re
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from html import unescape
from typing import List, Optional
from zoneinfo import ZoneInfo

from .official_events import OfficialEvent, get_official_events

logger = logging.getLogger(__name__)

_FINNHUB_BASE  = "https://finnhub.io/api/v1"
_HEADERS = {"User-Agent": "DiscordBot (https://github.com/stock-minute-ai, 1.0)"}
_CORE_OFFICIAL_EVENT_SOURCES = {
    "official_fed_fomc",
    "official_bls_calendar",
    "official_bea_calendar",
}

try:
    _PACIFIC_TZ = ZoneInfo("America/Los_Angeles")
except Exception:
    _PACIFIC_TZ = timezone(timedelta(hours=-8), "America/Los_Angeles")


def _now_pacific() -> datetime:
    return datetime.now(_PACIFIC_TZ)

# ── 影响级别显示 ──────────────────────────────────────────────────────────────

IMPACT_EMOJI = {
    "critical": "⚡⚡⚡",
    "high":     "⚡⚡",
    "medium":   "⚡",
}

# ── 经济事件名称 → 中文 + 简单说明 ────────────────────────────────────────────
# 只翻译最重要的，其余保留英文原名

_EVENT_CN: dict[str, tuple[str, str]] = {
    # (中文名, 影响说明)
    "fed interest rate decision":    ("美联储利率决议 🏦",    "直接影响全市场方向，极高波动"),
    "fomc meeting minutes":          ("美联储会议纪要 📋",    "解读降/加息路径，可能引发大幅波动"),
    "fomc press conference":         ("美联储主席记者会 🎤",  "鲍威尔讲话，市场将解读政策信号"),
    "nonfarm payrolls":              ("非农就业数据 💼",      "超预期数据将影响降息时间表"),
    "cpi m/m":                       ("CPI 通胀数据 📊",      "高于预期→推迟降息→股市可能下跌"),
    "core cpi m/m":                  ("核心 CPI 📊",          "剔除食品能源，美联储最关注的指标"),
    "pce price index m/m":           ("PCE 通胀指数 📊",      "美联储首选通胀指标，影响降息预期"),
    "core pce price index m/m":      ("核心 PCE 📊",          "美联储最关注的通胀指标"),
    "gdp q/q":                       ("GDP 季度数据 📈",      "经济强弱风向标"),
    "advance gdp q/q":               ("GDP 初值 📈",          "季度经济增速首次披露，市场敏感"),
    "retail sales m/m":              ("零售销售数据 🛒",      "反映消费景气，影响科技/消费板块"),
    "ism manufacturing pmi":         ("ISM 制造业 PMI 🏭",   "低于50=收缩，影响制造/工业板块"),
    "ism services pmi":              ("ISM 服务业 PMI 🏢",   "美国经济以服务业为主，高度关注"),
    "ppi m/m":                       ("PPI 生产者价格 🏭",   "生产端通胀先行指标"),
    "jolts job openings":            ("JOLTS 职位空缺 💼",   "劳动力市场热度，影响加息预期"),
    "initial jobless claims":        ("初请失业金 📋",        "每周就业健康度，影响美联储判断"),
    "michigan consumer sentiment":   ("密歇根消费者信心 📊", "消费者预期，影响消费板块"),
    "consumer confidence":           ("消费者信心指数 📊",   "消费景气先行指标"),
}

# 财报时段中文
_HOUR_CN = {
    "bmo": "盘前公布",  # before market open
    "amc": "盘后公布",  # after market close
    "dmh": "盘中公布",
}


@dataclass
class CalendarEvent:
    date:       str            # "2026-06-17"
    time_str:   str            # "08:30 ET" / "盘前" / "盘后"
    title_cn:   str            # 中文事件名
    title_en:   str            # 英文原名（备用）
    impact:     str            # "critical" / "high" / "medium"
    category:   str            # "fomc" / "economic" / "earnings"
    symbol:     Optional[str] = None
    note:       str = ""       # 影响说明（一句话）
    source:     str = ""


@dataclass
class EventSourceStatus:
    """单个日历数据源状态，用于区分“真无事件”和“源不可用”。"""

    name: str
    state: str                 # ok | missing_key | error | not_requested
    detail: str = ""


@dataclass
class CalendarEventsResult:
    """带数据源状态的事件日历结果。"""

    events: List[CalendarEvent] = field(default_factory=list)
    sources: List[EventSourceStatus] = field(default_factory=list)

    def source_state(self, name: str) -> str:
        for source in self.sources:
            if source.name == name:
                return source.state
        return "not_requested"

    @property
    def has_source_issue(self) -> bool:
        return not self.economic_available

    @property
    def official_available(self) -> bool:
        return any(
            source.name in _CORE_OFFICIAL_EVENT_SOURCES and source.state == "ok"
            for source in self.sources
        )

    @property
    def official_core_complete(self) -> bool:
        return all(
            self.source_state(source_name) == "ok"
            for source_name in _CORE_OFFICIAL_EVENT_SOURCES
        )

    @property
    def economic_available(self) -> bool:
        return (
            self.source_state("finnhub_economic") == "ok"
            or self.source_state("nasdaq_economic") == "ok"
            or self.official_core_complete
        )

    @property
    def has_partial_source_issue(self) -> bool:
        return any(
            _is_economic_source(s.name) and s.state in ("missing_key", "error")
            for s in self.sources
        ) and self.economic_available

    @property
    def warning_text(self) -> str:
        if self.economic_available:
            return ""
        issues = [
            f"{s.name}: {s.detail or s.state}"
            for s in self.sources
            if _is_economic_source(s.name) and s.state in ("missing_key", "error")
        ]
        return "；".join(issues)


# ── 主入口 ────────────────────────────────────────────────────────────────────

def get_upcoming_events(
    symbols: Optional[List[str]] = None,
    days: int = 7,
) -> List[CalendarEvent]:
    """
    返回未来 days 天内的高/极高影响力市场事件列表，按日期+重要性排序。

    参数：
        symbols — 关注的股票代码列表（用于抓取对应财报日期）
        days    — 向前看几天（默认7天）
    """
    return get_upcoming_events_with_status(symbols=symbols, days=days).events


def get_upcoming_events_with_status(
    symbols: Optional[List[str]] = None,
    days: int = 7,
) -> CalendarEventsResult:
    """
    返回事件列表以及每个数据源的可用性。

    晨报使用这个接口，避免把数据源失败误展示成“没有重要事件”。
    经济事件默认走 Nasdaq；Finnhub 经济日历不再请求，避免无效 key 反复 403。
    """
    now    = _now_pacific()
    to_dt  = now + timedelta(days=days)
    from_s = now.strftime("%Y-%m-%d")
    to_s   = to_dt.strftime("%Y-%m-%d")

    api_key = os.getenv("FINNHUB_API_KEY", "").strip()
    events: List[CalendarEvent] = []
    sources: List[EventSourceStatus] = []

    sources.append(EventSourceStatus(
        "finnhub_economic",
        "not_requested",
        "已由 Nasdaq economic calendar 替代",
    ))

    official_events, official_sources = _fetch_official_events_with_status(from_s, to_s)
    events += official_events
    sources += official_sources

    if api_key:
        if symbols:
            earnings, earnings_error = _fetch_earnings_with_status(
                api_key, from_s, to_s, symbols
            )
            events += earnings
            sources.append(EventSourceStatus(
                name="finnhub_earnings",
                state="error" if earnings_error else "ok",
                detail=earnings_error or "",
            ))
        else:
            sources.append(EventSourceStatus("finnhub_earnings", "not_requested"))
    else:
        logger.debug("FINNHUB_API_KEY 未配置，跳过 Finnhub 财报日历")
        state = "missing_key" if symbols else "not_requested"
        detail = "FINNHUB_API_KEY 未配置" if symbols else ""
        sources.append(EventSourceStatus("finnhub_earnings", state, detail))

    nasdaq_events, nasdaq_error = _fetch_nasdaq_economic_with_status(from_s, to_s)
    events += nasdaq_events
    sources.append(EventSourceStatus(
        name="nasdaq_economic",
        state="error" if nasdaq_error else "ok",
        detail=nasdaq_error or "",
    ))

    # 补充 yfinance 财报（不依赖 Finnhub，作为兜底）
    if symbols:
        try:
            yf_events = _fetch_yfinance_earnings(
                symbols, now, to_dt,
                existing_syms={e.symbol for e in events if e.symbol},
            )
            events += yf_events
            sources.append(EventSourceStatus("yfinance_earnings", "ok"))
        except Exception as exc:
            detail = _format_fetch_error(exc)
            logger.warning("yfinance 财报日历失败: %s", detail)
            sources.append(EventSourceStatus("yfinance_earnings", "error", detail))
    else:
        sources.append(EventSourceStatus("yfinance_earnings", "not_requested"))

    deduped = _dedupe_calendar_events(events)

    # 排序：日期 → 影响级别（critical 最前）
    _order = {"critical": 0, "high": 1, "medium": 2}
    deduped.sort(key=lambda e: (e.date, _order.get(e.impact, 3)))

    return CalendarEventsResult(events=deduped, sources=sources)


def _is_economic_source(name: str) -> bool:
    return name in ("finnhub_economic", "nasdaq_economic") or name.startswith("official_")


def _fetch_official_events_with_status(
    from_s: str, to_s: str
) -> tuple[List[CalendarEvent], List[EventSourceStatus]]:
    try:
        result = get_official_events(from_s, to_s)
    except Exception as exc:
        return [], [EventSourceStatus("official_events", "error", _format_fetch_error(exc))]

    events = [_official_event_to_calendar(event) for event in result.events]
    sources = [
        EventSourceStatus(source.name, source.state, source.detail)
        for source in result.sources
    ]
    return events, sources


def _official_event_to_calendar(event: OfficialEvent) -> CalendarEvent:
    return CalendarEvent(
        date=event.date,
        time_str=event.time_str,
        title_cn=event.title,
        title_en=event.title,
        impact=event.impact,
        category=event.category,
        note=event.note,
        source=event.source,
    )


def _dedupe_calendar_events(events: List[CalendarEvent]) -> List[CalendarEvent]:
    by_key: dict[tuple[str, str, str, str], CalendarEvent] = {}
    for event in events:
        key = _calendar_dedupe_key(event)
        existing = by_key.get(key)
        if existing is None or _event_source_priority(event) < _event_source_priority(existing):
            by_key[key] = event
    return list(by_key.values())


def _calendar_dedupe_key(event: CalendarEvent) -> tuple[str, str, str, str]:
    if event.symbol:
        return (event.date, "earnings", event.symbol.upper(), "")
    family = _event_family(event)
    if family:
        return (event.date, _event_time_bucket(event.time_str), family, "")
    title = re.sub(r"\s+", " ", (event.title_en or event.title_cn).strip().lower())
    return (event.date, _event_time_bucket(event.time_str), title, "")


def _event_family(event: CalendarEvent) -> str:
    text = f"{event.title_en} {event.title_cn}".lower()
    if event.category == "fomc" or any(
        token in text for token in ("fomc", "interest rate", "federal funds")
    ):
        return "fomc"
    if "core cpi" in text:
        return "core_cpi"
    if "consumer price index" in text or re.search(r"\bcpi\b", text):
        return "cpi"
    if "producer price index" in text or re.search(r"\bppi\b", text):
        return "ppi"
    if "nonfarm" in text or "employment situation" in text:
        return "nonfarm_payrolls"
    if "job openings and labor turnover" in text or "jolts" in text:
        return "jolts"
    if "personal income and outlays" in text or "core pce" in text or "pce price" in text:
        return "pce"
    if "gross domestic product" in text or re.search(r"\bgdp\b", text):
        return "gdp"
    if "crude oil inventories" in text or "petroleum status" in text:
        return "eia_crude"
    if "retail sales" in text:
        return "retail_sales"
    return ""


def _event_time_bucket(time_str: str) -> str:
    match = re.search(r"(\d{1,2}):(\d{2})", str(time_str or ""))
    if match:
        return f"{int(match.group(1)):02d}:{match.group(2)}"
    return str(time_str or "").strip().lower()


def _event_source_priority(event: CalendarEvent) -> int:
    source = (event.source or "").lower()
    if source.startswith("official_"):
        return 0
    if source == "nasdaq_economic":
        return 1
    if source == "finnhub_economic":
        return 2
    if source == "finnhub_earnings":
        return 3
    if source == "yfinance_earnings":
        return 4
    return 9


def _format_fetch_error(exc: Exception) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        return f"HTTP {exc.code}"
    if isinstance(exc, urllib.error.URLError):
        return str(exc.reason)
    return exc.__class__.__name__


def _clean_cell(value) -> str:
    text = unescape(str(value or "")).replace("\xa0", " ")
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()


def _fetch_economic(api_key: str, from_s: str, to_s: str) -> List[CalendarEvent]:
    events, _ = _fetch_economic_with_status(api_key, from_s, to_s)
    return events


# ── Nasdaq 经济日历（免费兜底，无需 API Key）──────────────────────────────────

def _fetch_nasdaq_economic_with_status(
    from_s: str, to_s: str
) -> tuple[List[CalendarEvent], Optional[str]]:
    try:
        start = datetime.strptime(from_s, "%Y-%m-%d").date()
        end = datetime.strptime(to_s, "%Y-%m-%d").date()
    except Exception as exc:
        return [], _format_fetch_error(exc)

    events: List[CalendarEvent] = []
    cur = start
    while cur <= end:
        url = f"https://api.nasdaq.com/api/calendar/economicevents?date={cur:%Y-%m-%d}"
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (compatible; stock-minute-ai/1.0)",
                "Accept": "application/json,text/plain,*/*",
                "Origin": "https://www.nasdaq.com",
                "Referer": "https://www.nasdaq.com/market-activity/economic-calendar",
            })
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read())
            rows = (data.get("data") or {}).get("rows") or []
        except Exception as exc:
            detail = _format_fetch_error(exc)
            logger.warning("Nasdaq 经济日历失败: %s", detail)
            return events, detail

        for item in rows:
            event = _nasdaq_row_to_event(str(cur), item)
            if event is not None:
                events.append(event)
        cur += timedelta(days=1)

    return events, None


def _nasdaq_row_to_event(date_str: str, item: dict) -> Optional[CalendarEvent]:
    country = _clean_cell(item.get("country"))
    if country != "United States":
        return None

    name = _clean_cell(item.get("eventName"))
    if not name:
        return None

    impact, title_cn, note, category = _classify_nasdaq_event(name)
    if impact is None:
        return None

    consensus = _clean_cell(item.get("consensus"))
    previous = _clean_cell(item.get("previous"))
    values = []
    if consensus:
        values.append(f"预期 {consensus}")
    if previous:
        values.append(f"前值 {previous}")
    if values:
        note = f"{note}；{' / '.join(values)}" if note else " / ".join(values)

    raw_time = _clean_cell(item.get("gmt")) or "时间待定"
    time_str = raw_time if raw_time in ("All Day", "时间待定") else f"{raw_time} ET"

    return CalendarEvent(
        date=date_str,
        time_str=time_str,
        title_cn=title_cn,
        title_en=name,
        impact=impact,
        category=category,
        note=note,
        source="nasdaq_economic",
    )


def _classify_nasdaq_event(name: str) -> tuple[Optional[str], str, str, str]:
    key = name.lower()

    critical = [
        ("interest rate", "美联储利率/点阵图相关 🏦", "直接影响全市场方向，极高波动", "fomc"),
        ("fomc", "FOMC 事件 🏦", "美联储政策信号，可能引发大幅波动", "fomc"),
        ("federal funds", "联邦基金利率 🏦", "直接影响利率预期和估值", "fomc"),
        ("powell", "鲍威尔讲话 🎤", "市场将解读政策信号", "fomc"),
    ]
    high = [
        ("core cpi", "核心 CPI 📊", "美联储最关注的通胀指标之一", "economic"),
        ("cpi", "CPI 通胀数据 📊", "通胀超预期会压制降息预期", "economic"),
        ("core pce", "核心 PCE 📊", "美联储首选通胀指标", "economic"),
        ("pce price", "PCE 通胀指数 📊", "影响降息预期", "economic"),
        ("nonfarm", "非农就业数据 💼", "就业超预期会影响降息路径", "economic"),
        ("unemployment rate", "失业率 💼", "劳动力市场核心指标", "economic"),
        ("initial jobless claims", "初请失业金 📋", "就业健康度高频指标", "economic"),
        ("continuing jobless claims", "续请失业金 📋", "就业压力高频指标", "economic"),
        ("retail sales", "零售销售 🛒", "消费景气指标，影响风险偏好", "economic"),
        ("gdp", "GDP 数据 📈", "经济强弱风向标", "economic"),
        ("ppi", "PPI 生产者价格 🏭", "生产端通胀先行指标", "economic"),
        ("ism manufacturing", "ISM 制造业 PMI 🏭", "制造业景气核心指标", "economic"),
        ("ism services", "ISM 服务业 PMI 🏢", "服务业景气核心指标", "economic"),
        ("jolts", "JOLTS 职位空缺 💼", "劳动力市场热度", "economic"),
        ("consumer confidence", "消费者信心指数 📊", "消费预期指标", "economic"),
        ("michigan consumer", "密歇根消费者信心 📊", "消费者预期和通胀预期", "economic"),
        ("durable goods", "耐用品订单 🏭", "企业需求和制造景气指标", "economic"),
    ]
    medium = [
        ("crude oil inventories", "EIA 原油库存 🛢", "影响能源、通胀和风险偏好", "economic"),
        ("gasoline inventories", "EIA 汽油库存 🛢", "影响能源板块", "economic"),
        ("natural gas storage", "天然气库存 🛢", "影响能源板块", "economic"),
        ("philadelphia fed", "费城联储制造业指数 🏭", "区域制造业景气指标", "economic"),
        ("philly fed", "费城联储制造业指数 🏭", "区域制造业景气指标", "economic"),
        ("fed's balance sheet", "美联储资产负债表", "美元流动性参考，通常不是开盘方向主因", "economic"),
        ("reserve balances", "联储银行准备金余额", "美元流动性参考，通常不是开盘方向主因", "economic"),
        ("fed", name, "美联储官员讲话，留意利率预期变化", "fomc"),
        ("pending home sales", "成屋签约销售 🏠", "地产需求指标", "economic"),
        ("new home sales", "新屋销售 🏠", "地产需求指标", "economic"),
        ("existing home sales", "成屋销售 🏠", "地产需求指标", "economic"),
        ("leading index", "领先指标 📈", "经济动能参考", "economic"),
        ("gdpnow", "Atlanta Fed GDPNow 📈", "实时 GDP 预估参考", "economic"),
        ("trump speaks", "美国总统讲话 🎤", "可能影响政策预期和板块情绪", "economic"),
        ("juneteenth", "美国假期 Juneteenth", "注意美股交易日/流动性安排", "holiday"),
    ]

    for token, title, note, category in critical:
        if token in key:
            return "critical", title, note, category
    for token, title, note, category in high:
        if token in key:
            return "high", title, note, category
    for token, title, note, category in medium:
        if token in key:
            return "medium", title, note, category

    return None, name, "", "economic"


# ── Finnhub 经济日历 ──────────────────────────────────────────────────────────

def _fetch_economic_with_status(
    api_key: str, from_s: str, to_s: str
) -> tuple[List[CalendarEvent], Optional[str]]:
    url = f"{_FINNHUB_BASE}/calendar/economic?token={api_key}"
    try:
        req = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=8) as r:
            raw = json.loads(r.read()).get("economicCalendar", [])
    except Exception as exc:
        detail = _format_fetch_error(exc)
        logger.warning("Finnhub 经济日历失败: %s", detail)
        return [], detail

    events: List[CalendarEvent] = []
    for item in raw:
        # 只要美国 + 高/中影响
        if item.get("country", "").upper() != "US":
            continue
        impact_raw = item.get("impact", "low").lower()
        if impact_raw not in ("high", "medium"):
            continue

        event_name = item.get("event", "")
        event_dt_str = item.get("time", "")  # "2026-06-17 14:30:00"

        # 解析日期和时间
        try:
            event_dt = datetime.strptime(event_dt_str, "%Y-%m-%d %H:%M:%S")
            date_str = event_dt.strftime("%Y-%m-%d")
            time_str = event_dt.strftime("%H:%M ET")
        except Exception:
            date_str = event_dt_str[:10] if event_dt_str else from_s
            time_str = "待定"

        # 日期范围过滤
        if date_str < from_s or date_str > to_s:
            continue

        # 翻译事件名
        key = event_name.lower().strip()
        title_cn, note = _EVENT_CN.get(key, (event_name, ""))

        # 修正 impact：Fed 相关提升为 critical
        if any(w in key for w in ("fed interest rate", "fomc")):
            impact = "critical"
        elif impact_raw == "high":
            impact = "high"
        else:
            impact = "medium"

        # 确定 category
        if any(w in key for w in ("fed", "fomc")):
            category = "fomc"
        else:
            category = "economic"

        events.append(CalendarEvent(
            date=date_str, time_str=time_str,
            title_cn=title_cn, title_en=event_name,
            impact=impact, category=category, note=note,
            source="finnhub_economic",
        ))

    return events, None


# ── Finnhub 财报日历 ──────────────────────────────────────────────────────────

def _fetch_earnings(
    api_key: str, from_s: str, to_s: str, symbols: List[str]
) -> List[CalendarEvent]:
    events, _ = _fetch_earnings_with_status(api_key, from_s, to_s, symbols)
    return events


def _fetch_earnings_with_status(
    api_key: str, from_s: str, to_s: str, symbols: List[str]
) -> tuple[List[CalendarEvent], Optional[str]]:
    url = f"{_FINNHUB_BASE}/calendar/earnings?from={from_s}&to={to_s}&token={api_key}"
    try:
        req = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=8) as r:
            raw = json.loads(r.read()).get("earningsCalendar", [])
    except Exception as exc:
        detail = _format_fetch_error(exc)
        logger.warning("Finnhub 财报日历失败: %s", detail)
        return [], detail

    sym_set = {s.upper() for s in symbols}
    events: List[CalendarEvent] = []
    for item in raw:
        sym = (item.get("symbol") or "").upper()
        if sym not in sym_set:
            continue
        date_str = item.get("date", "")
        if date_str < from_s or date_str > to_s:
            continue
        hour = item.get("hour", "").lower()
        time_str = _HOUR_CN.get(hour, "时间待定")
        eps_est  = item.get("epsEstimate")
        note = f"EPS 预期 ${eps_est:.2f}" if eps_est is not None else ""
        events.append(CalendarEvent(
            date=date_str, time_str=time_str,
            title_cn=f"{sym} 财报发布",
            title_en=f"{sym} Earnings",
            impact="high", category="earnings",
            symbol=sym, note=note,
            source="finnhub_earnings",
        ))

    return events, None


# ── yfinance 财报兜底 ─────────────────────────────────────────────────────────

def _fetch_yfinance_earnings(
    symbols: List[str],
    from_dt: datetime,
    to_dt: datetime,
    existing_syms: set,
) -> List[CalendarEvent]:
    """用 yfinance 补充 Finnhub 未覆盖的财报日期。"""
    events: List[CalendarEvent] = []
    try:
        import yfinance as yf
    except ImportError:
        return events

    for sym in symbols:
        if sym.upper() in existing_syms:
            continue
        try:
            cal = yf.Ticker(sym).calendar
            if cal is None:
                continue

            # yfinance ≥0.2.40 返回 dict；旧版返回 DataFrame
            if isinstance(cal, dict):
                dates_raw = cal.get("Earnings Date", [])
                if not isinstance(dates_raw, list):
                    dates_raw = [dates_raw]
            elif hasattr(cal, "columns"):
                # DataFrame: 列名是财报日期 Timestamp
                dates_raw = list(cal.columns)
            else:
                continue

            for raw_dt in dates_raw:
                try:
                    earn_date = raw_dt.date() if hasattr(raw_dt, "date") else None
                    if earn_date is None:
                        continue
                    date_str = str(earn_date)[:10]
                    if from_dt.strftime("%Y-%m-%d") <= date_str <= to_dt.strftime("%Y-%m-%d"):
                        events.append(CalendarEvent(
                            date=date_str, time_str="时间待定",
                            title_cn=f"{sym.upper()} 财报发布",
                            title_en=f"{sym.upper()} Earnings",
                            impact="high", category="earnings",
                            symbol=sym.upper(),
                            source="yfinance_earnings",
                        ))
                        break  # 只取最近一次
                except Exception:
                    pass
        except Exception:
            pass

    return events
