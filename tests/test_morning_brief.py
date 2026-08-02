from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _disable_official_event_network(monkeypatch):
    import trader.calendar_events as ce

    monkeypatch.setattr(ce, "_fetch_official_events_with_status", lambda *_: ([], []))


def _calendar_result(events=None, issue: bool = False):
    from trader.calendar_events import CalendarEventsResult, EventSourceStatus

    sources = [
        EventSourceStatus(
            "finnhub_economic",
            "error" if issue else "ok",
            "HTTP 403" if issue else "",
        ),
        EventSourceStatus("finnhub_earnings", "ok"),
        EventSourceStatus("yfinance_earnings", "ok"),
    ]
    return CalendarEventsResult(events=events or [], sources=sources)


def test_morning_brief_keeps_four_messages_with_action_card(monkeypatch):
    import trader.morning_brief as mb

    mkt = {
        "^VIX": {"price": 18.0, "pct": 1.0},
        "ES=F": {"price": 6000.0, "pct": 0.6},
        "NQ=F": {"price": 22000.0, "pct": 0.5},
        "HYG": {"price": 78.0, "pct": 0.1},
        "^TNX": {"price": 4.2, "pct": 0.5},
        "XLK": {"price": 100.0, "pct": 1.1},
        "XLF": {"price": 100.0, "pct": 0.7},
    }
    technicals = {
        "SPY": {
            "close": 600.0,
            "chg": 0.3,
            "ma20": 590.0,
            "ma50": 580.0,
            "ma200": 530.0,
            "vs20": 1.7,
            "vs50": 3.4,
            "vs200": 13.2,
            "rsi14": 58.0,
            "atr14_pct": 0.9,
            "support": 595.0,
            "resistance": 603.0,
            "trend": "多头排列",
        }
    }
    monkeypatch.setattr(mb, "_fetch_batch", lambda: mkt)
    monkeypatch.setattr(mb, "_get_calendar_events_result", lambda symbols, days=5: _calendar_result())
    monkeypatch.setattr(mb, "_fetch_index_technicals", lambda: technicals)
    monkeypatch.setattr(mb, "_fetch_priority_market_news", lambda symbols=None: [])
    monkeypatch.setattr(
        mb,
        "_fetch_premarket_movers",
        lambda symbols: [{"symbol": "NVDA", "pct": 1.2, "pre_price": 101.2, "prev": 100.0}],
    )
    monkeypatch.setattr(mb, "_build_regime_section", lambda live_vix=None: ("**市场环境**\n中性", 0))
    monkeypatch.setattr(mb, "_build_fear_greed", lambda: "")
    monkeypatch.setattr(mb, "_build_market_overview", lambda mkt: "大盘")
    monkeypatch.setattr(mb, "_build_macro_section", lambda mkt: "")
    monkeypatch.setattr(mb, "_build_sector_section", lambda mkt: "板块")
    monkeypatch.setattr(mb, "_build_technical_indicator_section", lambda technicals: "技术")
    monkeypatch.setattr(mb, "_build_event_section", lambda symbols, result=None: "事件")
    monkeypatch.setattr(mb, "_build_opex_section", lambda: "OpEx")
    monkeypatch.setattr(mb, "_build_trending_section", lambda symbols=None: "")
    monkeypatch.setattr(mb, "_build_analyst_changes", lambda symbols: "")
    monkeypatch.setattr(mb, "_build_news_section", lambda symbols=None, events=None: "")
    monkeypatch.setattr(mb, "_build_stock_catalysts", lambda symbols: "")
    monkeypatch.setattr(mb, "_build_premarket_section", lambda symbols, movers=None: "盘前")
    monkeypatch.setattr(mb, "_build_status_section", lambda db_path, movers=None: "账户")

    msgs = mb.build_morning_brief(["NVDA", "AAPL"], db_path=":memory:")

    assert len(msgs) == 4
    assert "今日交易作战卡" in msgs[0].title

    # 四条各自声明独立的业务去重身份：2026-08-02 那次事故里，同一份晨报在 31
    # 秒内推了两遍，靠内容哈希只挡住了正文碰巧一字不差的那一条。
    keys = [m.dedupe_key for m in msgs]
    assert all(keys), "晨报消息必须声明 dedupe_key，否则跨进程重复挡不住"
    assert len(set(keys)) == 4, "四条消息的身份不能撞车"
    assert all(k.startswith("morning_brief:") for k in keys)
    for text in (
        "方向倾向",
        "风险档位",
        "开盘策略",
        "关键触发",
        "重点观察",
        "事件风险",
        "来源/置信度",
        "不确定性",
        "高级执行窗口",
        "Base 剧本",
        "Bull 剧本",
        "Bear/防守",
        "禁做窗口",
        "盘前价位",
    ):
        assert text in msgs[0].body


def test_event_section_warns_when_source_unavailable():
    import trader.morning_brief as mb

    text = mb._build_event_section(["NVDA"], result=_calendar_result(issue=True))

    assert "事件源不可用" in text
    assert "未来 5 天没有高影响力事件 ✅" not in text


def test_calendar_events_status_uses_nasdaq_without_finnhub_economic(monkeypatch):
    import trader.calendar_events as ce

    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    monkeypatch.setattr(ce, "_fetch_nasdaq_economic_with_status", lambda *_: ([], None))

    result = ce.get_upcoming_events_with_status(symbols=[], days=5)

    assert result.events == []
    assert result.source_state("finnhub_economic") == "not_requested"
    assert result.source_state("nasdaq_economic") == "ok"
    assert result.has_source_issue is False
    assert result.warning_text == ""


def test_calendar_events_does_not_call_finnhub_economic(monkeypatch):
    import trader.calendar_events as ce

    monkeypatch.setenv("FINNHUB_API_KEY", "configured-but-not-used-for-economic")
    monkeypatch.setattr(ce, "_fetch_nasdaq_economic_with_status", lambda *_: ([], None))

    def fail_if_called(*_):
        raise AssertionError("Finnhub economic should not be called")

    monkeypatch.setattr(ce, "_fetch_economic_with_status", fail_if_called)

    result = ce.get_upcoming_events_with_status(symbols=[], days=5)

    assert result.source_state("finnhub_economic") == "not_requested"
    assert result.source_state("nasdaq_economic") == "ok"


def test_calendar_events_query_window_uses_pacific_date(monkeypatch):
    import trader.calendar_events as ce

    captured = {}

    def fake_nasdaq(from_s, to_s):
        captured["from_s"] = from_s
        captured["to_s"] = to_s
        return [], None

    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    monkeypatch.setattr(
        ce,
        "_now_pacific",
        lambda: datetime(2026, 6, 18, 1, tzinfo=timezone(timedelta(hours=-7))),
    )
    monkeypatch.setattr(ce, "_fetch_nasdaq_economic_with_status", fake_nasdaq)

    ce.get_upcoming_events_with_status(symbols=[], days=5)

    assert captured == {"from_s": "2026-06-18", "to_s": "2026-06-23"}


def test_event_section_warns_when_all_economic_sources_unavailable():
    import trader.morning_brief as mb
    from trader.calendar_events import CalendarEventsResult, EventSourceStatus

    result = CalendarEventsResult(
        events=[],
        sources=[
            EventSourceStatus("finnhub_economic", "error", "HTTP 403"),
            EventSourceStatus("nasdaq_economic", "error", "network blocked"),
            EventSourceStatus("yfinance_earnings", "ok"),
        ],
    )

    text = mb._build_event_section(["NVDA"], result=result)

    assert "事件源不可用" in text
    assert "未来 5 天没有高影响力事件 ✅" not in text


def test_event_confidence_uses_official_and_nasdaq_crosscheck():
    import trader.morning_brief as mb
    from trader.calendar_events import CalendarEventsResult, EventSourceStatus

    result = CalendarEventsResult(
        events=[],
        sources=[
            EventSourceStatus("official_fed_fomc", "ok"),
            EventSourceStatus("official_bls_calendar", "ok"),
            EventSourceStatus("official_bea_calendar", "ok"),
            EventSourceStatus("nasdaq_economic", "ok"),
        ],
    )

    assert "官方核心源 + Nasdaq" in mb._event_confidence(result)
    assert "Nasdaq 单源" not in mb._build_uncertainty_line(result, [], [])


def test_action_event_risk_dedupes_repeated_events():
    import trader.morning_brief as mb
    from trader.calendar_events import CalendarEvent, CalendarEventsResult

    today = str(mb._today_pacific_date())
    retail = CalendarEvent(
        date=today,
        time_str="08:30 ET",
        title_cn="零售销售 🛒",
        title_en="Retail Sales",
        impact="high",
        category="economic",
    )
    fomc_projection = CalendarEvent(
        date=today,
        time_str="14:00 ET",
        title_cn="美联储利率/点阵图相关 🏦",
        title_en="Interest Rate Projection",
        impact="critical",
        category="fomc",
    )
    fomc_decision = CalendarEvent(
        date=today,
        time_str="14:00 ET",
        title_cn="FOMC 事件 🏦",
        title_en="FOMC Statement",
        impact="critical",
        category="fomc",
    )
    result = CalendarEventsResult(events=[retail, fomc_projection, fomc_decision])

    text = mb._action_event_risk(result)

    assert text.index("14:00 ET") < text.index("08:30 ET")
    assert text.count("14:00 ET") == 1


def test_today_events_uses_pacific_event_date(monkeypatch):
    import trader.morning_brief as mb
    from trader.calendar_events import CalendarEvent, CalendarEventsResult

    monkeypatch.setattr(
        mb,
        "_now_pacific",
        lambda: datetime(2026, 6, 17, 22, tzinfo=timezone(timedelta(hours=-7))),
    )
    event = CalendarEvent(
        date="2026-06-18",
        time_str="00:30 ET",
        title_cn="午夜后事件",
        title_en="Late Event",
        impact="high",
        category="economic",
    )

    today = mb._today_events(CalendarEventsResult(events=[event]))

    assert today == [event]
    assert mb._format_event_time(event).startswith("21:30 PT / 00:30 ET")


def test_pro_execution_window_uses_levels_events_and_relative_strength():
    import trader.morning_brief as mb
    from trader.calendar_events import CalendarEvent, CalendarEventsResult

    today = str(mb._today_pacific_date())
    technicals = {
        "SPY": {
            "ma20": 590.0,
            "support": 585.0,
            "resistance": 603.0,
        },
        "QQQ": {
            "ma20": 500.0,
            "support": 492.0,
            "resistance": 510.0,
        },
    }
    event_result = CalendarEventsResult(events=[
        CalendarEvent(
            date=today,
            time_str="14:00 ET",
            title_cn="FOMC 事件",
            title_en="FOMC Statement",
            impact="critical",
            category="fomc",
        )
    ])
    text = mb._build_pro_execution_window(
        mkt={"ES=F": {"pct": 1.0}, "NQ=F": {"pct": 1.0}},
        event_result=event_result,
        technicals=technicals,
        movers=[
            {
                "symbol": "NVDA",
                "pct": 2.4,
                "pm_high": 103.0,
                "pm_low": 99.0,
                "prior_high": 101.0,
                "prior_low": 96.0,
            },
            {
                "symbol": "MSFT",
                "pct": -1.2,
                "pm_high": 98.0,
                "pm_low": 94.0,
                "prior_high": 100.0,
                "prior_low": 95.0,
            },
        ],
    )

    assert "Base 剧本" in text
    assert "重大事件日" in text
    assert "Bull 剧本" in text
    assert "SPY 站回20MA 590.00" in text
    assert "QQQ 跌破昨低 492.00" in text
    assert "14:00 ET" in text
    assert "强于指数：NVDA +2.40%" in text
    assert "弱于指数：MSFT -1.20%" in text
    assert "NVDA: gap +2.40%；盘前区间 99.00-103.00；昨区间 96.00-101.00" in text


def test_technical_indicator_section_shows_computed_values():
    import trader.morning_brief as mb

    text = mb._build_technical_indicator_section({
        "SPY": {
            "close": 600.0,
            "chg": 0.4,
            "ma20": 590.0,
            "ma50": 580.0,
            "ma200": 530.0,
            "vs20": 1.7,
            "vs50": 3.4,
            "vs200": 13.2,
            "rsi14": 58.0,
            "atr14_pct": 0.9,
            "support": 595.0,
            "resistance": 603.0,
            "trend": "多头排列",
        }
    })

    assert "关键技术指标" in text
    assert "RSI14 58" in text
    assert "ATR14 0.9%" in text
    assert "20MA" in text


def test_sector_section_does_not_call_all_red_market_offensive():
    import trader.morning_brief as mb

    mkt = {
        ticker: {"pct": -0.1 - idx * 0.1}
        for idx, (ticker, _) in enumerate(mb._SECTOR_MAP)
    }

    text = mb._build_sector_section(mkt)

    assert "全线收跌" in text
    assert "进攻型板块领涨" not in text


def test_company_symbols_filters_broad_etfs():
    import trader.morning_brief as mb

    assert mb._company_symbols(["QQQ", "SPY", "NVDA", "AAPL"]) == ["NVDA", "AAPL"]


def test_nasdaq_economic_row_becomes_calendar_event():
    from trader.calendar_events import _nasdaq_row_to_event

    event = _nasdaq_row_to_event(
        "2026-06-18",
        {
            "country": "United States",
            "gmt": "08:30",
            "eventName": "Retail Sales",
            "consensus": "0.5%",
            "previous": "0.4%",
        },
    )

    assert event is not None
    assert event.date == "2026-06-18"
    assert event.time_str == "08:30 ET"
    assert event.title_en == "Retail Sales"
    assert event.impact == "high"
    assert event.category == "economic"
    assert "0.5%" in event.note
    assert "0.4%" in event.note


def test_nasdaq_empty_calendar_day_is_not_source_failure(monkeypatch):
    import urllib.request

    import trader.calendar_events as ce

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *_, **__: _FakeResponse({"data": None}),
    )

    events, error = ce._fetch_nasdaq_economic_with_status(
        "2026-06-18", "2026-06-18"
    )

    assert events == []
    assert error is None


def test_nasdaq_fed_regional_data_stays_economic():
    from trader.calendar_events import _nasdaq_row_to_event

    philly = _nasdaq_row_to_event(
        "2026-06-19",
        {
            "country": "United States",
            "gmt": "08:30",
            "eventName": "Philadelphia Fed Manufacturing Index",
        },
    )
    balance_sheet = _nasdaq_row_to_event(
        "2026-06-19",
        {
            "country": "United States",
            "gmt": "16:30",
            "eventName": "Fed's Balance Sheet",
        },
    )

    assert philly is not None
    assert philly.category == "economic"
    assert philly.impact == "medium"
    assert balance_sheet is not None
    assert balance_sheet.category == "economic"


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return json.dumps(self._payload).encode("utf-8")


def test_stock_catalysts_balances_finnhub_company_news(monkeypatch):
    import trader.morning_brief as mb
    import trader.news as news

    now = datetime.now(timezone.utc)
    events = [
        SimpleNamespace(
            symbol="TSLA",
            title=f"TSLA earnings guidance update {i}",
            summary="earnings guidance",
            ts=now - timedelta(hours=i),
            severity=0.5,
        )
        for i in range(5)
    ] + [
        SimpleNamespace(
            symbol="NVDA",
            title="NVDA price target upgrade",
            summary="analyst upgrade",
            ts=now - timedelta(hours=1),
            severity=0.5,
        ),
        SimpleNamespace(
            symbol="AAPL",
            title="AAPL daily market update",
            summary="quiet roundup",
            ts=now - timedelta(hours=2),
            severity=0.5,
        ),
        SimpleNamespace(
            symbol="MSFT",
            title="MSFT antitrust investigation",
            summary="regulatory investigation",
            ts=now - timedelta(hours=3),
            severity=0.5,
        ),
    ]

    class QuietWSCN:
        def __init__(self, *_, **__):
            pass

        def poll(self, *_):
            return []

    class FakeFinnhub:
        def __init__(self, *_, **__):
            pass

        def poll(self, *_):
            return events

    monkeypatch.setattr(news, "WallStreetCNSource", QuietWSCN)
    monkeypatch.setattr(news, "FinnhubSource", FakeFinnhub)

    text = mb._build_stock_catalysts(["TSLA", "NVDA", "AAPL", "MSFT"])

    assert text.count("**TSLA**") <= 2
    assert "**NVDA**" in text
    assert "**MSFT**" in text
    assert "**AAPL**" not in text


def test_premarket_movers_use_yahoo_chart_snapshot(monkeypatch):
    import urllib.request

    import trader.morning_brief as mb
    import yfinance as yf

    payload = {
        "chart": {
            "result": [{
                "meta": {
                    "chartPreviousClose": 100.0,
                    "regularMarketPrice": 101.0,
                    "currentTradingPeriod": {
                        "pre": {"start": 3},
                        "regular": {"start": 5},
                    },
                },
                "timestamp": [1, 2, 3, 4],
                "indicators": {
                    "quote": [{
                        "close": [98.0, 100.0, 101.0, 102.0],
                        "high": [99.0, 101.0, 102.5, 103.0],
                        "low": [97.0, 99.0, 101.5, 101.8],
                        "open": [98.0, 99.5, 100.0, 101.0],
                        "volume": [100, 200, 300, 400],
                    }]
                },
            }]
        }
    }
    monkeypatch.setattr(urllib.request, "urlopen", lambda *_, **__: _FakeResponse(payload))
    monkeypatch.setattr(
        yf,
        "Ticker",
        lambda _: SimpleNamespace(
            info={
                "preMarketPrice": 102.0,
                "regularMarketPreviousClose": 100.0,
            }
        ),
    )

    movers = mb._fetch_premarket_movers(["NVDA"])

    assert movers[0]["symbol"] == "NVDA"
    assert movers[0]["pre_price"] == 102.0
    assert movers[0]["pct"] == 2.0
    assert movers[0]["pm_high"] == 103.0
    assert movers[0]["pm_low"] == 101.5
    assert movers[0]["pm_volume"] == 700
    assert movers[0]["prior_high"] == 101.0
    assert movers[0]["prior_low"] == 97.0
    assert movers[0]["source"] == "Yahoo chart + yfinance check"
    assert movers[0]["as_of"]
    assert "盘前量 700" in mb._build_premarket_section(["NVDA"], movers)
    assert "盘前区间 101.50-103.00" in mb._build_premarket_section(["NVDA"], movers)


def test_yfinance_crosscheck_marks_price_discrepancy():
    import trader.morning_brief as mb

    movers = [{"symbol": "NVDA", "pre_price": 100.0, "source": "Yahoo chart"}]

    mb._attach_yfinance_crosscheck(movers, "NVDA", 101.0)

    assert movers[0]["price_discrepancy_pct"] == 1.0
    assert "yfinance check" in movers[0]["source"]
    assert "价差 +1.00%" in mb._format_premarket_quality(movers[0])


def test_account_risk_lines_surface_largest_and_moving_positions():
    import trader.morning_brief as mb

    lines = mb._build_account_risk_lines(
        positions=[
            {"symbol": "AAPL", "market_value": 60000},
            {"symbol": "NVDA", "market_value": 15000},
        ],
        equity=100000,
        cash_pct=18.0,
        movers=[
            {"symbol": "AAPL", "pct": -1.2},
            {"symbol": "MSFT", "pct": 2.0},
        ],
    )
    text = "\n".join(lines)

    assert "最大暴露：AAPL 60.0%" in text
    assert "集中度高" in text
    assert "不新增同方向高 beta 仓位" in text
    assert "盘前异动持仓：AAPL -1.20%" in text
    assert "现金缓冲 18.0% 偏低" in text
    assert "新计划优先减半" in text


def test_removed_dead_sources_stay_removed():
    """WSB 与 CBOE Put/Call 已随上游封禁/下线一起删除，别再被顺手加回来。

    Reddit 的 .json 接口对未授权请求恒 403（RSS 虽可访问但不带 upvote 分数，
    支撑不了原来的热度阈值逻辑）；CBOE 两个公开 CSV 端点分别 403/404。
    """
    import trader.morning_brief as mb

    assert not hasattr(mb, "_build_wsb_section")
    assert not hasattr(mb, "_fetch_cboe_put_call")


def test_yahoo_trending_only_shows_watchlist_or_large_moves(monkeypatch):
    import urllib.request

    import trader.morning_brief as mb

    payload = {
        "finance": {
            "result": [{"quotes": [{"symbol": "AAA"}, {"symbol": "NVDA"}]}]
        }
    }
    monkeypatch.setattr(urllib.request, "urlopen", lambda *_, **__: _FakeResponse(payload))

    fake_yf = SimpleNamespace(
        download=lambda *_, **__: pd.DataFrame({
            "AAA": [100.0, 101.0],
            "NVDA": [100.0, 101.0],
        })
    )
    monkeypatch.setitem(sys.modules, "yfinance", fake_yf)

    text = mb._build_trending_section(["NVDA"])

    assert "NVDA" in text
    assert "AAA" not in text
