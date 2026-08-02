"""
trader/morning_brief.py
每日晨报 Discord 推送 — 美东时间可配置小时自动触发（默认 9AM），发 4 条消息。

消息 1 — 今日作战卡
  风险档位 · 建议仓位 · 开盘策略 · 事件风险 · 重点观察/谨慎标的

消息 2 — 市场依据
  市场环境 · 恐慌贪婪指数 · 大盘/期货 · 宏观异动 · 板块轮动 · SPY/QQQ 技术面

消息 3 — 事件新闻
  今日+近期重要事件 · OpEx · 财报 · 异常社交/热搜 · 多源市场要闻

消息 4 — 今日操作
  自选股盘前异动 · 模拟账户快照（权益/仓位/浮盈）

调用方式：
  runtime.py 每天美东 MORNING_BRIEF_HOUR_ET 点（.env 可配置，默认 9）自动触发一次
  手动测试：uv run python -m trader.morning_brief
  UI：系统 Tab → Discord 推送 → 报告类型选「晨报」→ 发送
"""

from __future__ import annotations

import json
import logging
import os
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from .config import settings
from .models import Notification

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[1]
_DB_PATH = str(_ROOT / "trade.duckdb")

try:
    _PACIFIC_TZ = ZoneInfo("America/Los_Angeles")
    _EASTERN_TZ = ZoneInfo("America/New_York")
except Exception:
    _PACIFIC_TZ = timezone(timedelta(hours=-8), "America/Los_Angeles")
    _EASTERN_TZ = timezone(timedelta(hours=-5), "America/New_York")


def _now_pacific() -> datetime:
    return datetime.now(_PACIFIC_TZ)


def _today_pacific_date():
    return _now_pacific().date()


# 指数 ETF
_INDEX_MAP = [
    ("SPY", "S&P 500"),
    ("QQQ", "Nasdaq 100"),
    ("IWM", "小盘股 Russell 2000"),
    ("DIA", "道琼斯"),
]
# 主要期货（用于 _build_market_overview 合并显示，保留供 _ALL_BATCH 批量拉取）
_FUTURES_MAP = [
    ("ES=F", "S&P 500 期货"),
    ("NQ=F", "Nasdaq 期货"),
    ("RTY=F", "Russell 期货"),
]
# 指数 ETF → 对应期货（用于合并显示，避免重复）
_ETF_TO_FUTURES = {"SPY": "ES=F", "QQQ": "NQ=F", "IWM": "RTY=F"}
_NON_COMPANY_SYMBOLS = {
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "VTI",
    "VOO",
    "IVV",
    "RSP",
    "XLK",
    "XLC",
    "XLY",
    "XLF",
    "XLE",
    "XLI",
    "XLV",
    "XLB",
    "XLP",
    "XLU",
    "XLRE",
    "TQQQ",
    "SQQQ",
    "SOXL",
    "SOXS",
    "SPXL",
    "SPXS",
}

# 有新闻、但没有财报和分析师评级的 ETF。
#
# 不能直接并进 _NON_COMPANY_SYMBOLS：那个集合会把标的整个排除出"个股"处理，
# 而这些 ETF 的 Finnhub 公司新闻和盘前异动都是有效内容（GLD/TLT 每天都有
# 实质新闻）。这里只用来跳过财报日历和分析师评级两类查询——对 ETF 调
# yfinance 的 calendar / upgrades_downgrades 必定返回
# "No fundamentals data found"，每份晨报都在日志里刷一串 404 告警，还白白
# 多打了往返请求。
_ETF_NO_FUNDAMENTALS = {
    # 债券
    "TLT", "IEF", "SHY", "AGG", "BND", "LQD", "HYG", "JNK", "TIP", "MUB",
    # 商品 / 贵金属
    "GLD", "IAU", "SLV", "USO", "UNG", "DBC", "PDBC",
    # 货币
    "UUP", "FXE", "FXY",
    # 波动率
    "VXX", "UVXY", "VIXY", "SVXY",
    # 行业/主题 ETF（成分股有财报，ETF 本身没有）
    "GDX", "GDXJ", "SMH", "XBI", "IBB", "KRE", "XOP", "XME", "ITB", "JETS",
    "ARKK", "ARKG", "ARKW",
    # 海外 / 区域
    "EEM", "EFA", "FXI", "EWZ", "EWJ", "INDA", "VEA", "VWO",
}

# 宏观指标（^TYX 30Y 改为 ^IRX 3M 短端，用于计算收益率曲线倒挂）
_MACRO_MAP = [
    # 利率（^IRX 只用于收益率曲线计算，不单独显示）
    ("^TNX", "10Y 美债", "%", True),
    ("^IRX", "_curve_only", "%", True),
    # 恐慌 & 压力
    ("^VIX", "VIX 恐慌指数", "", True),  # 主恐慌指数，VIX9D 折叠进此行
    ("^VIX9D", "_vix9d_only", "", True),  # 只供 VIX 行计算近期溢价
    ("TLT", "20Y 美债 TLT", "", True),  # 避险资金流向
    ("HYG", "高收益债 HYG", "", True),  # 信用利差代理
    # 美元
    ("UUP", "美元指数", "", True),
    # 大宗商品
    ("GC=F", "黄金", "/oz", False),
    ("CL=F", "原油 WTI", "/桶", False),
    ("HG=F", "铜（经济先行）", "/lb", False),
    # 风险资产
    ("BTC-USD", "比特币", "", False),
    # 海外三市（跨时区风险传导）
    ("EWJ", "日本 EWJ", "", True),
    ("FXI", "中国 FXI", "", True),
    ("VGK", "欧洲 VGK", "", True),
]
# 11个板块 ETF
_SECTOR_MAP = [
    ("XLK", "科技"),
    ("XLC", "通信"),
    ("XLY", "非必需消费"),
    ("XLF", "金融"),
    ("XLE", "能源"),
    ("XLI", "工业"),
    ("XLV", "医疗"),
    ("XLB", "材料"),
    ("XLP", "必需消费"),
    ("XLU", "公用事业"),
    ("XLRE", "房地产"),
]

_ALL_BATCH = (
    [t for t, _ in _INDEX_MAP]
    + [t for t, _ in _FUTURES_MAP]
    + [t for t, _, __, ___ in _MACRO_MAP]
    + [t for t, _ in _SECTOR_MAP]
    + ["^VIX"]
)


# ═══════════════════════════════════════════════════════════════════════════
# 对外接口
# ═══════════════════════════════════════════════════════════════════════════


def should_send_brief(now_utc: datetime, last_date: Optional[str]) -> bool:
    today = now_utc.astimezone(_PACIFIC_TZ).strftime("%Y-%m-%d")
    now_et = now_utc.astimezone(_EASTERN_TZ)
    return now_et.hour == settings.morning_brief_hour_et and today != last_date


def send_morning_brief(
    symbols: List[str],
    db_path: str = _DB_PATH,
    auto_trade: Optional[bool] = None,
) -> Any:
    """发送晨报，返回汇总后的 DeliveryOutcome（真值语义 = 确实送达）。"""
    from .notify import DiscordNotifier

    # Both callers of send_morning_brief() — the "发送晨报" UI button and the
    # 9AM ET auto-brief in runtime._maybe_morning_brief — are already a
    # deliberate, authorized send (explicit click, or a schedule the user
    # turned on). Not passing external_send_enabled=True here left every
    # brief silently BLOCKED behind the DISCORD_EXTERNAL_SEND_ENABLED gate,
    # unlike manual_push.py's send_intraday_levels_push() which does opt in.
    from .notify import DeliveryOutcome, summarize

    notifier = DiscordNotifier(external_send_enabled=True)
    msgs = build_morning_brief(
        symbols=symbols, db_path=db_path, auto_trade=auto_trade
    )
    outcomes = []
    for msg in msgs:
        try:
            outcomes.append(notifier.send(msg))
        except Exception as exc:
            logger.warning("晨报推送失败: %s", exc)
            outcomes.append(DeliveryOutcome("FAILED", type(exc).__name__))
    result = summarize(outcomes)
    logger.info("晨报推送完成，共 %d 条消息，结果 %s", len(msgs), result.status)
    return result


def build_morning_brief(
    symbols: List[str],
    db_path: str = _DB_PATH,
    auto_trade: Optional[bool] = None,
) -> List[Notification]:
    """组装 4 条 Notification，不实际推送，方便测试。

    这里刻意不做任何长度裁剪：内容该写多完整就写多完整，压进 Discord 尺寸是
    发送层 discord_limits.fit_notification() 的职责（沿语义边界分页，并如实
    标注省略量）。以前每条正文各写一次 ``[:4000]``，数字既不对（正文上限是
    4096），又会从半句话中间砍断。
    """
    now = _now_pacific()
    date_str = now.strftime("%m/%d")
    wd_cn = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][now.weekday()]

    # 去重身份用美东交易日：这是一份美股报告，"哪一天的晨报"该按交易日算。
    # 四条消息各有各的身份，因为它们是四条独立的 Discord 消息，其中任何一条
    # 重发都算重复。
    brief_date = datetime.now(timezone.utc).astimezone(_EASTERN_TZ).strftime("%Y-%m-%d")

    # 批量拉行情（一次 HTTP 请求）
    company_symbols = _company_symbols(symbols)
    mkt = _fetch_batch()
    event_result = _get_calendar_events_result(company_symbols, days=5)
    technical_snapshot = _fetch_index_technicals()
    priority_news = _fetch_priority_market_news(symbols)
    premarket_movers = _fetch_premarket_movers(company_symbols)

    # ── 消息 1：今日作战卡 ──────────────────────────────────────────────────
    action_text, direction_bias, direction_reasons = _build_action_card(
        mkt=mkt,
        symbols=symbols,
        event_result=event_result,
        technicals=technical_snapshot,
        macro_news=priority_news,
        premarket_movers=premarket_movers,
    )
    # 把今天的方向判断记下来，收盘报告才能对账。只记第一次：手动重发晨报会
    # 用当时的行情重算出不同的方向，覆盖掉早上那次就等于让判断跟着结果走。
    try:
        from .brief_review import BriefCallStore

        BriefCallStore(db_path).record(
            trading_date=brief_date,
            bias=direction_bias,
            reasons=direction_reasons,
        )
    except Exception as exc:  # noqa: BLE001 - 记录失败不该让晨报发不出去
        logger.warning("晨报方向记录失败: %s", exc)
    # 整份晨报的措辞（"开盘策略"、"盘前快照"等）默认假设是盘前发的。手动
    # "发送晨报"按钮没有时段限制，收盘后重发会呈现已经过期的盘前框架，
    # 之前完全没有提示——这里显式标注一下，而不是让读者误以为是最新判断。
    from .market_calendar import session_now as _session_now
    _session = _session_now()
    if _session != "pre":
        _session_note = {
            "open": "⚠️ 当前是盘中，不是盘前——下面的"
            "「开盘策略」「盘前快照」是这份简报生成时的盘前框架，"
            "可能已经被盘中走势打破，请结合实时行情判断。",
            "post": "⚠️ 当前是盘后/已收盘，不是盘前——这份简报的"
            "「开盘策略」框架针对的是已经过去的开盘时段，"
            "不适用于隔夜/次日交易，仅供复盘参考。",
            "closed": "⚠️ 当前休市，不是盘前——这份简报的"
            "「开盘策略」框架针对的是上一个交易日，"
            "不代表下一个交易日的最新判断。",
        }[_session]
        action_text = _session_note + "\n\n" + action_text
    msg1 = Notification(
        title=f"🎯 {date_str} {wd_cn} · 今日交易作战卡",
        body=action_text,
        kind="review",
        dedupe_key=f"morning_brief:action:{brief_date}",
    )

    # ── 消息 2：市场依据 + 板块技术 ──────────────────────────────────────────
    _live_vix_data = mkt.get("^VIX")
    regime_text, _ = _build_regime_section(
        live_vix=float(_live_vix_data["price"]) if _live_vix_data else None
    )
    fg_text = _build_fear_greed()
    market_text = _build_market_overview(mkt)  # 合并指数+期货，不再分开
    macro_text = _build_macro_section(mkt)  # 含收益率曲线倒挂判断
    sector_text = _build_sector_section(mkt)
    tech_text = _build_technical_indicator_section(technical_snapshot)

    body2 = _join(
        [regime_text, fg_text, market_text, macro_text, sector_text, tech_text]
    )
    msg2 = Notification(
        title="📊 市场依据 · 指标计算",
        body=body2 or "市场依据暂不可用，请以开盘价格和账户风控为准。",
        kind="review",
        dedupe_key=f"morning_brief:market:{brief_date}",
    )

    # ── 消息 3：事件 + 异常新闻/社交 ────────────────────────────────────────
    event_text = _build_event_section(symbols, result=event_result)
    earnings_text = (
        "" if "财报" in event_text else _build_earnings_section(company_symbols)
    )
    opex_text = _build_opex_section()
    trending_text = _build_trending_section(symbols)
    analyst_text = _build_analyst_changes(company_symbols)
    news_text = _build_news_section(symbols, events=priority_news)

    body3 = _join(
        [
            event_text,
            news_text,
            earnings_text,
            opex_text,
            analyst_text,
            trending_text,
        ]
    )
    msg3 = Notification(
        title="📅 事件风险 · 异常新闻",
        body=body3 or "暂无可确认的高优先级事件/新闻，请开盘前人工核对经济日历。",
        kind="news",
        dedupe_key=f"morning_brief:events:{brief_date}",
    )

    # ── 消息 4：各股前瞻 + 盘前快照 + 账户 ────────────────────────────────
    catalysts_text = _build_stock_catalysts(company_symbols)
    premarket_text = _build_premarket_section(symbols, movers=premarket_movers)
    status_text = _build_status_section(db_path, movers=premarket_movers)

    carryover_text = _build_carryover_section(db_path)
    automation_text = _build_automation_notice(auto_trade)

    body4 = _join(
        [carryover_text, automation_text, catalysts_text, premarket_text, status_text]
    )
    msg4 = Notification(
        title="📋 今日准备清单 · 各股前瞻",
        body=body4,
        kind="plan",
        dedupe_key=f"morning_brief:checklist:{brief_date}",
    )

    return [msg1, msg2, msg3, msg4]


# ═══════════════════════════════════════════════════════════════════════════
# 消息 1：行动优先作战卡
# ═══════════════════════════════════════════════════════════════════════════


def _get_calendar_events_result(symbols: List[str], days: int = 5) -> Any:
    try:
        from .calendar_events import get_upcoming_events_with_status

        return get_upcoming_events_with_status(symbols=symbols, days=days)
    except Exception as exc:
        logger.warning("事件日历获取失败: %s", exc)
        from .calendar_events import CalendarEventsResult, EventSourceStatus

        return CalendarEventsResult(
            events=[],
            sources=[
                EventSourceStatus("calendar_events", "error", exc.__class__.__name__)
            ],
        )


def _company_symbols(symbols: List[str]) -> List[str]:
    out = []
    for symbol in symbols:
        sym = str(symbol or "").upper()
        if not sym or sym in _NON_COMPANY_SYMBOLS:
            continue
        out.append(sym)
    return out


def _fundamentals_symbols(symbols: List[str]) -> List[str]:
    """能查到财报/评级的标的：在 _company_symbols 基础上再去掉 ETF。"""
    return [s for s in _company_symbols(symbols) if s not in _ETF_NO_FUNDAMENTALS]


def _build_action_card(
    mkt: Dict[str, dict],
    symbols: List[str],
    event_result: Any,
    premarket_movers: List[dict],
    technicals: Optional[Dict[str, dict]] = None,
    macro_news: Optional[List[Any]] = None,
) -> str:
    technicals = technicals or {}
    macro_news = macro_news or []
    risk_label, position_line, risk_reasons = _action_risk_profile(mkt, event_result)
    bias_line, trigger_line, direction_reasons = _action_market_bias(
        mkt, event_result, technicals, macro_news
    )
    open_line = _action_open_strategy(mkt)
    event_line = _action_event_risk(event_result)
    macro_line = _action_macro_news_line(macro_news)
    watch_line = _action_watchlist(symbols, mkt, premarket_movers)
    caution_line = _action_caution_list(event_result, premarket_movers, risk_reasons)
    confidence_line = _build_data_confidence_line(
        mkt, event_result, premarket_movers, macro_news
    )
    uncertainty_line = _build_uncertainty_line(
        event_result, premarket_movers, macro_news
    )
    pro_window = _build_pro_execution_window(
        mkt=mkt,
        event_result=event_result,
        technicals=technicals,
        movers=premarket_movers,
    )
    main_reasons = _unique_texts(direction_reasons + risk_reasons)

    lines = [
        "**🎯 今日交易结论（先看这个）**",
        f"**方向倾向：** {bias_line}",
        f"**风险档位：** {risk_label}",
        f"**建议仓位：** {position_line}",
        f"**开盘策略：** {open_line}",
        f"**关键触发：** {trigger_line}",
        f"**事件风险：** {event_line}",
        f"**宏观新闻：** {macro_line}",
        f"**重点观察：** {watch_line}",
        f"**谨慎标的：** {caution_line}",
        f"**来源/置信度：** {confidence_line}",
        f"**不确定性：** {uncertainty_line}",
        "",
        pro_window,
        "",
        "纪律：不追第一根 5m K；事件前后 15-30 分钟不新开仓；所有计划仍以止损和确定性风控为准。",
        "_以上是 AI 模型基于当前数据给出的直接结论，不是万无一失的预测——自行判断是否采纳，仓位/止损自担。_",
    ]

    if main_reasons:
        lines.insert(4, f"**主要原因：** {'；'.join(main_reasons[:4])}")

    # 连同方向判断一起返回：收盘复盘要拿它来对账"今天早上说对了没有"，在外面
    # 重算一遍会因为行情已经变化而得出不同的方向，那样对账就失去意义了。
    return "\n".join(lines), bias_line, main_reasons


def _build_uncertainty_line(
    event_result: Any,
    movers: List[dict],
    macro_news: List[Any],
) -> str:
    items: List[str] = []
    source_state = getattr(event_result, "source_state", lambda _name: "not_requested")
    official_complete = _official_core_sources_complete(event_result)
    official_any = _official_event_sources_ok(event_result)
    nasdaq_ok = source_state("nasdaq_economic") == "ok"
    if bool(getattr(event_result, "has_source_issue", False)):
        items.append("经济事件源降级")
    elif official_complete and nasdaq_ok:
        pass
    elif official_any and not nasdaq_ok:
        items.append("官方事件源为重点覆盖，非全量经济日历")
    elif nasdaq_ok:
        items.append("经济事件为 Nasdaq 单源")
    if bool(getattr(event_result, "has_partial_source_issue", False)):
        items.append("部分补充事件源异常")

    if not movers:
        items.append("盘前个股价缺失")
    elif not any(m.get("pm_volume") for m in movers):
        items.append("盘前量能未确认")
    if any(m.get("price_discrepancy_pct", 0) for m in movers):
        items.append("盘前价存在源间差异")
    if macro_news:
        items.append("新闻未做英文原始源交叉验证")
    else:
        items.append("高优先级新闻为空不等于无风险")
    return "；".join(items[:4]) if items else "暂无明显数据缺口。"


def _build_data_confidence_line(
    mkt: Dict[str, dict],
    event_result: Any,
    movers: List[dict],
    macro_news: List[Any],
) -> str:
    parts = [
        f"事件 {_event_confidence(event_result)}",
        f"行情 {_market_confidence(mkt)}",
        f"盘前 {_premarket_confidence(movers)}",
        f"新闻 {_news_confidence(macro_news)}",
    ]
    return "；".join(parts)


def _event_confidence(event_result: Any) -> str:
    source_state = getattr(event_result, "source_state", lambda _name: "not_requested")
    if bool(getattr(event_result, "has_source_issue", False)):
        return "低（经济事件源不可用，需人工复核）"
    official_complete = _official_core_sources_complete(event_result)
    official_any = _official_event_sources_ok(event_result)
    nasdaq_ok = source_state("nasdaq_economic") == "ok"
    if official_complete and nasdaq_ok:
        return "高（官方核心源 + Nasdaq 交叉）"
    if official_complete:
        return "中高（Fed/BLS/BEA 官方核心源）"
    if nasdaq_ok:
        return "中高（Nasdaq 单源）"
    if official_any:
        return "中（官方源部分覆盖，需人工补核）"
    return "中（非官方兜底源）"


def _official_event_sources_ok(event_result: Any) -> bool:
    if bool(getattr(event_result, "official_available", False)):
        return True
    return any(
        getattr(source, "name", "").startswith("official_")
        and getattr(source, "state", "") == "ok"
        for source in getattr(event_result, "sources", []) or []
    )


def _official_core_sources_complete(event_result: Any) -> bool:
    if bool(getattr(event_result, "official_core_complete", False)):
        return True
    source_state = getattr(event_result, "source_state", lambda _name: "not_requested")
    return all(
        source_state(name) == "ok"
        for name in (
            "official_fed_fomc",
            "official_bls_calendar",
            "official_bea_calendar",
        )
    )


def _market_confidence(mkt: Dict[str, dict]) -> str:
    sources = _source_set(mkt.values())
    if not sources:
        return "低（行情缺失）"
    if "yfinance" in sources and "yahoo_chart" in sources:
        return "中高（yfinance + Yahoo chart）"
    return f"中（{_display_sources(sources)} 单源）"


def _premarket_confidence(movers: List[dict]) -> str:
    sources = _source_set(movers)
    if not sources:
        return "低（盘前价缺失或无异动）"
    return f"中（{_display_sources(sources)} 免费源）"


def _news_confidence(macro_news: List[Any]) -> str:
    if not macro_news:
        return "低（未发现高优先级新闻）"
    return "中（华尔街见闻单源聚合）"


def _source_set(rows: Any) -> set[str]:
    out: set[str] = set()
    for row in rows:
        source = row.get("source") if isinstance(row, dict) else None
        if source:
            out.add(str(source))
    return out


def _display_sources(sources: set[str]) -> str:
    labels = {
        "yfinance": "yfinance",
        "yahoo_chart": "Yahoo chart",
        "Yahoo chart": "Yahoo chart",
    }
    return " + ".join(labels.get(s, s) for s in sorted(sources))


def _build_pro_execution_window(
    mkt: Dict[str, dict],
    event_result: Any,
    technicals: Dict[str, dict],
    movers: List[dict],
) -> str:
    return "\n".join(
        [
            "**🧭 高级执行窗口（可选）**",
            f"**Base 剧本：** {_pro_base_case(mkt, event_result, technicals)}",
            f"**Bull 剧本：** {_pro_long_setup(mkt, technicals)}",
            f"**Bear/防守：** {_pro_short_setup(mkt, technicals, movers)}",
            f"**禁做窗口：** {_pro_no_trade_window(event_result)}",
            f"**相对强弱：** {_pro_relative_strength(mkt, movers)}",
            f"**盘前价位：** {_pro_premarket_levels(movers)}",
        ]
    )


def _pro_base_case(
    mkt: Dict[str, dict],
    event_result: Any,
    technicals: Dict[str, dict],
) -> str:
    critical = any(
        getattr(e, "impact", "") == "critical" for e in _today_events(event_result)
    )
    gap_text = _gap_context_text(mkt)
    spy = technicals.get("SPY")
    qqq = technicals.get("QQQ")
    trend_text = ""
    if spy and qqq and "close" in spy and "close" in qqq:
        trend_text = (
            f"SPY {'低于' if spy['close'] < spy['ma20'] else '高于'}20MA，"
            f"QQQ {'低于' if qqq['close'] < qqq['ma20'] else '高于'}20MA"
        )
    if critical:
        return (
            f"重大事件日，默认轻仓/观望；{gap_text}{trend_text}，事件后再按确认方向。"
        )
    return f"默认等开盘 5/15m 区间形成；{gap_text}{trend_text or '趋势需开盘确认'}。"


def _gap_context_text(mkt: Dict[str, dict]) -> str:
    es_pct = _mkt_pct(mkt, "ES=F")
    nq_pct = _mkt_pct(mkt, "NQ=F")
    values = [v for v in (es_pct, nq_pct) if v is not None]
    if not values:
        return "盘前期货不可用；"
    avg = sum(values) / len(values)
    if avg >= 0.8:
        return f"期货明显高开（均值 {avg:+.2f}%），不追第一波；"
    if avg <= -0.8:
        return f"期货明显低开（均值 {avg:+.2f}%），先看支撑是否失守；"
    if abs(avg) >= 0.35:
        return f"期货有盘前偏移（均值 {avg:+.2f}%）；"
    return f"期货接近平开（均值 {avg:+.2f}%）；"


def _pro_long_setup(mkt: Dict[str, dict], technicals: Dict[str, dict]) -> str:
    spy = technicals.get("SPY")
    qqq = technicals.get("QQQ")
    es_pct = _mkt_pct(mkt, "ES=F")

    if spy and qqq:
        return (
            f"SPY 站回20MA {spy['ma20']:.2f} 并突破昨高 {spy['resistance']:.2f}，"
            f"QQQ 同步站回20MA {qqq['ma20']:.2f}；开盘 5/15m OR 上方承接才考虑。"
        )
    if es_pct is not None and es_pct > 0.5:
        return "期货高开日只做回踩承接，不追第一波；开盘 5/15m OR 上破后再看多。"
    return "等待 SPY/QQQ 同向突破开盘 5/15m 区间，并且回踩不破 VWAP/OR 低点。"


def _pro_short_setup(
    mkt: Dict[str, dict],
    technicals: Dict[str, dict],
    movers: List[dict],
) -> str:
    spy = technicals.get("SPY")
    qqq = technicals.get("QQQ")
    weak = _relative_weak_movers(mkt, movers)
    weak_text = (
        f"优先看相对弱标的 {', '.join(_fmt_mover(m) for m in weak[:3])}。"
        if weak
        else ""
    )

    if spy and qqq:
        return (
            f"高开失败或 SPY 跌破昨低 {spy['support']:.2f} / QQQ 跌破昨低 {qqq['support']:.2f} "
            f"才转防守；{weak_text or '没有相对弱标的时，少做单股空头。'}"
        )
    return f"指数跌破开盘区间低点且 HYG/VIX 不配合风险偏好时才防守；{weak_text}"


def _pro_no_trade_window(event_result: Any) -> str:
    critical = [
        e
        for e in sorted(_today_events(event_result), key=_event_priority_key)
        if getattr(e, "impact", "") == "critical"
    ]
    if critical:
        event = critical[0]
        return (
            f"{_format_event_time(event)} 前 30 分钟降风险；事件后 10-15 分钟不追第一波，"
            "等第二次确认。"
        )
    high = [
        e
        for e in sorted(_today_events(event_result), key=_event_priority_key)
        if getattr(e, "impact", "") == "high"
    ]
    if high:
        event = high[0]
        return f"{_format_event_time(event)} 前后减少新开仓；数据公布后等 5m 收线确认。"
    return "开盘前 5 分钟和开盘第一根 5m K 不追单；没有量能确认就降级为观察。"


def _pro_relative_strength(mkt: Dict[str, dict], movers: List[dict]) -> str:
    strong = _relative_strong_movers(mkt, movers)
    weak = _relative_weak_movers(mkt, movers)
    if strong or weak:
        parts = []
        if strong:
            parts.append("强于指数：" + "、".join(_fmt_mover(m) for m in strong[:3]))
        if weak:
            parts.append("弱于指数：" + "、".join(_fmt_mover(m) for m in weak[:3]))
        return "；".join(parts)
    return "暂无明显相对强弱，先以 SPY/QQQ 方向为主，不单独押个股。"


def _pro_premarket_levels(movers: List[dict]) -> str:
    ranked = sorted(movers, key=lambda m: abs(float(m.get("pct", 0))), reverse=True)
    chunks = []
    for mover in ranked[:2]:
        levels = _format_premarket_levels(mover)
        if levels:
            chunks.append(f"{mover.get('symbol')}: {levels}")
    return (
        "；".join(chunks)
        if chunks
        else "盘前高低点暂不可用，开盘后以前 5/15m 区间替代。"
    )


def _relative_strong_movers(mkt: Dict[str, dict], movers: List[dict]) -> List[dict]:
    index_pct = _index_reference_pct(mkt)
    return sorted(
        [m for m in movers if float(m.get("pct", 0)) - index_pct >= 1.0],
        key=lambda m: float(m.get("pct", 0)) - index_pct,
        reverse=True,
    )


def _relative_weak_movers(mkt: Dict[str, dict], movers: List[dict]) -> List[dict]:
    index_pct = _index_reference_pct(mkt)
    return sorted(
        [m for m in movers if float(m.get("pct", 0)) - index_pct <= -1.0],
        key=lambda m: float(m.get("pct", 0)) - index_pct,
    )


def _index_reference_pct(mkt: Dict[str, dict]) -> float:
    values = [
        pct for pct in (_mkt_pct(mkt, "NQ=F"), _mkt_pct(mkt, "ES=F")) if pct is not None
    ]
    return sum(values) / len(values) if values else 0.0


def _action_market_bias(
    mkt: Dict[str, dict],
    event_result: Any,
    technicals: Dict[str, dict],
    macro_news: List[Any],
) -> tuple[str, str, List[str]]:
    score = 0
    reasons: List[str] = []

    for ticker, label in (("ES=F", "S&P 期货"), ("NQ=F", "Nasdaq 期货")):
        pct = _mkt_pct(mkt, ticker)
        if pct is None:
            continue
        if pct >= 0.5:
            score += 1
            reasons.append(f"{label}盘前走强 {pct:+.2f}%")
        elif pct <= -0.5:
            score -= 1
            reasons.append(f"{label}盘前走弱 {pct:+.2f}%")

    hyg_pct = _mkt_pct(mkt, "HYG")
    if hyg_pct is not None:
        if hyg_pct >= 0.35:
            score += 1
            reasons.append(f"HYG 走强 {hyg_pct:+.2f}%")
        elif hyg_pct <= -0.35:
            score -= 1
            reasons.append(f"HYG 走弱 {hyg_pct:+.2f}%")

    vix = _mkt_price(mkt, "^VIX")
    vix_pct = _mkt_pct(mkt, "^VIX")
    if vix is not None and vix >= 22:
        score -= 1
        reasons.append(f"VIX {vix:.1f} 高波动")
    if vix_pct is not None and vix_pct >= 8:
        score -= 1
        reasons.append(f"VIX 日变动 {vix_pct:+.1f}%")

    tech_score, tech_reasons = _technical_bias_score(technicals)
    score += tech_score
    reasons.extend(tech_reasons)

    news_score, news_reasons = _macro_news_bias_score(macro_news)
    score += news_score
    reasons.extend(news_reasons)

    today_events = _today_events(event_result)
    has_critical_event = any(
        getattr(e, "impact", "") == "critical" for e in today_events
    )
    trigger = _action_trigger_line(technicals)
    if has_critical_event:
        return (
            "事件前观望，不抢多空；事件后只跟随确认方向",
            trigger,
            _unique_texts(["重大事件日，消息前方向可信度下降"] + reasons),
        )

    if score >= 3:
        bias = "偏多，但只做回踩/突破确认后的多头，不追第一波"
    elif score <= -3:
        bias = "偏空/防守，反弹不过关键位再考虑空头或降低风险"
    else:
        bias = "中性震荡，先等开盘区间给方向"
    return bias, trigger, _unique_texts(reasons)


def _action_trigger_line(technicals: Dict[str, dict]) -> str:
    parts = []
    for name in ("SPY", "QQQ"):
        row = technicals.get(name)
        if not row:
            continue
        parts.append(
            f"{name} 站回20MA {row['ma20']:.2f} 且上破昨高 {row['resistance']:.2f} 才偏多，"
            f"跌破昨低 {row['support']:.2f} 转防守"
        )
    if parts:
        return "；".join(parts[:2])
    return "以前 15-30 分钟高低点为边界，放量突破再跟随。"


def _action_macro_news_line(events: List[Any]) -> str:
    if not events:
        return "近 14h 未发现高优先级宏观异常。"
    selected = sorted(
        events, key=lambda e: getattr(e, "ts", datetime.min), reverse=True
    )[:2]
    chunks = []
    for ev in selected:
        ts = getattr(ev, "ts", None)
        hhmm = _format_dt_pt(ts) if ts else ""
        text = (getattr(ev, "summary", "") or getattr(ev, "title", "")).strip()
        chunks.append(f"{hhmm} {text[:52]}")
    return "；".join(chunks)


def _technical_bias_score(technicals: Dict[str, dict]) -> tuple[int, List[str]]:
    score = 0
    reasons: List[str] = []
    for name in ("SPY", "QQQ"):
        row = technicals.get(name)
        if not row:
            continue
        close = row["close"]
        ma20 = row["ma20"]
        ma50 = row["ma50"]
        rsi = row["rsi14"]
        if close >= ma20 >= ma50:
            score += 1
            reasons.append(f"{name} 站上20/50MA")
        elif close < ma20 and close < ma50:
            score -= 1
            reasons.append(f"{name} 跌破20/50MA")
        if rsi >= 70:
            score -= 1
            reasons.append(f"{name} RSI {rsi:.0f} 过热")
        elif rsi <= 30:
            score -= 1
            reasons.append(f"{name} RSI {rsi:.0f} 极弱/高波动")
    return score, reasons


def _macro_news_bias_score(events: List[Any]) -> tuple[int, List[str]]:
    if not events:
        return 0, []

    negative_words = (
        "通胀",
        "加息",
        "收益率",
        "关税",
        "制裁",
        "战争",
        "袭击",
        "衰退",
        "下调",
        "调查",
        "暴跌",
        "危机",
        "油价上涨",
        "hawkish",
        "tariff",
    )
    positive_words = (
        "降息",
        "停火",
        "协议",
        "刺激",
        "回购",
        "上调",
        "创新高",
        "放缓",
        "dovish",
        "cut rates",
        "stimulus",
    )
    pos = 0
    neg = 0
    for ev in sorted(events, key=lambda e: getattr(e, "severity", 0), reverse=True)[:4]:
        text = f"{getattr(ev, 'title', '')} {getattr(ev, 'summary', '')}".lower()
        severity = float(getattr(ev, "severity", 0) or 0)
        if any(w.lower() in text for w in negative_words):
            neg += 1
        elif severity >= 0.9 and any(w.lower() in text for w in positive_words):
            pos += 1

    score = max(-2, min(2, pos - neg))
    if pos and neg:
        return score, ["高优先级宏观新闻多空混杂"]
    if neg:
        return score, ["高优先级宏观新闻偏风险"]
    if pos:
        return score, ["高优先级宏观新闻偏利好"]
    return 0, []


def _unique_texts(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _action_risk_profile(
    mkt: Dict[str, dict], event_result: Any
) -> tuple[str, str, List[str]]:
    score = 0
    reasons: List[str] = []

    vix = _mkt_price(mkt, "^VIX")
    if vix is not None:
        if vix >= 30:
            score += 3
            reasons.append(f"VIX {vix:.1f} 极端恐慌")
        elif vix >= 20:
            score += 2
            reasons.append(f"VIX {vix:.1f} 波动升温")
        elif vix >= 15:
            score += 1
            reasons.append(f"VIX {vix:.1f} 正常偏紧")

    es_pct = _mkt_pct(mkt, "ES=F")
    if es_pct is not None:
        if es_pct <= -0.7:
            score += 2
            reasons.append(f"S&P 期货低开 {es_pct:.2f}%")
        elif es_pct >= 0.8:
            score += 1
            reasons.append(f"S&P 期货高开 {es_pct:.2f}%")
        elif abs(es_pct) >= 0.4:
            score += 1
            reasons.append(f"S&P 期货盘前异动 {es_pct:+.2f}%")

    hyg_pct = _mkt_pct(mkt, "HYG")
    if hyg_pct is not None and hyg_pct <= -0.5:
        score += 1
        reasons.append(f"HYG 走弱 {hyg_pct:.2f}%")

    tnx_pct = _mkt_pct(mkt, "^TNX")
    if tnx_pct is not None and tnx_pct >= 3:
        score += 1
        reasons.append(f"10Y 美债收益率急升 {tnx_pct:.2f}%")

    today_events = _today_events(event_result)
    if any(e.impact == "critical" for e in today_events):
        score += 2
        reasons.append("今天有极高影响事件")
    elif any(e.impact == "high" for e in today_events):
        score += 1
        reasons.append("今天有高影响事件")

    if _event_source_has_issue(event_result):
        score += 1
        reasons.append("经济日历源不可用")

    if score >= 5:
        return "🔴 高风险，轻仓观望优先", "0-30%，只做确认度最高的计划", reasons
    if score >= 3:
        return "🟠 中高风险，小仓试错", "30-50%，分批入场，不加速追单", reasons
    if score >= 1:
        return "🟡 中性可交易", "50%以内，等待开盘确认后执行", reasons
    return "🟢 风险偏低，按计划交易", "不超过 70%，仍保留现金缓冲", reasons


def _action_open_strategy(mkt: Dict[str, dict]) -> str:
    es_pct = _mkt_pct(mkt, "ES=F")
    if es_pct is None:
        return "盘前期货不可用，先等开盘 5-15 分钟形成方向。"
    if es_pct >= 0.5:
        return "高开不追，等回踩或突破后第二次确认。"
    if es_pct <= -0.5:
        return "低开先看 SPY/QQQ 支撑，跌幅收敛再考虑计划。"
    return "平开/小幅波动，按计划等待入场价，不提前抢跑。"


def _action_event_risk(event_result: Any) -> str:
    today_events = sorted(_today_events(event_result), key=_event_priority_key)
    if today_events:
        labels = []
        for e in today_events[:3]:
            sym = f" {e.symbol}" if e.symbol else ""
            labels.append(f"{_format_event_time(e)} {e.title_cn}{sym}")
        suffix = f"；另有 {len(today_events) - 3} 项" if len(today_events) > 3 else ""
        return "今天有事件：" + "；".join(labels) + suffix
    if _event_source_has_issue(event_result):
        return "⚠️ 事件源不可用，人工核对 CPI/FOMC/非农/财报；不要把空结果当作无事件。"
    return "今天暂无已确认高影响事件；开盘前仍快速核对经济日历。"


def _action_watchlist(
    symbols: List[str], mkt: Dict[str, dict], movers: List[dict]
) -> str:
    positive = sorted(
        [m for m in movers if m.get("pct", 0) >= 0.5],
        key=lambda m: m.get("pct", 0),
        reverse=True,
    )
    if positive:
        return (
            "、".join(_fmt_mover(m) for m in positive[:5]) + "（只观察确认，不直接追）"
        )

    sectors = _top_positive_sectors(mkt)
    if sectors:
        return "强势板块：" + "、".join(sectors[:3]) + "；自选股等待开盘确认。"

    if symbols:
        return "、".join(symbols[:5]) + " 等待盘中突破/回踩信号。"
    return "暂无自选股，先观察 SPY/QQQ 方向。"


def _action_caution_list(
    event_result: Any, movers: List[dict], risk_reasons: List[str]
) -> str:
    cautions: List[str] = []

    negative = sorted(
        [m for m in movers if m.get("pct", 0) <= -0.5],
        key=lambda m: m.get("pct", 0),
    )
    cautions.extend(_fmt_mover(m) for m in negative[:4])

    today_event_symbols = [
        e.symbol for e in _today_events(event_result) if getattr(e, "symbol", None)
    ]
    if today_event_symbols:
        cautions.append("事件窗口：" + "、".join(today_event_symbols[:4]))

    if _event_source_has_issue(event_result):
        cautions.append("事件源未确认，避免事件前后盲目开仓")

    if any("VIX" in r or "低开" in r for r in risk_reasons):
        cautions.append("高波动日避免追高和摊低")

    return (
        "；".join(cautions[:5]) if cautions else "暂无明显回避标的，仍按止损纪律执行。"
    )


def _today_events(event_result: Any) -> List[Any]:
    today = _today_pacific_date()
    events = getattr(event_result, "events", []) or []
    return [
        e for e in _dedupe_calendar_events(events) if _event_pacific_date(e) == today
    ]


def _dedupe_calendar_events(events: List[Any]) -> List[Any]:
    seen = set()
    out = []
    for event in sorted(events, key=_event_sort_key):
        key = _event_identity(event)
        if key in seen:
            continue
        seen.add(key)
        out.append(event)
    return out


def _event_identity(event: Any) -> tuple:
    if getattr(event, "category", "") == "fomc":
        return (
            getattr(event, "date", ""),
            getattr(event, "time_str", ""),
            "fomc",
            getattr(event, "symbol", "") or "",
        )
    title = str(getattr(event, "title_cn", "") or getattr(event, "title_en", ""))
    title = title.replace("🏦", "").replace("🎤", "").replace("📊", "").strip().lower()
    return (
        getattr(event, "date", ""),
        getattr(event, "time_str", ""),
        title,
        getattr(event, "symbol", "") or "",
    )


def _event_sort_key(event: Any) -> tuple:
    impact_order = {"critical": 0, "high": 1, "medium": 2}
    return (
        getattr(event, "date", ""),
        _event_time_minutes(getattr(event, "time_str", "")),
        impact_order.get(getattr(event, "impact", ""), 9),
        getattr(event, "title_cn", ""),
    )


def _event_priority_key(event: Any) -> tuple:
    impact_order = {"critical": 0, "high": 1, "medium": 2}
    return (
        impact_order.get(getattr(event, "impact", ""), 9),
        _event_time_minutes(getattr(event, "time_str", "")),
        getattr(event, "title_cn", ""),
    )


def _event_time_minutes(time_str: str) -> int:
    text = str(time_str or "")
    if "All Day" in text or "全天" in text:
        return 0
    if "盘前" in text or "BMO" in text.upper():
        return 8 * 60
    if "盘后" in text or "AMC" in text.upper():
        return 16 * 60
    try:
        hh, mm = text[:5].split(":")
        return int(hh) * 60 + int(mm)
    except Exception:
        return 24 * 60


def _event_pacific_date(event: Any):
    date_str = getattr(event, "date", "")
    time_str = str(getattr(event, "time_str", "") or "")
    try:
        if "ET" in time_str and len(time_str) >= 5:
            hh, mm = time_str[:5].split(":")
            dt_et = datetime.strptime(date_str, "%Y-%m-%d").replace(
                hour=int(hh),
                minute=int(mm),
                tzinfo=_EASTERN_TZ,
            )
            return dt_et.astimezone(_PACIFIC_TZ).date()
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except Exception:
        return _today_pacific_date()


def _format_event_time(event: Any) -> str:
    time_str = str(getattr(event, "time_str", "") or "")
    date_str = getattr(event, "date", "")
    if "ET" not in time_str or len(time_str) < 5:
        return time_str
    try:
        hh, mm = time_str[:5].split(":")
        dt_et = datetime.strptime(date_str, "%Y-%m-%d").replace(
            hour=int(hh),
            minute=int(mm),
            tzinfo=_EASTERN_TZ,
        )
        pt = dt_et.astimezone(_PACIFIC_TZ).strftime("%H:%M PT")
        return f"{pt} / {time_str}"
    except Exception:
        return time_str


def _event_source_has_issue(event_result: Any) -> bool:
    return bool(getattr(event_result, "has_source_issue", False))


def _event_source_warning(event_result: Any) -> str:
    warning = getattr(event_result, "warning_text", "")
    if warning:
        return warning
    return "事件源不可用"


def _mkt_price(mkt: Dict[str, dict], ticker: str) -> Optional[float]:
    try:
        return float(mkt[ticker]["price"])
    except Exception:
        return None


def _mkt_pct(mkt: Dict[str, dict], ticker: str) -> Optional[float]:
    try:
        return float(mkt[ticker]["pct"])
    except Exception:
        return None


def _top_positive_sectors(mkt: Dict[str, dict]) -> List[str]:
    rows = []
    for tk, name in _SECTOR_MAP:
        pct = _mkt_pct(mkt, tk)
        if pct is not None and pct > 0:
            rows.append((name, pct))
    rows.sort(key=lambda x: x[1], reverse=True)
    return [f"{name} {pct:+.2f}%" for name, pct in rows]


def _fmt_mover(m: dict) -> str:
    sym = str(m.get("symbol", "")).upper()
    pct = float(m.get("pct", 0))
    return f"{sym} {pct:+.2f}%"


def _fetch_yahoo_chart_snapshot(symbols: List[str]) -> Dict[str, dict]:
    data: Dict[str, dict] = {}
    for symbol in symbols:
        quote = _fetch_yahoo_chart_quote(symbol)
        if quote:
            data[symbol] = quote
    return data


def _fetch_yahoo_chart_quote(symbol: str) -> Optional[dict]:
    try:
        encoded = urllib.parse.quote(symbol, safe="")
        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded}"
            "?interval=1m&range=2d&includePrePost=true"
        )
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; morning-brief/1.0)",
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            payload = json.loads(resp.read())
        result = (payload.get("chart", {}).get("result") or [None])[0]
        if not result:
            return None
        meta = result.get("meta", {})
        quote = (result.get("indicators", {}).get("quote") or [{}])[0]
        timestamps = result.get("timestamp") or []
        closes = quote.get("close") or []
        highs = quote.get("high") or []
        lows = quote.get("low") or []
        opens = quote.get("open") or []
        volumes = quote.get("volume") or []
        period = meta.get("currentTradingPeriod", {}) if isinstance(meta, dict) else {}
        last = _last_number(closes) or meta.get("regularMarketPrice")
        prev = (
            meta.get("chartPreviousClose")
            or meta.get("previousClose")
            or meta.get("regularMarketPreviousClose")
        )
        if not last or not prev:
            return None
        last = float(last)
        prev = float(prev)
        pct = (last - prev) / prev * 100 if prev else 0.0
        pre_start = _period_timestamp(period, "pre", "start")
        regular_start = _period_timestamp(period, "regular", "start")
        pm_high = _session_extreme(timestamps, highs, pre_start, regular_start, max)
        pm_low = _session_extreme(timestamps, lows, pre_start, regular_start, min)
        pm_volume = _session_volume(timestamps, volumes, pre_start, regular_start)
        prior_high = _session_extreme(timestamps, highs, None, pre_start, max)
        prior_low = _session_extreme(timestamps, lows, None, pre_start, min)
        return {
            "price": last,
            "prev": prev,
            "pct": pct,
            "high": pm_high or _last_number(highs) or last,
            "low": pm_low or _last_number(lows) or last,
            "open": _last_number(opens) or last,
            "pm_high": pm_high,
            "pm_low": pm_low,
            "pm_volume": pm_volume,
            "prior_high": prior_high,
            "prior_low": prior_low,
            "gap_pct": pct,
            "source": "yahoo_chart",
            "as_of": _last_timestamp_iso(timestamps, closes),
        }
    except Exception as exc:
        logger.debug("Yahoo chart %s 获取失败: %s", symbol, exc)
        return None


def _last_number(values: List[Any]) -> Optional[float]:
    for value in reversed(values or []):
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            continue
    return None


def _last_timestamp_iso(timestamps: List[Any], values: List[Any]) -> str:
    for ts, value in reversed(list(zip(timestamps or [], values or []))):
        if value is None:
            continue
        try:
            return (
                datetime.fromtimestamp(float(ts), tz=timezone.utc)
                .astimezone(_PACIFIC_TZ)
                .strftime("%H:%M PT")
            )
        except Exception:
            return ""
    return ""


def _period_timestamp(period: dict, name: str, field: str) -> Optional[float]:
    try:
        value = period.get(name, {}).get(field)
        return float(value) if value is not None else None
    except Exception:
        return None


def _session_extreme(
    timestamps: List[Any],
    values: List[Any],
    start: Optional[float],
    end: Optional[float],
    fn: Any,
) -> Optional[float]:
    selected = []
    for ts, value in zip(timestamps or [], values or []):
        if value is None:
            continue
        try:
            ts_f = float(ts)
            if start is not None and ts_f < start:
                continue
            if end is not None and ts_f >= end:
                continue
            selected.append(float(value))
        except Exception:
            continue
    return fn(selected) if selected else None


def _session_volume(
    timestamps: List[Any],
    values: List[Any],
    start: Optional[float],
    end: Optional[float],
) -> Optional[int]:
    total = 0
    found = False
    for ts, value in zip(timestamps or [], values or []):
        if value is None:
            continue
        try:
            ts_f = float(ts)
            if start is not None and ts_f < start:
                continue
            if end is not None and ts_f >= end:
                continue
            total += int(float(value))
            found = True
        except Exception:
            continue
    return total if found else None


# ═══════════════════════════════════════════════════════════════════════════
# 数据拉取
# ═══════════════════════════════════════════════════════════════════════════


def _fetch_batch() -> Dict[str, dict]:
    """一次 yfinance 批量下载，返回 {ticker: {price, prev, pct, high, low, open}} 。"""
    try:
        import yfinance as yf

        raw = yf.download(
            _ALL_BATCH,
            period="5d",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        close = raw["Close"]
        high = raw.get("High", raw["Close"])
        low = raw.get("Low", raw["Close"])
        open_ = raw.get("Open", raw["Close"])

        result: Dict[str, dict] = {}
        for tk in _ALL_BATCH:
            try:
                c = close[tk].dropna()
                if len(c) < 2:
                    continue
                prev = float(c.iloc[-2])
                last = float(c.iloc[-1])
                pct = (last - prev) / prev * 100 if prev else 0.0
                result[tk] = {
                    "price": last,
                    "prev": prev,
                    "pct": pct,
                    "high": float(high[tk].iloc[-1]) if tk in high else last,
                    "low": float(low[tk].iloc[-1]) if tk in low else last,
                    "open": float(open_[tk].iloc[-1]) if tk in open_ else last,
                    "source": "yfinance",
                    "as_of": _now_pacific().strftime("%H:%M PT"),
                }
            except Exception:
                pass
        return result
    except Exception as exc:
        logger.warning("批量行情拉取失败: %s", exc)
        return _fetch_yahoo_chart_snapshot(_ALL_BATCH)


# ═══════════════════════════════════════════════════════════════════════════
# 消息 1 各节
# ═══════════════════════════════════════════════════════════════════════════


def _build_regime_section(live_vix: Optional[float] = None) -> Tuple[str, int]:
    """`live_vix`（可选）：同一次晨报里已经实时拉到的 VIX（_fetch_batch 的
    ^VIX），用来跟这里读的市场环境缓存（手动刷新才更新，可能是几天前的）
    做一次核对——避免同一份简报里出现两个互相矛盾又没有互相说明的 VIX 数字。
    """
    try:
        from .teams.market_env import read_regime_cache

        cached = read_regime_cache()
    except Exception:
        cached = None

    if cached is None:
        return (
            "**📊 市场环境**\n"
            "暂无缓存 — 请到 UI「总览」刷新市场环境\n"
            "今天先按中性处理：控制仓位，等候方向。",
            0x8B949E,
        )

    age_h = (datetime.now(timezone.utc) - cached.as_of).total_seconds() / 3600
    if age_h > 20:
        stale = f"（缓存已 {age_h:.1f} 小时未刷新，建议去 UI「总览」重新刷新）"
    else:
        stale = ""

    vix = cached.vix
    spy = cached.spy_vs_200ma_pct

    if vix is None:
        vix_line = "VIX 数据不可用"
    elif vix < 15:
        vix_line = f"VIX {vix:.1f} — 市场极度平静"
    elif vix < 20:
        vix_line = f"VIX {vix:.1f} — 波动正常"
    elif vix < 25:
        vix_line = f"VIX {vix:.1f} — 开始紧张，留意风险"
    elif vix < 35:
        vix_line = f"VIX {vix:.1f} ⚠️ — 市场在恐慌，价格波动大"
    else:
        vix_line = f"VIX {vix:.1f} 🚨 — 极度恐慌，建议场外观望"

    if (
        live_vix is not None
        and vix is not None
        and abs(live_vix - vix) >= 1.0
    ):
        vix_line += f"（同一份简报里的实时 VIX 是 {live_vix:.1f}，缓存已过期，以实时为准）"

    if spy is None:
        spy_line = "大盘趋势数据不可用"
    elif spy > 5:
        spy_line = f"SPY 在 200MA 上方 +{spy:.1f}%  ▶ 大趋势强势"
    elif spy > 0:
        spy_line = f"SPY 在 200MA 上方 +{spy:.1f}%  ▶ 趋势偏多"
    elif spy > -5:
        spy_line = f"SPY 在 200MA 下方 {spy:.1f}%  ▶ 趋势偏弱"
    else:
        spy_line = f"SPY 在 200MA 下方 {spy:.1f}%  ▶ 空头趋势"

    from .teams.base import RegimeType

    r = cached.regime
    if r == RegimeType.BULL_TREND:
        label = "趋势多头 🟢"
        stance = "可以正常操作，顺趋势做多，严格执行止损。"
        color = 0x2ECC71
    elif r == RegimeType.BEAR_TREND:
        label = "趋势空头 🔴"
        stance = "减少多单，等信号分数 ≥ 70 再进场。考虑防御性板块。"
        color = 0xE74C3C
    elif r == RegimeType.HIGH_VOL:
        label = "高波动 ⚠️"
        stance = "观望为主，不追涨。若已持仓，止损放宽 1.5× 防止被刷掉。"
        color = 0xD29922
    else:
        label = "震荡中性 ➡️"
        stance = "轻仓等待方向，做好双向准备。"
        color = 0x8B949E

    text = (
        f"**📊 市场环境** {stale}\n"
        f"**{label}**\n"
        f"{spy_line}\n"
        f"{vix_line}\n\n"
        f"💡 今天建议：{stance}"
    )
    return text, color


def _build_market_overview(mkt: dict) -> str:
    """合并指数昨收 + 对应期货方向，避免重复显示同一标的。

    期货那一栏的涨跌幅永远是 yfinance "1d" 整根 K 线的涨跌（不区分盘前/
    盘中/盘后），过去无脑贴"盘前"标签——如果晨报是收盘后手动重发的，这
    根本不是盘前数据，是当天已经走完的全天涨跌。改成按真实市场时段动态
    加标签。
    """
    from .market_calendar import session_now

    session = session_now()
    fut_label = {"pre": "盘前", "open": "盘中", "post": "盘后", "closed": "最新"}[session]

    lines = ["**📈 大盘昨收 & 期货方向**"]
    if session != "pre":
        lines.append(
            f"_（当前非盘前时段 [{fut_label}]，下面期货涨跌是最新整根 K 线涨跌，"
            "不是纯盘前变动）_"
        )
    any_data = False
    for etf, name in _INDEX_MAP:
        d_etf = mkt.get(etf)
        if not d_etf:
            continue
        any_data = True
        dot = "🟢" if d_etf["pct"] >= 0 else "🔴"
        arrow = "▲" if d_etf["pct"] >= 0 else "▼"
        line = (
            f"{dot} {name:<20} ${d_etf['price']:>7,.0f}  "
            f"{arrow}{abs(d_etf['pct']):.2f}%（昨收）"
        )
        fut_tk = _ETF_TO_FUTURES.get(etf)
        d_fut = mkt.get(fut_tk) if fut_tk else None
        if d_fut:
            fa = "▲" if d_fut["pct"] >= 0 else "▼"
            line += f"  │  {fut_label} {fa}{abs(d_fut['pct']):.2f}%"
        lines.append(line)

    es = mkt.get("ES=F")
    if es:
        if es["pct"] > 0.5:
            lines.append("→ 期货高开信号，追高需谨慎，等开盘稳定后再进。")
        elif es["pct"] < -0.5:
            lines.append("→ 期货低开信号，开盘后关注支撑能否守住再操作。")
    return "\n".join(lines) if any_data else ""


def _is_notable(tk: str, price: float, pct: float, mkt: dict) -> tuple[bool, str]:
    """
    返回 (值得显示, 解读文字)。
    正常无异动 → False，智能过滤掉；有信号 → True + 原因。
    """
    ap = abs(pct)
    if tk == "^TNX":
        if price >= 4.8:
            return True, "← 美债高位，成长股/估值承压"
        if price <= 3.5:
            return True, "← 美债低位，避险情绪或降息预期"
        if pct >= 3:
            return True, "← 收益率大涨，科技/成长股承压"
        if pct <= -3:
            return True, "← 收益率大跌，利好科技/成长"
        return False, ""
    if tk == "^VIX":
        if price >= 30:
            return True, "← 极端恐慌，历史上接近阶段底部"
        if price >= 20:
            return True, "← 恐慌升温，关注波动放大风险"
        if pct >= 15:
            return True, "← VIX 急升，市场突发风险情绪"
        return False, ""
    if tk == "TLT":
        if pct >= 1.2:
            return True, "← 避险资金涌入债券，风险偏好下降"
        if pct <= -1.2:
            return True, "← 债券抛售，利率压力上升"
        return False, ""
    if tk == "HYG":
        if pct <= -0.5:
            return True, "← 信用利差扩大，风险情绪降温"
        if pct >= 0.5:
            return True, "← 信用利差收窄，风险偏好改善"
        return False, ""
    if tk == "UUP":
        if pct >= 0.5:
            return True, "← 美元走强，大宗商品/新兴市场承压"
        if pct <= -0.5:
            return True, "← 美元走弱，利好大宗商品"
        return False, ""
    if tk == "GC=F":
        if ap >= 1.2:
            return (
                True,
                f"← 黄金{'大涨，避险情绪升温' if pct > 0 else '下跌，风险偏好改善'}",
            )
        return False, ""
    if tk == "CL=F":
        if ap >= 2.5:
            return (
                True,
                f"← 油价{'大涨，通胀压力+能源板块受益' if pct > 0 else '大跌，经济需求预期走弱'}",
            )
        return False, ""
    if tk == "HG=F":
        if ap >= 2.0:
            return (
                True,
                f"← 铜价{'上涨，全球经济需求预期改善' if pct > 0 else '下跌，经济活动走弱信号'}",
            )
        return False, ""
    if tk == "BTC-USD":
        if ap >= 5.0:
            return (
                True,
                f"← BTC {'大涨，风险偏好上升' if pct > 0 else '大跌，留意风险资产联动'}",
            )
        return False, ""
    if tk in ("EWJ", "FXI", "VGK"):
        if ap >= 1.2:
            return True, f"← 海外市场{'走强' if pct > 0 else '走弱'}，关注隔夜风险传导"
        return False, ""
    return False, ""


def _build_macro_section(mkt: dict) -> str:
    """
    宏观指标——只报有信号的项目。
    指标在正常范围且无明显异动时跳过，避免每天都是同一堆数字。
    """
    lines: list[str] = []
    _SKIP = {"_curve_only", "_vix9d_only"}

    for tk, name, unit, _ in _MACRO_MAP:
        if name in _SKIP:
            continue
        d = mkt.get(tk)
        if not d:
            continue
        price, pct = d["price"], d["pct"]
        notable, interp = _is_notable(tk, price, pct, mkt)
        if not notable:
            continue

        # 格式化价格
        if tk == "^TNX":
            val_str = f"{price:.2f}%"
        elif tk == "^VIX":
            val_str = f"{price:.1f}"
            # 折叠 VIX9D：显示近期溢价
            vix9d_d = mkt.get("^VIX9D")
            if vix9d_d:
                ratio = vix9d_d["price"] / price
                if ratio > 1.1:
                    interp += (
                        f"  VIX9D 溢价 +{(ratio - 1) * 100:.0f}%（近期恐慌高于中期）"
                    )
        elif tk in ("GC=F", "CL=F", "HG=F"):
            val_str = f"${price:,.1f}{unit}"
        elif tk == "BTC-USD":
            val_str = f"${price:,.0f}"
        else:
            val_str = f"${price:.2f}"

        arrow = "↑" if pct >= 0 else "↓"
        lines.append(f"  {name:<20} {val_str}  {arrow}{abs(pct):.2f}%  {interp}")

    # ── 收益率曲线（仅倒挂或趋平时显示） ─────────────────────────────────
    tnx = mkt.get("^TNX")
    irx = mkt.get("^IRX")
    if tnx and irx:
        spread = tnx["price"] - irx["price"]
        if spread < 0:
            lines.append(f"⚠️ 收益率曲线倒挂  10Y-3M={spread:+.2f}%  历史衰退预警信号")
        elif spread < 0.4:
            lines.append(f"→ 收益率曲线趋平  10Y-3M={spread:+.2f}%  关注风险")
        # spread >= 0.4 属于正常，不显示

    if not lines:
        return ""
    return "**💰 宏观异动**\n" + "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# 消息 2 各节
# ═══════════════════════════════════════════════════════════════════════════


def _build_sector_section(mkt: dict) -> str:
    rows = []
    for tk, name in _SECTOR_MAP:
        d = mkt.get(tk)
        if d:
            rows.append((name, d["pct"]))
    if not rows:
        return ""
    rows.sort(key=lambda x: x[1], reverse=True)

    # 全部 11 个板块用 bar 显示
    lines = ["**🗂 板块轮动（昨日全部 11 板块）**"]
    for name, pct in rows:
        dot = "🟢" if pct >= 0 else "🔴"
        bar = "█" * min(int(abs(pct) * 3), 8)
        lines.append(f"{dot} {name:<8} {pct:+.2f}%  {bar}")

    # 快速解读
    top_names = [n for n, _ in rows[:3]]
    best_pct = rows[0][1]
    worst_pct = rows[-1][1]
    offense = {"科技", "通信", "非必需消费", "金融"}
    defense = {"公用事业", "必需消费", "医疗", "房地产"}
    top_set = set(top_names)
    if best_pct <= 0:
        interp = (
            f"11 个板块全线收跌，最弱 {rows[-1][0]} {worst_pct:+.2f}% → 风险偏好不足。"
        )
    elif top_set & offense:
        interp = "进攻型板块领涨 → 风险偏好高，可积极操作。"
    elif top_set & defense:
        interp = "防御型板块领涨 → 市场在避险，谨慎为主。"
    else:
        interp = "板块轮动混乱，没有明确方向。"
    lines.append(f"→ {interp}")

    return "\n".join(lines)


def _fetch_index_technicals() -> Dict[str, dict]:
    try:
        import yfinance as yf
    except Exception as exc:
        logger.debug("技术面 yfinance 导入失败: %s", exc)
        return {}

    snapshot: Dict[str, dict] = {}
    for name in ("SPY", "QQQ"):
        try:
            df = yf.Ticker(name).history(period="1y", interval="1d", auto_adjust=True)
            row = _calc_index_technical(df, name)
            if row:
                snapshot[name] = row
        except Exception as exc:
            logger.debug("%s 技术面获取失败: %s", name, exc)
    return snapshot


def _calc_index_technical(df: Any, name: str) -> Optional[dict]:
    if df is None or getattr(df, "empty", True) or len(df) < 50:
        return None
    c = df["Close"].dropna()
    h = df["High"].dropna()
    low_series = df["Low"].dropna()
    if len(c) < 50:
        return None

    close = float(c.iloc[-1])
    prev = float(c.iloc[-2])
    ma20 = float(c.tail(20).mean())
    ma50 = float(c.tail(50).mean())
    ma200 = float(c.tail(200).mean()) if len(c) >= 200 else ma50
    high = float(h.iloc[-1])
    low = float(low_series.iloc[-1])
    chg = (close - prev) / prev * 100 if prev else 0.0
    rsi14 = _calc_rsi(c, period=14)
    atr14_pct = _calc_atr_pct(df, close, period=14)

    if close >= ma20 >= ma50:
        trend = "多头排列"
    elif close < ma20 and close < ma50:
        trend = "短线转弱"
    else:
        trend = "震荡"

    return {
        "name": name,
        "close": close,
        "prev": prev,
        "chg": chg,
        "ma20": ma20,
        "ma50": ma50,
        "ma200": ma200,
        "vs20": (close - ma20) / ma20 * 100 if ma20 else 0.0,
        "vs50": (close - ma50) / ma50 * 100 if ma50 else 0.0,
        "vs200": (close - ma200) / ma200 * 100 if ma200 else 0.0,
        "rsi14": rsi14,
        "atr14_pct": atr14_pct,
        "support": low,
        "resistance": high,
        "trend": trend,
    }


def _calc_rsi(close: Any, period: int = 14) -> float:
    delta = close.diff().dropna()
    if len(delta) < period:
        return 50.0
    recent = delta.tail(period)
    gain = float(recent.clip(lower=0).mean())
    loss = float((-recent.clip(upper=0)).mean())
    if loss == 0:
        return 100.0
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def _calc_atr_pct(df: Any, close: float, period: int = 14) -> float:
    try:
        high = df["High"]
        low = df["Low"]
        prev_close = df["Close"].shift(1)
        tr = (high - low).to_frame("hl")
        tr["hc"] = (high - prev_close).abs()
        tr["lc"] = (low - prev_close).abs()
        atr = float(tr.max(axis=1).dropna().tail(period).mean())
        return atr / close * 100 if close else 0.0
    except Exception:
        return 0.0


def _build_technical_indicator_section(technicals: Dict[str, dict]) -> str:
    if not technicals:
        return ""

    lines = ["**📐 关键技术指标（SPY & QQQ）**"]
    for name in ("SPY", "QQQ"):
        row = technicals.get(name)
        if not row:
            continue
        dot = "🟢" if row["chg"] >= 0 else "🔴"
        lines.append(
            f"{dot} **{name}** 昨收 ${row['close']:.2f}（{row['chg']:+.2f}%） "
            f"| 趋势：{row['trend']}"
        )
        lines.append(
            f"   20MA ${row['ma20']:.2f}（{row['vs20']:+.1f}%） · "
            f"50MA ${row['ma50']:.2f}（{row['vs50']:+.1f}%） · "
            f"200MA ${row['ma200']:.2f}（{row['vs200']:+.1f}%）"
        )
        lines.append(
            f"   RSI14 {row['rsi14']:.0f} · ATR14 {row['atr14_pct']:.1f}% · "
            f"昨低 ${row['support']:.2f} / 昨高 ${row['resistance']:.2f}"
        )

    return "\n".join(lines)


def _build_spy_technicals() -> str:
    return _build_technical_indicator_section(_fetch_index_technicals())


def _build_event_section(symbols: List[str], result: Any = None) -> str:
    try:
        from .calendar_events import IMPACT_EMOJI

        result = result or _get_calendar_events_result(symbols, days=5)
        events = _dedupe_calendar_events(getattr(result, "events", []) or [])
    except Exception as exc:
        logger.debug("重要事件获取失败: %s", exc)
        return (
            "**📅 重要事件**\n"
            "⚠️ 事件源不可用，需人工核对 CPI/FOMC/非农/财报；不要把空结果当作无事件。"
        )

    if not events:
        if _event_source_has_issue(result):
            return (
                "**📅 重要事件**\n"
                f"⚠️ 事件源不可用：{_event_source_warning(result)}\n"
                "请人工核对 CPI/FOMC/非农/财报；今天不要把“无事件”当作已确认。"
            )
        return "**📅 重要事件**\n未来 5 天未发现已知高影响力事件 ✅"

    now = _now_pacific()
    today_date = now.date()

    today_ev = [e for e in events if _event_pacific_date(e) == today_date]
    future_ev = [e for e in events if _event_pacific_date(e) > today_date]

    lines = ["**📅 重要事件**"]
    if _event_source_has_issue(result):
        lines.append(
            f"⚠️ 事件源覆盖不完整：{_event_source_warning(result)}；"
            "需人工核对 CPI/FOMC/非农/财报。"
        )

    if today_ev:
        lines.append("**🔴 今天：**")
        for e in today_ev:
            em = IMPACT_EMOJI.get(e.impact, "⚡")
            lines.append(f"{em} **{_format_event_time(e)}**  {e.title_cn}")
            if e.note:
                lines.append(f"  → {e.note}")
    else:
        lines.append("今天未发现已知高影响事件 ✅")

    if future_ev:
        lines.append("\n**近期（5 天内）：**")
        for e in future_ev[:8]:
            label = _rel_date(str(_event_pacific_date(e)), now)
            em = IMPACT_EMOJI.get(e.impact, "⚡")
            sym = f"（{e.symbol}）" if e.symbol else ""
            lines.append(f"{em} {label} {_format_event_time(e)}  {e.title_cn}{sym}")
            if e.note and e.impact == "critical":
                lines.append(f"  → {e.note}")

    if _event_source_has_issue(result):
        lines += [
            "",
            f"⚠️ 事件源状态：{_event_source_warning(result)}",
            "→ 经济日历未完全确认，开盘前请人工核对 CPI/FOMC/非农/财报。",
        ]

    return "\n".join(lines)


def _build_earnings_section(symbols: List[str]) -> str:
    """本周自选股财报日历（yfinance calendar）。"""
    symbols = _fundamentals_symbols(symbols)
    if not symbols:
        return ""
    try:
        import yfinance as yf
        from datetime import date as _date_cls
        from datetime import timedelta

        today = _today_pacific_date()
        week_end = today + timedelta(days=7)
        upcoming: list[dict] = []

        for sym in symbols[:15]:
            try:
                cal = yf.Ticker(sym).calendar
                if cal is None:
                    continue
                # yfinance 返回 dict {"Earnings Date": [Timestamp, ...], ...}
                dates_raw = (
                    cal.get("Earnings Date", []) if isinstance(cal, dict) else []
                )
                if not isinstance(dates_raw, list):
                    dates_raw = [dates_raw]
                for raw_dt in dates_raw:
                    try:
                        # yfinance returns plain datetime.date objects for
                        # calendar entries, not datetime.datetime/Timestamp —
                        # a date has no .date() method, so the old
                        # hasattr(raw_dt, "date") check was always False and
                        # this whole function always returned "" silently.
                        if isinstance(raw_dt, datetime):
                            dt = raw_dt.date()
                        elif isinstance(raw_dt, _date_cls):
                            dt = raw_dt
                        else:
                            dt = None
                        if dt and today <= dt <= week_end:
                            upcoming.append({"symbol": sym, "date": dt})
                            break
                    except Exception:
                        pass
            except Exception:
                pass

        if not upcoming:
            return ""

        upcoming.sort(key=lambda x: x["date"])
        lines = ["**💰 本周财报（自选股）**"]
        for ev in upcoming:
            diff = (ev["date"] - today).days
            label = {0: "**今天**", 1: "**明天**", 2: "**后天**"}.get(
                diff, f"**{ev['date'].strftime('%m/%d')}**"
            )
            lines.append(f"🔔 {label}  **{ev['symbol']}** 报告财报")
        return "\n".join(lines)
    except Exception as exc:
        logger.debug("财报日历获取失败: %s", exc)
        return ""


def _rel_date(date_str: str, now: datetime) -> str:
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=_PACIFIC_TZ)
        diff = (dt.date() - now.date()).days
        if diff == 1:
            return "**明天**"
        if diff == 2:
            return "**后天**"
        return f"**{date_str[5:]}**"
    except Exception:
        return date_str


def _fetch_priority_market_news(symbols: Optional[List[str]] = None) -> List[Any]:
    try:
        from .news import WallStreetCNSource

        since = datetime.now(timezone.utc) - timedelta(hours=14)
        # us 频道已废弃（code=20000 但 items 恒为空），去掉以免每次多打一个
        # 无效请求；global/forex/oil/gold 实测都正常返回。
        src = WallStreetCNSource(
            channels=["global", "forex", "oil", "gold"],
            num=40,
            min_score=2,
            timeout=8.0,
        )
        events = src.poll(since)
    except Exception as exc:
        logger.debug("WallStreetCN 获取失败: %s", exc)
        return []

    sym_set = {s.upper() for s in (symbols or [])}
    filtered = []
    for ev in events:
        text = f"{getattr(ev, 'symbol', '') or ''} {ev.title} {ev.summary}".upper()
        symbol_hit = any(sym in text for sym in sym_set)
        if ev.severity >= 0.9 or symbol_hit or _is_macro_news_relevant(ev):
            filtered.append(ev)

    filtered.sort(key=lambda e: getattr(e, "ts", datetime.min), reverse=True)
    return filtered[:10]


def _is_macro_news_relevant(ev: Any) -> bool:
    text = f"{getattr(ev, 'title', '')} {getattr(ev, 'summary', '')}".lower()
    keywords = (
        "fomc",
        "fed",
        "美联储",
        "鲍威尔",
        "利率",
        "降息",
        "加息",
        "cpi",
        "pce",
        "ppi",
        "非农",
        "失业",
        "初请",
        "gdp",
        "通胀",
        "关税",
        "制裁",
        "地缘",
        "战争",
        "停火",
        "美元",
        "美债",
        "收益率",
        "原油",
        "黄金",
        "opec",
    )
    return float(getattr(ev, "severity", 0) or 0) >= 0.6 and any(
        k in text for k in keywords
    )


def _build_news_section(
    symbols: Optional[List[str]] = None,
    events: Optional[List[Any]] = None,
) -> str:
    """
    国际宏观要闻 — 直接复用 trader/news.WallStreetCNSource（华尔街见闻全球快讯）。
    覆盖：全球要闻 · 外汇 · 原油 · 黄金，score≥2 重要以上，中文内容。
    （美股频道已被上游废弃，见 _fetch_priority_market_news 注释。）
    """
    if events is None:
        events = _fetch_priority_market_news(symbols)

    if not events:
        return ""

    filtered = sorted(
        events, key=lambda e: getattr(e, "ts", datetime.min), reverse=True
    )

    lines = ["**🌍 异常宏观新闻（近14h，最近在前）**"]
    for ev in filtered[:8]:
        ts = getattr(ev, "ts", None)
        time_s = _format_dt_pt(ts) if ts else "--:--"
        mark = "🔴" if ev.severity >= 0.9 else ("🟡" if ev.severity >= 0.6 else "•")
        text = (ev.summary or ev.title)[:85]
        lines.append(f"{mark} **{time_s}**  {text}")

    return "\n".join(lines)


def _format_dt_pt(ts: datetime) -> str:
    return ts.astimezone(_PACIFIC_TZ).strftime("%H:%M PT")


# CBOE Put/Call Ratio 已移除：两个公开 CSV 端点分别返回 403 / 404（CBOE 已
# 撤下免费下载），官方 market_statistics JSON 同样 403，只剩需要爬 HTML 的
# 页面，不值得为一个可选指标背这个脆弱性。情绪维度已由 VIX（_build_regime_
# section）和 CNN Fear & Greed（_build_fear_greed）覆盖。


def _build_fear_greed() -> str:
    """
    CNN Fear & Greed Index — 公开 API，无需 Key。
    仅在极端区间（<30 恐惧 / >70 贪婪）时输出，中性区间静默。
    """
    try:
        import urllib.request
        import json

        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        # CNN 的边缘防护要求 UA + Referer + Accept 三个头同时存在，缺任何
        # 一个都返回 418（实测：UA 单独、UA+Referer、UA+Accept 全是 418，
        # 三者齐备才 200）。原来只发 "Mozilla/5.0 (compatible)"，于是每天
        # 都在日志里刷 "Fear & Greed 获取失败: HTTP Error 418"，晨报这一
        # 节长期是空的。
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/126.0.0.0 Safari/537.36"
                ),
                "Referer": "https://www.cnn.com/markets/fear-and-greed",
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read())
        fg = data["fear_and_greed"]
        score = int(float(fg["score"]))
        prev = int(float(fg.get("previous_close", score)))

        # 中性区间（30-70）→ 不打扰，直接跳过
        if 30 <= score <= 70:
            return ""

        if score >= 85:
            emoji, label, note = (
                "😈",
                "极度贪婪",
                "情绪极度亢奋，追高风险极高，考虑减仓锁利。",
            )
        elif score > 70:
            emoji, label, note = "😏", "贪婪", "市场偏乐观，保持纪律、严守止损。"
        elif score < 15:
            emoji, label, note = (
                "😱",
                "极度恐惧",
                "极度恐慌，历史上往往接近阶段性底部。",
            )
        else:
            emoji, label, note = (
                "😨",
                "恐惧",
                "市场偏谨慎，等候做多信号，关注超卖机会。",
            )

        bar = "█" * min(score // 10, 10) + "░" * max(0, 10 - score // 10)
        trend = (
            f"↑(昨{prev})"
            if score > prev
            else (f"↓(昨{prev})" if score < prev else f"→(昨{prev})")
        )

        return (
            f"**😨 恐慌贪婪指数（Fear & Greed）**\n"
            f"`{bar}` **{score}** / 100  {emoji} {label}  {trend}\n"
            f"→ {note}"
        )
    except Exception as exc:
        logger.warning("Fear & Greed 获取失败: %s", exc)
        return ""


def _build_opex_section() -> str:
    """期权到期日（OpEx）提醒 — 纯日历逻辑，无需外部数据。"""
    from datetime import date, timedelta

    today = _today_pacific_date()

    def _third_friday(y: int, m: int) -> date:
        d = date(y, m, 1)
        return d + timedelta(days=(4 - d.weekday()) % 7) + timedelta(weeks=2)

    # 周度 OpEx：最近的下一个周五
    days_to_fri = (4 - today.weekday()) % 7 or 7
    next_fri = today + timedelta(days=days_to_fri)

    # 月度 OpEx：当月第三个周五
    y, m = today.year, today.month
    monthly = _third_friday(y, m)
    if monthly < today:
        m = m % 12 + 1
        y = y + (1 if m == 1 else 0)
        monthly = _third_friday(y, m)

    # 季度 OpEx：3/6/9/12 月第三个周五
    qm = ((m - 1) // 3 + 1) * 3
    qy = y + (1 if qm > 12 else 0)
    qm = qm if qm <= 12 else qm - 12
    quarterly = _third_friday(qy, qm)
    if quarterly < today:
        qm += 3
        if qm > 12:
            qm -= 12
            qy += 1
        quarterly = _third_friday(qy, qm)

    lines = ["**⏰ 期权到期（OpEx）**"]
    wd = (next_fri - today).days
    if wd == 0:
        lines.append(
            "🔔 **今天** 周度 OpEx（SPY/QQQ 周五到期，注意 pin risk 和 gamma 挤压）"
        )
    else:
        lines.append(f"周度：{next_fri.strftime('%m/%d')}（{wd}天后）")

    md = (monthly - today).days
    if md == 0:
        lines.append("🚨 **今天** 月度 OpEx！头寸强制平仓 → 日内波动可能异常放大")
    elif md <= 4:
        lines.append(f"⚠️ 月度：{monthly.strftime('%m/%d')}（{md}天后 · 临近）")
    else:
        lines.append(f"月度：{monthly.strftime('%m/%d')}（{md}天后）")

    qd = (quarterly - today).days
    if qd <= 7:
        lines.append(
            f"📌 季度 OpEx：{quarterly.strftime('%m/%d')}（{qd}天后 · 大型机构调仓窗口）"
        )

    return "\n".join(lines)


def _build_analyst_changes(symbols: List[str]) -> str:
    """近 48h 自选股分析师评级变动（yfinance upgrades_downgrades）。"""
    symbols = _fundamentals_symbols(symbols)
    if not symbols:
        return ""
    try:
        import yfinance as yf

        cutoff = datetime.now(timezone.utc) - timedelta(hours=48)
        changes = []
        for sym in symbols[:12]:
            try:
                df = yf.Ticker(sym).upgrades_downgrades
                if df is None or df.empty:
                    continue
                if df.index.tz is None:
                    df.index = df.index.tz_localize("UTC")
                recent = df[df.index >= cutoff]
                for _, row in recent.iterrows():
                    action = str(row.get("Action", "")).lower()
                    changes.append(
                        {
                            "symbol": sym,
                            "action": action,
                            "from": row.get("FromGrade", ""),
                            "to": row.get("ToGrade", ""),
                            "firm": row.get("Firm", ""),
                        }
                    )
            except Exception:
                pass

        if not changes:
            return ""

        # yfinance 的 Action 取值：up / down / main(维持) / reit(重申) / init(首次覆盖)。
        # 原来把 reit/init 也画成 ⬆️ 升级箭头，于是 "Buy → Buy" 这种"维持原评级"
        # 被摆在"评级变动"标题下面显示成利好——对交易者是实打实的误导。
        # 现在按真实含义分开标注，标题也改成如实描述。
        _ACTION_LABEL = {
            "up":   ("⬆️", "上调"),
            "down": ("⬇️", "下调"),
            "init": ("🆕", "首次覆盖"),
            "reit": ("🔁", "重申"),
            "main": ("➖", "维持"),
        }
        lines = ["**📊 分析师评级动态（近 48h）**"]
        for c in changes[:8]:
            action = str(c["action"]).lower()
            if action in _ACTION_LABEL:
                emoji, label = _ACTION_LABEL[action]
            elif "up" in action:
                emoji, label = "⬆️", "上调"
            elif "down" in action:
                emoji, label = "⬇️", "下调"
            else:
                emoji, label = "🔄", action or "变动"
            grade = f"{c['from']} → {c['to']}" if c["from"] and c["to"] else c["to"]
            lines.append(f"{emoji} **{c['symbol']}**  {label}：{grade}  （{c['firm']}）")
        return "\n".join(lines)
    except Exception as exc:
        logger.debug("分析师评级获取失败: %s", exc)
        return ""


def _select_balanced_company_news(
    events: List[Any],
    limit: int = 8,
    per_symbol: int = 2,
) -> List[Any]:
    ranked = sorted(
        events,
        key=lambda ev: (
            _company_news_score(ev),
            getattr(ev, "severity", 0),
            getattr(ev, "ts", datetime.min),
        ),
        reverse=True,
    )
    counts: dict[str, int] = {}
    selected = []
    for ev in ranked:
        sym = (getattr(ev, "symbol", "") or "").upper()
        if not sym:
            continue
        if counts.get(sym, 0) >= per_symbol:
            continue
        selected.append(ev)
        counts[sym] = counts.get(sym, 0) + 1
        if len(selected) >= limit:
            break
    selected.sort(key=lambda ev: getattr(ev, "ts", datetime.min), reverse=True)
    return selected


def _company_news_score(ev: Any) -> int:
    text = f"{getattr(ev, 'title', '')} {getattr(ev, 'summary', '')}".lower()
    important = (
        "earnings",
        "revenue",
        "profit",
        "eps",
        "guidance",
        "forecast",
        "upgrade",
        "downgrade",
        "price target",
        "initiates",
        "rating",
        "acquisition",
        "merger",
        "deal",
        "partnership",
        "contract",
        "approval",
        "fda",
        "antitrust",
        "investigation",
        "lawsuit",
        "sec",
        "doj",
        "recall",
        "launch",
        "buyback",
        "dividend",
        "split",
        "layoff",
        "ceo",
        "cfo",
        "财报",
        "营收",
        "利润",
        "指引",
        "上调",
        "下调",
        "评级",
        "目标价",
        "收购",
        "并购",
        "合作",
        "合同",
        "批准",
        "监管",
        "调查",
        "诉讼",
        "回购",
        "裁员",
    )
    quiet = (
        "why shares",
        "stock moves",
        "market update",
        "etf",
        "watch",
        "roundup",
        "盘前异动",
        "美股异动",
    )
    score = sum(1 for word in important if word in text)
    if any(word in text for word in quiet):
        score -= 1
    if float(getattr(ev, "severity", 0) or 0) >= 0.8:
        score += 1
    return score


def _company_news_tag(ev: Any) -> str:
    text = f"{getattr(ev, 'title', '')} {getattr(ev, 'summary', '')}".lower()
    tag_map = [
        (("earnings", "revenue", "eps", "财报", "营收"), "业绩"),
        (("guidance", "forecast", "指引"), "指引"),
        (("upgrade", "downgrade", "price target", "rating", "评级", "目标价"), "评级"),
        (("acquisition", "merger", "deal", "收购", "并购"), "交易"),
        (("fda", "approval", "批准"), "审批"),
        (("antitrust", "investigation", "lawsuit", "监管", "调查", "诉讼"), "监管"),
        (("buyback", "dividend", "回购"), "资本动作"),
    ]
    for words, tag in tag_map:
        if any(word in text for word in words):
            return tag
    return ""


def _format_news_age(ts: Any) -> str:
    if not ts:
        return ""
    try:
        age_h = (datetime.now(timezone.utc) - ts).total_seconds() / 3600
        if age_h < 1:
            return f"{max(1, int(age_h * 60))}m前"
        return f"{int(age_h)}h前"
    except Exception:
        return ""


def _build_stock_catalysts(symbols: List[str]) -> str:
    """各股关键动态：Finnhub 公司新闻（48h，不过滤关键词）。

    原本这里还有一段华尔街见闻 us-channel 的"按标的过滤"逻辑，已删除：该频道
    上游已废弃（返回 code=20000 但 items 恒为空），而且即便改抓 global 频道也
    没用——那边是中文宏观快讯，几乎不带美股 ticker，按 ticker 过滤永远匹配不
    到。这一节的用途（个股 48h 催化剂）由 Finnhub 公司新闻完整覆盖，它本身就
    是按 symbol 拉取的英文源。
    """
    symbols = _company_symbols(symbols)
    if not symbols:
        return ""

    lines = ["**📌 各股关键动态（48h）**"]
    any_content = False
    since = datetime.now(timezone.utc) - timedelta(hours=48)

    # ── Finnhub 公司新闻（不过滤关键词，需 FINNHUB_API_KEY）────────────────────
    try:
        from .news import FinnhubSource

        fh = FinnhubSource(universe=symbols, timeout=8.0)
        fh_ev = [ev for ev in fh.poll(since) if _company_news_score(ev) > 0]
        if fh_ev:
            any_content = True
            lines.append("**📋 Finnhub 公司新闻**")
            for ev in _select_balanced_company_news(fh_ev, limit=8, per_symbol=2):
                title = (ev.title or "")[:110]
                tag = _company_news_tag(ev)
                tag_s = f"{tag} · " if tag else ""
                lines.append(
                    f"• **{ev.symbol}**  {tag_s}{title}  _({_format_news_age(ev.ts)})_"
                )
    except Exception as exc:
        logger.debug("stock_catalysts finnhub: %s", exc)

    return "\n".join(lines) if any_content else ""


# r/WallStreetBets 板块已移除：Reddit 自 2023 年起封锁未授权的 .json 接口，
# hot.json 无论换什么 User-Agent 都返回 403（old.reddit 同样 403）。唯一还能
# 匿名访问的是 RSS，但 RSS 不带 upvote 分数，而原实现的"异常热度"判定完全
# 建立在 score>=1500 / 自选股相关 score>=500 的阈值上——没有分数就只剩"有人
# 提过"，噪音远大于信号。散户热度这一维度改由 _build_trending_section
# （Yahoo Finance 热搜，实测可用）承担。


def _build_trending_section(symbols: Optional[List[str]] = None) -> str:
    """Yahoo Finance 异常热搜 — 只显示自选股命中或日涨跌幅较大的标的。"""
    try:
        import json
        import urllib.request

        import yfinance as yf

        url = "https://query1.finance.yahoo.com/v1/finance/trending/US"
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0 (compatible)"}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        watch_set = {s.upper() for s in (symbols or [])}
        trending_symbols = [
            r["symbol"] for r in data["finance"]["result"][0]["quotes"][:10]
        ]

        raw = yf.download(
            trending_symbols,
            period="2d",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        close = raw.get("Close", raw)

        lines = ["**🔥 Yahoo Finance 异常热搜**"]
        for sym in trending_symbols[:8]:
            try:
                c = close[sym].dropna() if sym in close.columns else None
                if c is not None and len(c) >= 2:
                    last = float(c.iloc[-1])
                    prev = float(c.iloc[-2])
                    pct = (last - prev) / prev * 100
                    if sym.upper() not in watch_set and abs(pct) < 3.0:
                        continue
                    dot = "🟢" if pct >= 0 else "🔴"
                    sign = "+" if pct >= 0 else ""
                    lines.append(f"{dot} **{sym}** ${last:.2f}  {sign}{pct:.2f}%")
                elif sym.upper() in watch_set:
                    lines.append(f"• **{sym}**")
            except Exception:
                if sym.upper() in watch_set:
                    lines.append(f"• **{sym}**")
        return "\n".join(lines) if len(lines) > 1 else ""
    except Exception as exc:
        logger.debug("Yahoo 热搜获取失败: %s", exc)
        return ""


# ═══════════════════════════════════════════════════════════════════════════
# 消息 3 各节
# ═══════════════════════════════════════════════════════════════════════════


def _fetch_premarket_movers(symbols: List[str]) -> List[dict]:
    symbols = _company_symbols(symbols)
    if not symbols:
        return []
    movers: list[dict] = []
    seen: set[str] = set()

    chart_quotes = _fetch_yahoo_chart_snapshot(symbols[:12])
    for sym, quote in chart_quotes.items():
        movers.append(
            {
                "symbol": sym,
                "pre_price": quote["price"],
                "pct": quote["pct"],
                "prev": quote["prev"],
                "pm_high": quote.get("high"),
                "pm_low": quote.get("low"),
                "pm_open": quote.get("open"),
                "pm_volume": quote.get("pm_volume"),
                "prior_high": quote.get("prior_high"),
                "prior_low": quote.get("prior_low"),
                "gap_pct": quote.get("gap_pct", quote["pct"]),
                "as_of": quote.get("as_of", ""),
                "source": "Yahoo chart",
            }
        )
        seen.add(sym.upper())

    try:
        import yfinance as yf

        for sym in symbols[:12]:
            try:
                info = yf.Ticker(sym).info
                pre_price = info.get("preMarketPrice")
                prev_close = info.get("regularMarketPreviousClose") or info.get(
                    "previousClose"
                )
                reg_price = info.get("regularMarketPrice") or info.get("currentPrice")
                if sym.upper() in seen:
                    _attach_yfinance_crosscheck(movers, sym, pre_price or reg_price)
                    continue
                if pre_price and prev_close and prev_close > 0:
                    pct = (pre_price - prev_close) / prev_close * 100
                    movers.append(
                        {
                            "symbol": sym,
                            "pre_price": pre_price,
                            "pct": pct,
                            "prev": prev_close,
                            "as_of": _now_pacific().strftime("%H:%M PT"),
                            "source": "yfinance",
                        }
                    )
                elif reg_price and prev_close and prev_close > 0:
                    # 开盘后盘中数据
                    pct = (reg_price - prev_close) / prev_close * 100
                    movers.append(
                        {
                            "symbol": sym,
                            "pre_price": reg_price,
                            "pct": pct,
                            "prev": prev_close,
                            "as_of": _now_pacific().strftime("%H:%M PT"),
                            "source": "yfinance",
                        }
                    )
            except Exception:
                pass
    except ImportError:
        # Yahoo chart data is already usable; yfinance is only an optional
        # cross-check and must not discard the primary snapshot.
        pass

    return movers


def _attach_yfinance_crosscheck(
    movers: List[dict],
    symbol: str,
    yfinance_price: Optional[float],
) -> None:
    if not yfinance_price:
        return
    sym = symbol.upper()
    for mover in movers:
        if str(mover.get("symbol", "")).upper() != sym:
            continue
        try:
            yf_price = float(yfinance_price)
            base = float(mover.get("pre_price", 0))
            if base <= 0:
                return
            diff = (yf_price - base) / base * 100
            mover["yfinance_price"] = yf_price
            mover["price_discrepancy_pct"] = diff
            mover["source"] = f"{mover.get('source', 'Yahoo chart')} + yfinance check"
        except Exception:
            return


def _build_premarket_section(
    symbols: List[str], movers: Optional[List[dict]] = None
) -> str:
    if not symbols:
        return ""
    if movers is None:
        movers = _fetch_premarket_movers(symbols)

    significant = sorted(
        [m for m in movers if abs(m["pct"]) >= 0.5],
        key=lambda x: abs(x["pct"]),
        reverse=True,
    )
    flat = [m for m in movers if abs(m["pct"]) < 0.5]

    lines = ["**🔍 自选股盘前快照**"]

    if significant:
        lines.append("**有异动（≥ 0.5%）：**")
        for m in significant:
            dot = "🟢" if m["pct"] > 0 else "🔴"
            arrow = "▲" if m["pct"] > 0 else "▼"
            source = f" · {m.get('source')}" if m.get("source") else ""
            as_of = f" · {m.get('as_of')}" if m.get("as_of") else ""
            lines.append(
                f"{dot} **{m['symbol']:<5}** ${m['pre_price']:.2f}  "
                f"{arrow}{abs(m['pct']):.2f}%  （昨收 ${m['prev']:.2f}{source}{as_of}）"
            )
            quality = _format_premarket_quality(m)
            if quality:
                lines.append(f"   ↳ {quality}")
            level_text = _format_premarket_levels(m)
            if level_text:
                lines.append(f"   ↳ {level_text}")

    if flat:
        syms = "  ".join(m["symbol"] for m in flat)
        lines.append(f"无明显异动：{syms}")

    if len(movers) == 0:
        lines.append("盘前数据暂不可用（市场未开或 API 超时）")
    else:
        lines.append("_盘前/最新分钟价来自免费源，流动性低，仅供方向参考。_")

    # 复用 SECEdgarSource：自选股 8-K 重大事件申报（过去 24h）
    try:
        from .news import SECEdgarSource

        since_8k = datetime.now(timezone.utc) - timedelta(hours=24)
        edgar = SECEdgarSource(universe=symbols[:10], count=3, timeout=8.0)
        filings = edgar.poll(since_8k)
        if filings:
            lines.append("\n**📋 SEC 8-K 重大事件申报（过去 24h）**")
            for ev in filings[:5]:
                age_h = (datetime.now(timezone.utc) - ev.ts).total_seconds() / 3600
                lines.append(f"🔔 **{ev.symbol}**  {ev.title}  ({int(age_h)}h前)")
    except Exception:
        pass

    return "\n".join(lines)


def _format_premarket_levels(mover: dict) -> str:
    parts = []
    pm_high = _safe_float(mover.get("pm_high"))
    pm_low = _safe_float(mover.get("pm_low"))
    prior_high = _safe_float(mover.get("prior_high"))
    prior_low = _safe_float(mover.get("prior_low"))
    gap_pct = _safe_float(mover.get("gap_pct", mover.get("pct")))
    if gap_pct is not None:
        parts.append(f"gap {gap_pct:+.2f}%")
    if pm_high is not None and pm_low is not None:
        parts.append(f"盘前区间 {pm_low:.2f}-{pm_high:.2f}")
    if prior_high is not None and prior_low is not None:
        parts.append(f"昨区间 {prior_low:.2f}-{prior_high:.2f}")
    return "；".join(parts)


def _format_premarket_quality(mover: dict) -> str:
    parts = []
    pm_volume = mover.get("pm_volume")
    if pm_volume:
        parts.append(f"盘前量 {int(pm_volume):,}")
    discrepancy = _safe_float(mover.get("price_discrepancy_pct"))
    if discrepancy is not None:
        if abs(discrepancy) >= 0.3:
            parts.append(f"⚠️ Yahoo/yfinance 价差 {discrepancy:+.2f}%")
        else:
            parts.append(f"双源价差 {discrepancy:+.2f}%")
    return "；".join(parts)


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _build_carryover_section(db_path: str) -> str:
    """昨日承接 —— 读者醒来第一件事是"我手上还有什么"。

    晨报此前只报当前权益和持仓市值，但没说这些仓位的止损止盈挂在哪里，也没说
    昨天挂出去的计划还有几个活着。这两件事恰恰决定了开盘前要不要先处理点什
    么，而不是从零开始看今天的机会。
    """
    lines: List[str] = []

    try:
        from .signal_reports import SignalState, SignalStore

        store = SignalStore(db_path)
        active = store.active()
    except Exception as exc:  # noqa: BLE001
        logger.debug("承接信息读取失败: %s", exc)
        return ""

    holding = [
        r for r in active
        if getattr(r, "state", None) in (SignalState.ENTERED, SignalState.HOLD)
    ]
    waiting = [r for r in active if getattr(r, "state", None) == SignalState.READY]

    if holding:
        lines.append("**持仓的止损/止盈位**")
        for report in holding[:8]:
            lines.append(
                f"• {report.symbol}　止损 ${report.stop_loss:,.2f}"
                f"　止盈 ${report.take_profit:,.2f}"
            )
    if waiting:
        lines.append("**仍在等待的计划**")
        for report in waiting[:8]:
            valid = getattr(report, "valid_until", None)
            expiry = f"（有效至 {valid:%m/%d %H:%M UTC}）" if valid else ""
            lines.append(
                f"• {report.symbol}　入场区间 ${report.entry_low:,.2f}"
                f"–${report.entry_high:,.2f}{expiry}"
            )
    if not lines:
        return ""
    return "**🔄 昨日承接**\n" + "\n".join(lines)


def _build_automation_notice(auto_trade: Optional[bool] = None) -> str:
    """今天引擎会自动做什么。

    这套系统在 auto_trade_paper 打开时会自行下单。读者有权在开盘前就知道它今
    天打算怎么动，而不是等成交回报到了才发现——尤其当他自己也可能手动操作同
    一个账户时。

    auto_trade 由调用方传入（它在 TradingConfig 上，晨报模块本身拿不到）。传
    None 时退而读环境变量；连环境变量都没有就整节不显示——宁可不说，也不能显
    示一个可能与实际相反的自动化状态。
    """
    if auto_trade is None:
        raw = os.getenv("AUTO_TRADE_PAPER", "").strip().lower()
        if raw not in {"1", "true", "yes", "on", "0", "false", "no", "off"}:
            return ""
        auto_trade = raw in {"1", "true", "yes", "on"}
    if not auto_trade:
        return (
            "**🤖 今日自动化**\n"
            "自动交易已关闭，引擎只做监控与播报，不会自行下单。"
        )
    return (
        "**🤖 今日自动化**\n"
        "自动交易已开启（模拟盘）：引擎会在常规时段按计划自行挂单，"
        "并按各计划的止损/止盈自动离场。每一笔成交都会推送。"
    )


def _build_status_section(db_path: str, movers: Optional[List[dict]] = None) -> str:
    out = ["**📌 模拟交易账户（Alpaca Paper）**"]
    movers = movers or []

    try:
        from .monitor_data import live_alpaca_equity, live_alpaca_positions

        acc = live_alpaca_equity()
        equity = float(acc.get("equity", 0)) if acc else 0.0
        cash = float(acc.get("cash", 0)) if acc else 0.0
        bp = float(acc.get("buying_power", 0)) if acc else 0.0

        if equity <= 0:
            out.append("账户数据暂不可用（Alpaca API 未配置或网络超时）")
        else:
            cash_pct = cash / equity * 100
            out.append(
                f"💰 权益 **${equity:,.0f}**  ·  "
                f"现金 ${cash:,.0f}（{cash_pct:.1f}%）  ·  "
                f"购买力 ${bp:,.0f}"
            )

            positions = live_alpaca_positions()
            if positions:
                sorted_pos = sorted(
                    positions,
                    key=lambda x: abs(float(x.get("market_value", 0))),
                    reverse=True,
                )[:8]
                total_mv = sum(float(p.get("market_value", 0)) for p in positions)
                used_pct = total_mv / equity * 100
                free_pct = 100 - used_pct
                out.extend(
                    _build_account_risk_lines(
                        positions=positions,
                        equity=equity,
                        cash_pct=cash_pct,
                        movers=movers,
                    )
                )

                # 可视化仓位条（每格 = 5%，共 20 格）
                n_filled = min(int(used_pct / 5), 20)
                bar = "█" * n_filled + "░" * (20 - n_filled)
                out.append(
                    f"\n**持仓 {len(positions)} 只**  "
                    f"`{bar}`  {used_pct:.1f}% 已用 / {free_pct:.1f}% 空余"
                )

                # 代码块等宽表格：各列宽度固定，在 Discord 等宽字体下对齐
                rows = ["```"]
                for p in sorted_pos:
                    sym = p.get("symbol", "")
                    cur_pr = float(p.get("current_price", 0))
                    mv = float(p.get("market_value", 0))
                    pl = float(p.get("unrealized_pl", 0))
                    pl_pct = float(p.get("unrealized_plpc", 0)) * 100
                    wt = mv / equity * 100
                    sign = "+" if pl >= 0 else "-"

                    sym_s = f"{sym:<5}"
                    wt_s = f"{wt:>5.1f}%"
                    pct_s = f"{sign}{abs(pl_pct):.1f}%"
                    pr_s = f"${cur_pr:.2f}"
                    pl_s = f"{sign}${abs(pl):,.0f}"

                    rows.append(f"{sym_s} {wt_s}  {pct_s:>7}  {pr_s:>9}  {pl_s:>9}")
                rows.append("```")
                out.extend(rows)
            else:
                out.append("📊 空仓（现金 100%）")

    except Exception as exc:
        logger.debug("账户状态获取失败: %s", exc)
        out.append("⚠️ 账户数据获取失败（检查 Alpaca API Key），以上仓位/盈亏可能不是最新的。")

    return "\n".join(out)


def _build_account_risk_lines(
    positions: List[dict],
    equity: float,
    cash_pct: float,
    movers: List[dict],
) -> List[str]:
    from .account_risk import build_account_risk_lines

    return build_account_risk_lines(
        positions=positions,
        equity=equity,
        cash_pct=cash_pct,
        movers=movers,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 工具
# ═══════════════════════════════════════════════════════════════════════════


def _join(parts: List[str]) -> str:
    """把非空的段落用分隔线拼接。"""
    sep = "\n\n──────────────────────\n\n"
    return sep.join(p for p in parts if p.strip())


# ═══════════════════════════════════════════════════════════════════════════
# CLI 入口
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if sys.platform == "win32":
        # Windows 控制台默认编码（cp1252/GBK 等）打不出中文会直接 UnicodeEncodeError
        # 崩溃退出（main.py 之前踩过这个坑）。换新机器、用默认 cmd.exe 时最容易复现。
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # yfinance 对取不到数据的 ETF/指数（无 fundamentals/sector）会自行打 ERROR 日志，
    # 跟我们的 try/except 无关，单独调高级别屏蔽（不影响我们自己的报错处理）。
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    from dotenv import load_dotenv as _load

    _load(
        _ROOT / ".env", override=True
    )  # .env 优先于已存在的 OS 环境变量，见 config.py 注释

    _syms_raw = os.getenv("SYMBOLS", "QQQ,SPY,NVDA,AAPL,MSFT,META,GOOGL,AMZN")
    _syms = [s.strip().upper() for s in _syms_raw.split(",") if s.strip()]

    print(f"发送晨报，标的：{_syms}")
    _ok = send_morning_brief(symbols=_syms, db_path=_DB_PATH)
    sys.exit(0 if _ok else 1)
