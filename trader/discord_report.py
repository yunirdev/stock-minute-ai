"""
trader/discord_report.py
Discord 推送内容模板 — 面向新手用户，用大白话解释分析结果。

━━ 修改指南 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  改文案措辞  → 修改本文件中各 _TEXT / _LABEL 常量，或各 _build_* 函数里的字符串
  改推送数量  → 修改 MAX_STOCK_CARDS（每次最多推几只股票）
  加新推送类型 → 新建 build_xxx() 函数，在调用方引入即可
  改字段排列  → 修改各 build_*() 函数里的 fields 字典
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from .models import Notification  # noqa: E402

# ━━ 可调参数 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MAX_STOCK_CARDS = 5  # 每轮最多推几只个股详情
MIN_SCORE_MENTION = 35  # 综合分低于此值的 AVOID 标的不单独推（只统计数量）

# ━━ 文案常量（改这里调整措辞）━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_VERDICT_STYLE: Dict[str, tuple] = {
    # verdict → (emoji, 一句话建议, Discord embed 颜色)
    "BUY": ("🟢", "建议关注买入", 0x2ECC71),
    "WATCHLIST": ("🟡", "可以持续观望", 0xF39C12),
    "AVOID": ("🔴", "建议暂时回避", 0xE74C3C),
}

_SCORE_LABEL: List[tuple] = [
    # (分数下界, 分数上界, 文字标签)
    (80, 101, "AI 非常看好 🔥"),
    (65, 80, "AI 偏向看好 📈"),
    (45, 65, "AI 中性观望 ➡️"),
    (30, 45, "AI 偏向谨慎 📉"),
    (0, 30, "AI 非常谨慎 ⚠️"),
]

_MACRO_REGIME: Dict[str, tuple] = {
    # regime → (氛围描述, 操作建议)
    "risk_on": ("市场情绪积极，资金整体流入股市", "可以正常操作，适度参与"),
    "neutral": ("市场情绪中性，方向不明朗", "建议轻仓，等待信号明朗"),
    "risk_off": ("市场情绪偏谨慎，资金开始避险", "建议控制仓位，减少新开仓"),
    "crisis": ("市场处于恐慌状态，大幅波动", "建议大幅减仓，优先保本"),
}

_TREND_LABEL: Dict[str, str] = {
    "strong_uptrend": "强势上涨 ↑↑",
    "uptrend": "上涨趋势 ↑",
    "neutral": "横盘整理 →",
    "downtrend": "下跌趋势 ↓",
    "strong_downtrend": "强势下跌 ↓↓",
}

# ━━ 内部工具函数 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _score_label(score: float) -> str:
    for lo, hi, label in _SCORE_LABEL:
        if lo <= score < hi:
            return label
    return "—"


def _verdict_style(verdict: str) -> tuple:
    return _VERDICT_STYLE.get(verdict.upper(), ("⚪", verdict, 0x95A5A6))


def _now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _g(d: Optional[dict], *keys, default: Any = None) -> Any:
    """安全地从嵌套字典取值，任何一层为 None 则返回 default。"""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


def _trunc(s: Any, n: int) -> str:
    s = str(s) if s else ""
    return s[:n] + "…" if len(s) > n else s


# ━━ 推送类型 1：AI 决策台分析完成 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def build_ai_analysis_messages(report_data: list) -> List[Notification]:
    """
    决策台「运行一轮」完成后调用。

    参数：
        report_data — _build_report_data() 返回的列表，每项包含一只股票的全部 agent 分析

    返回：
        [市场概况卡片, 个股详情卡片 × N]
    """
    if not report_data:
        return []

    msgs: List[Notification] = []

    # 市场概况（提取第一个有 macro 数据的条目）
    macro = next((r["macro"] for r in report_data if r.get("macro")), None)
    msgs.append(_build_market_overview(macro, report_data))

    # 个股详情：优先推 BUY，再推 WATCHLIST，AVOID 只在综合分 ≥ MIN_SCORE_MENTION 时推
    picks = [
        r
        for r in report_data
        if r["verdict"].upper() in ("BUY", "WATCHLIST")
        or r["composite_score"] >= MIN_SCORE_MENTION
    ][:MAX_STOCK_CARDS]

    for r in picks:
        msgs.append(_build_stock_card(r))

    return msgs


def _build_market_overview(macro: Optional[dict], report_data: list) -> Notification:
    """市场概况卡片：今天市场氛围如何，总体该怎么做。"""
    ts = _now_str()

    if macro:
        regime = macro.get("regime", "neutral")
        vix = macro.get("vix_level", "—")
        rate = macro.get("rate_outlook", "—")
        dollar = macro.get("dollar_signal", "—")
        desc, advice = _MACRO_REGIME.get(regime, ("市场状态未知", "保持观察"))
        reasoning = _trunc(macro.get("reasoning", ""), 180)
    else:
        vix = rate = dollar = "—"
        desc, advice = "暂无宏观数据", "保持观察"
        reasoning = ""

    buys = sum(1 for r in report_data if r["verdict"].upper() == "BUY")
    watchs = sum(1 for r in report_data if r["verdict"].upper() == "WATCHLIST")
    avoids = sum(1 for r in report_data if r["verdict"].upper() == "AVOID")

    body_lines = [
        f"🕐 {ts}",
        "",
        f"**今日市场氛围：** {desc}",
        f"**整体操作建议：** {advice}",
    ]
    if reasoning:
        body_lines += ["", f"> {reasoning}"]

    return Notification(
        title="📊 AI 市场分析简报",
        body="\n".join(body_lines),
        kind="ai",
        fields={
            "恐慌指数 VIX": str(vix),
            "利率走势": str(rate),
            "美元信号": str(dollar),
            "🟢 看好": f"{buys} 只",
            "🟡 观望": f"{watchs} 只",
            "🔴 回避": f"{avoids} 只",
        },
    )


def _build_stock_card(r: dict) -> Notification:
    """
    单只股票详情卡片，用大白话解释为什么值得关注、在哪里买、在哪里止损。
    """
    sym = r["symbol"]
    score = r["composite_score"]
    verdict = r.get("verdict", "WATCHLIST")
    emoji, verdict_text, _ = _verdict_style(verdict)

    # 技术面价格
    tech = r.get("technical") or {}
    close = tech.get("close")
    trend = _TREND_LABEL.get(tech.get("trend", ""), tech.get("trend", "—"))

    # 多空辩论结果
    bb = r.get("bull_bear") or {}
    bull = bb.get("bull") or {}
    bear = bb.get("bear") or {}
    key_fac = bb.get("key_factor", "")
    action_hint = bb.get("suggested_action", "")
    upside = bull.get("upside_target")
    stop_pct = bear.get("stop_loss_pct")
    bull_thesis = bull.get("thesis", "")
    bear_thesis = bear.get("thesis", "")

    # 基本面
    fund = r.get("fundamental") or {}
    pe = fund.get("pe_forward")
    rev_g = fund.get("revenue_growth_pct")

    # 大咖持仓
    elite = r.get("elite_holdings") or {}
    ark_held = _g(elite, "ark", "held_by") or []
    berk_held = _g(elite, "berkshire", "held") or False
    scion_held = _g(elite, "scion", "held") or False

    # 新闻催化剂
    cats = (r.get("news") or {}).get("catalysts", [])

    # ── 正文 ─────────────────────────────────────────────────────────────────
    lines: List[str] = [
        f"{emoji} **{verdict_text}**  综合评分 **{score:.0f} / 100**（{_score_label(score)}）",
    ]

    if close:
        lines.append(f"📍 当前价格：**${float(close):,.2f}**　走势：{trend}")

    # 价位建议
    price_lines = []
    if upside and close:
        pct = (float(upside) / float(close) - 1) * 100
        price_lines.append(
            f"🎯 目标价位：**${float(upside):,.2f}**（距今约 {pct:+.1f}%）"
        )
    if stop_pct and close:
        stop_px = float(close) * (1 - float(stop_pct) / 100)
        price_lines.append(
            f"🛑 建议止损：**${stop_px:,.2f}**（跌破此价位建议离场，控制损失）"
        )
    if price_lines:
        lines.append("")
        lines.extend(price_lines)

    # 操作建议
    if action_hint:
        lines += ["", f"💡 **操作建议**：{_trunc(action_hint, 150)}"]
    if key_fac:
        lines.append(f"🔑 **关键因素**：{_trunc(key_fac, 120)}")

    # 看多理由（简化为大白话）
    if bull_thesis:
        lines += ["", "**为什么值得关注：**", f"> {_trunc(bull_thesis, 220)}"]

    # 风险提示
    if bear_thesis:
        lines += ["", "**需要注意的风险：**", f"> {_trunc(bear_thesis, 180)}"]

    # ── 字段（右侧补充信息）──────────────────────────────────────────────────
    fields: Dict[str, str] = {}

    if pe:
        fields["市盈率 PE"] = f"{float(pe):.1f}x"
    if rev_g:
        fields["营收增速"] = f"{float(rev_g):+.1f}%"
    if cats:
        fields["近期催化剂"] = "、".join(_trunc(c, 20) for c in cats[:3])

    holders = []
    if ark_held:
        holders.append(f"ARK ({','.join(ark_held[:2])})")
    if berk_held:
        holders.append("巴菲特")
    if scion_held:
        holders.append("Burry")
    if holders:
        fields["大咖持仓"] = " · ".join(holders)

    return Notification(
        title=f"{emoji} {sym}  {verdict_text}",
        body="\n".join(lines),
        kind="ai",
        fields=fields,
    )


# ━━ 推送类型 2：每日收盘复盘 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def build_daily_review_message(
    today: str,
    pnl: float,
    trade_count: int,
    market_summary: str = "",
    symbols: Optional[List[str]] = None,
) -> Notification:
    """
    每天美股收盘后（21:00 UTC）自动推送复盘 + 近期高波动事件预警。
    runtime.py 的 _maybe_daily_review() 调用此函数。

    参数：
        today          — 日期字符串 "2026-06-17"
        pnl            — 当日盈亏（USD）
        trade_count    — 当日成交笔数
        market_summary — 市场简评（可选）
        symbols        — 关注的股票列表，用于获取财报日期（可选）
    """
    # ── 盈亏 ─────────────────────────────────────────────────────────────────
    if pnl > 0:
        pnl_line = f"📈 今天盈利 **+${pnl:,.2f}**，不错！"
    elif pnl < 0:
        pnl_line = f"📉 今天亏损 **${pnl:,.2f}**，注意控制风险。"
    else:
        pnl_line = "➡️ 今天持平，继续观察。"

    lines = [pnl_line, ""]

    if trade_count:
        lines.append(f"今日共成交 **{trade_count}** 笔订单。")
    else:
        lines.append("今日无成交，系统处于观察模式。")

    if market_summary:
        lines += ["", "**今日市场简评：**", f"> {_trunc(market_summary, 200)}"]

    # ── 近期高波动事件预警 ────────────────────────────────────────────────────
    event_block = _build_event_warning(symbols)
    if event_block:
        lines += ["", event_block]

    lines += [
        "",
        "📌 有新信号时会第一时间推送到这里，请留意。",
    ]

    pnl_str = f"+${pnl:,.2f}" if pnl >= 0 else f"-${abs(pnl):,.2f}"
    return Notification(
        title=f"📋 每日复盘 · {today}",
        body="\n".join(lines),
        kind="review",
        fields={
            "今日盈亏": pnl_str,
            "成交笔数": str(trade_count),
        },
    )


def _build_event_warning(symbols: Optional[List[str]] = None) -> str:
    """
    生成近期高波动事件预警文本块。
    返回空字符串表示没有值得提醒的事件。

    ━━ 修改事件展示逻辑 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      改看几天内的事件 → 修改下方 _DAYS_BY_IMPACT
      改显示几条事件 → 修改 MAX_EVENTS
      改日期显示格式 → 修改 _date_label()
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
    MAX_EVENTS = 8  # 最多显示几条

    # 不同重要程度提前几天显示
    _DAYS_BY_IMPACT = {"critical": 7, "high": 5, "medium": 3}

    try:
        from .calendar_events import get_upcoming_events, IMPACT_EMOJI

        events = get_upcoming_events(symbols=symbols, days=7)
    except Exception as exc:
        logger.warning("获取事件日历失败: %s", exc)
        return ""

    if not events:
        return ""

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # 按天分组
    by_date: dict[str, list] = {}
    shown = 0
    for e in events:
        if shown >= MAX_EVENTS:
            break
        # 按重要程度决定提前多少天显示
        max_days = _DAYS_BY_IMPACT.get(e.impact, 3)
        days_away = (
            datetime.strptime(e.date, "%Y-%m-%d")
            - datetime.strptime(today_str, "%Y-%m-%d")
        ).days
        if days_away < 0 or days_away > max_days:
            continue
        by_date.setdefault(e.date, []).append(e)
        shown += 1

    if not by_date:
        return ""

    parts = [
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "🚨 **近期高波动事件预警（可能影响交易）**",
    ]

    for date_str in sorted(by_date.keys()):
        label = _date_label(date_str, today_str)
        parts.append(f"\n**【{label}】**")
        for e in by_date[date_str]:
            imp = IMPACT_EMOJI.get(e.impact, "⚡")
            line = f"{imp}  `{e.time_str}`  **{e.title_cn}**"
            if e.note:
                line += f"\n    → {e.note}"
            parts.append(line)

    parts.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    return "\n".join(parts)


def _date_label(date_str: str, today_str: str) -> str:
    """把日期转成易懂的相对表示：明天、后天、周五 6/20 等。"""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        today_dt = datetime.strptime(today_str, "%Y-%m-%d")
        diff = (dt - today_dt).days
        weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        wd = weekdays[dt.weekday()]
        md = f"{dt.month}/{dt.day}"
        if diff == 0:
            return f"今天 · {wd} {md}"
        elif diff == 1:
            return f"明天 · {wd} {md}"
        elif diff == 2:
            return f"后天 · {wd} {md}"
        else:
            return f"{wd} {md}（{diff} 天后）"
    except Exception:
        return date_str
