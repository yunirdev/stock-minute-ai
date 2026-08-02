"""开盘确认（09:45 ET）—— 晨报的回执。

晨报每天都给可证伪的关键触发位："SPY 站回 20MA 745.69 且上破昨高 748.90 才
偏多，跌破昨低 737.68 转防守"。开盘 15 分钟后这些条件已经有了明确答案，但在
这份报告出现之前没有任何东西回来告诉读者答案是什么——晨报给了三套剧本
（Bull/Base/Bear），却没人说今天到底走哪一套。

这里刻意不依赖晨报传递任何状态：20MA、昨高、昨低都是 T-1 收盘就固定下来的历
史数据，开盘后不会再变，所以重新算一遍必然得到和晨报一致的数值，比把它们序
列化存起来再读回来简单得多，也不会因为存储失败就整份报告做不出来。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from .models import Notification

logger = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")

#: 开盘区间取多少分钟。15 分钟是常用的 OR 窗口，也和晨报"不追第一根 5m K"
#: 的纪律对得上。
OPEN_RANGE_MINUTES = 15


@dataclass(frozen=True)
class TriggerCheck:
    """晨报给的一个触发条件，以及它现在成立与否。"""

    label: str
    level: float
    hit: bool
    actual: float

    def render(self) -> str:
        mark = "✅" if self.hit else "❌"
        return f"├ {self.label} {self.level:,.2f}　{mark} 现价 {self.actual:,.2f}"


def check_index_triggers(
    name: str,
    technical: Dict[str, Any],
    last_price: float,
) -> List[TriggerCheck]:
    """把晨报对一个指数给出的三个条件逐条对答案。

    条件和 morning_brief._action_trigger_line 用的是同一批数值（20MA、昨高、
    昨低），所以这里的打钩结果与早上那句话一一对应。
    """
    checks: List[TriggerCheck] = []
    # 字段名对齐 morning_brief._calc_index_technical 的输出：昨高/昨低在那里
    # 叫 resistance/support，晨报那句"上破昨高 748.90，跌破昨低 737.68"用的
    # 就是这两个值。取同一批数字才能保证打钩结果和早上说的一一对应。
    ma20 = technical.get("ma20")
    prior_high = technical.get("resistance") or technical.get("prior_high")
    prior_low = technical.get("support") or technical.get("prior_low")

    if ma20:
        checks.append(
            TriggerCheck(f"{name} 站回 20MA", float(ma20), last_price >= float(ma20), last_price)
        )
    if prior_high:
        checks.append(
            TriggerCheck(
                f"{name} 上破昨高",
                float(prior_high),
                last_price > float(prior_high),
                last_price,
            )
        )
    if prior_low:
        checks.append(
            TriggerCheck(
                f"{name} 跌破昨低",
                float(prior_low),
                last_price < float(prior_low),
                last_price,
            )
        )
    return checks


def decide_playbook(checks: List[TriggerCheck]) -> tuple[str, str]:
    """按打钩结果判定今天走哪套剧本。

    返回 (剧本名, 一句话行动)。判定顺序是先看防守——跌破昨低是明确的转防守
    信号，哪怕同时还站在 20MA 上方，也不该按多头剧本操作。
    """
    if not checks:
        return "无法判定", "缺少指数技术位，本轮不做剧本判定"

    broke_low = any(c.hit for c in checks if "跌破昨低" in c.label)
    if broke_low:
        return "Bear/防守", "已触发防守条件，不新开多单，优先看相对弱标的"

    broke_high = any(c.hit for c in checks if "上破昨高" in c.label)
    above_ma = any(c.hit for c in checks if "站回 20MA" in c.label)
    if broke_high and above_ma:
        return "Bull", "多头条件成立，但仍需 OR 上方有量能承接才考虑进场"
    if above_ma:
        return "Base", "站上均线但未破昨高，等回踩 OR 下沿或突破确认"
    return "Base", "关键条件均未触发，继续等开盘区间给方向"


def build_open_range_line(levels: Any) -> str:
    if levels is None:
        return "开盘区间：暂无足够分钟级数据"
    vwap = getattr(levels, "vwap", None)
    vwap_text = f"　VWAP {vwap:,.2f}" if vwap else ""
    stale = "　⚠️ 数据可能过期" if getattr(levels, "is_stale", False) else ""
    return (
        f"开盘 {getattr(levels, 'open_range_minutes', OPEN_RANGE_MINUTES)}m 区间："
        f"{levels.open_range_low:,.2f} – {levels.open_range_high:,.2f}"
        f"{vwap_text}{stale}"
    )


def build_premarket_drift_lines(
    premarket: Dict[str, float],
    actual: Dict[str, float],
    *,
    limit: int = 5,
) -> List[str]:
    """盘前预期 vs 开盘实际的偏差。

    盘前流动性稀薄，+5% 的盘前异动开盘后衰减到 +3% 是常事。晨报把这些标的
    列进了"重点观察"，读者需要知道那个理由现在还成不成立。
    """
    lines: List[str] = []
    for symbol, pre_pct in sorted(
        premarket.items(), key=lambda kv: abs(kv[1]), reverse=True
    )[:limit]:
        if symbol not in actual:
            continue
        now_pct = actual[symbol]
        drift = now_pct - pre_pct
        # 措辞要与方向无关：一个从 -1.70% 跌到 -2.40% 的标的，异动幅度是在
        # 放大，但说它"走强"会被读成股价上涨，恰好反了。这里描述的是异动本
        # 身的强弱，不是价格方向。
        if abs(drift) < 0.5:
            tag = "延续"
        elif abs(now_pct) < abs(pre_pct):
            tag = "异动减弱"
        else:
            tag = "异动加强"
        lines.append(
            f"• {symbol}　盘前 {pre_pct:+.2f}% → 开盘 {now_pct:+.2f}%　({tag})"
        )
    return lines


def build_open_confirmation_message(
    *,
    trading_date: str,
    index_levels: Dict[str, Any],
    technicals: Dict[str, Dict[str, Any]],
    premarket: Optional[Dict[str, float]] = None,
    actual: Optional[Dict[str, float]] = None,
    morning_bias: str = "",
    now_et: Optional[datetime] = None,
) -> Notification:
    """组装开盘确认报告。"""
    stamp = (now_et or datetime.now(_ET)).strftime("%H:%M ET")

    all_checks: List[TriggerCheck] = []
    body: List[str] = []

    if morning_bias:
        body.append(f"晨报早上的判断：{morning_bias}")
        body.append("")

    for name in ("SPY", "QQQ"):
        technical = technicals.get(name)
        levels = index_levels.get(name)
        if not technical or levels is None:
            continue
        checks = check_index_triggers(name, technical, float(levels.last_price))
        all_checks.extend(checks)
        body.append(f"**{name}**　现价 {levels.last_price:,.2f}")
        body.extend(check.render() for check in checks)
        body.append(build_open_range_line(levels))
        body.append("")

    playbook, action = decide_playbook(all_checks)

    drift_lines = build_premarket_drift_lines(premarket or {}, actual or {})
    if drift_lines:
        body.append("**盘前预期 vs 开盘实际**")
        body.extend(drift_lines)
        body.append("")

    body.append(f"**结论：按 {playbook} 剧本** — {action}")
    body.append(
        "_开盘区间刚形成，量能尚未充分确认；这是对晨报条件的对账，不是新的入场指令。_"
    )

    return Notification(
        title=f"🎯 开盘确认 · {trading_date} {stamp}",
        body="\n".join(body),
        kind="review",
        fields={"剧本": playbook},
        dedupe_key=f"open_confirmation:{trading_date}",
    )


def should_send_open_confirmation(
    now_utc: datetime,
    last_sent_date: Optional[str],
    *,
    hour_et: int = 9,
    minute_et: int = 45,
) -> bool:
    """开盘满 15 分钟后发一次，每天一次。

    用"到点之后"而不是"等于某分钟"：引擎 30 秒一 tick 看似不会错过，但一次
    重启、一次数据源卡顿就足以跳过某个特定分钟，那样这份报告当天就再也不会
    发了。晨报的 should_send_brief 至今仍是 hour == N 的精确匹配，有同样的
    脆弱性。
    """
    et = now_utc.astimezone(_ET)
    today = et.strftime("%Y-%m-%d")
    if today == last_sent_date:
        return False
    if et.weekday() >= 5:
        return False
    minutes = et.hour * 60 + et.minute
    start = hour_et * 60 + minute_et
    # 只在开盘后的头两小时内补发；更晚就失去"开盘确认"的意义了
    return start <= minutes < start + 120
