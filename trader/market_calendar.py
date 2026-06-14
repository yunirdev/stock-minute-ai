"""
market_calendar.py
美股交易时段判断（基于时区规则，不依赖外部服务）。

时段（美东时间 ET）：
  pre    : 04:00 – 09:30
  open   : 09:30 – 16:00
  post   : 16:00 – 20:00
  closed : 其余时间（含周末）
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

# UTC 偏移（EST=-5h，EDT=-4h；简化：用固定 -5，不做夏令时）
_UTC_TO_ET_OFFSET = -5


SessionName = Literal["pre", "open", "post", "closed"]


class SimpleMarketCalendar:
    """实现 MarketCalendar Protocol —— 基于固定时区规则的时段判断。"""

    def session_now(self) -> SessionName:
        return session_at(datetime.now(timezone.utc))


def session_at(dt: datetime) -> SessionName:
    """给定 UTC datetime 返回美股时段（简化规则，不含节假日）。"""
    if dt.tzinfo is not None:
        # 转换为 UTC
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)

    # 粗算美东时间
    et_hour = dt.hour + _UTC_TO_ET_OFFSET
    et_minute = dt.minute
    # 转为分钟数（从当日 00:00 开始）
    et_total = (et_hour % 24) * 60 + et_minute

    # 周末（0=Mon … 6=Sun in datetime.weekday()）
    # 注意 ET 转换后可能跨日，这里做简单处理
    et_day = (dt + __import__("datetime").timedelta(hours=_UTC_TO_ET_OFFSET)).weekday()
    if et_day >= 5:  # 周六=5, 周日=6
        return "closed"

    # 时段（分钟）
    pre_start   = 4 * 60           # 04:00
    market_open = 9 * 60 + 30     # 09:30
    market_close = 16 * 60        # 16:00
    post_end     = 20 * 60        # 20:00

    if pre_start <= et_total < market_open:
        return "pre"
    elif market_open <= et_total < market_close:
        return "open"
    elif market_close <= et_total < post_end:
        return "post"
    else:
        return "closed"


# 默认实例
_calendar = SimpleMarketCalendar()


def session_now() -> SessionName:
    return _calendar.session_now()
