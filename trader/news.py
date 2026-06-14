"""
news.py
新闻/异动模块：
  - PriceMoveSource：从本地 bars 计算涨跌幅异动，生成 price_move 类 NewsEvent。
  - NewsSourceStub：占位，返回空列表。
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import List

from .data_cache import get_bars
from .models import NewsEvent, new_id, utc_now

logger = logging.getLogger(__name__)

_DEFAULT_THRESHOLD = 0.03   # 3% 涨跌幅触发异动


class PriceMoveSource:
    """实现 NewsSource Protocol —— 基于本地 bars 的价格异动侦测。"""

    def __init__(
        self,
        universe: List[str] | None = None,
        timeframe: str = "5m",
        threshold: float = _DEFAULT_THRESHOLD,
    ) -> None:
        self._universe = universe or []
        self._timeframe = timeframe
        self._threshold = threshold

    def poll(self, since: datetime) -> List[NewsEvent]:
        events: List[NewsEvent] = []
        for symbol in self._universe:
            try:
                df = get_bars(symbol, self._timeframe)
                if df is None or len(df) < 2:
                    continue
                recent = df[df["timestamp_utc"] >= since] if "timestamp_utc" in df.columns else df.tail(12)
                if recent.empty:
                    continue
                first_close = float(recent["close"].iloc[0])
                last_close = float(recent["close"].iloc[-1])
                if first_close <= 0:
                    continue
                pct = (last_close - first_close) / first_close
                if abs(pct) >= self._threshold:
                    direction = "上涨" if pct > 0 else "下跌"
                    events.append(NewsEvent(
                        event_id=new_id(),
                        kind="price_move",
                        symbol=symbol,
                        title=f"{symbol} {direction} {abs(pct)*100:.1f}%",
                        summary=f"从 {first_close:.2f} 到 {last_close:.2f}，变动 {pct*100:+.2f}%",
                        severity=min(abs(pct) / 0.10, 1.0),  # 10% 为满分
                        ts=utc_now(),
                        source="price_move",
                    ))
                    logger.info("📈 异动 %s %s pct=%.2f%%", symbol, direction, pct * 100)
            except Exception as exc:
                logger.warning("price_move 跳过 %s: %s", symbol, exc)
        return events


class NewsSourceStub:
    """占位实现，始终返回空列表。"""

    def poll(self, since: datetime) -> List[NewsEvent]:
        return []
