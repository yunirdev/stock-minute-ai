"""
news.py
新闻/异动模块：
  - PriceMoveSource：从本地 bars 计算涨跌幅异动，生成 price_move 类 NewsEvent。
  - WallStreetCNSource：华尔街见闻 7x24 快讯 API（全球/外汇/黄金等频道）。
  - NewsSourceStub：占位，返回空列表。
"""
from __future__ import annotations

import json as _json
import logging
import urllib.request as _ur
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

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


class WallStreetCNSource:
    """
    华尔街见闻 7x24 快讯 API。

    直接调 api-prod.wallstreetcn.com JSON 接口，无需浏览器/Selenium。
    robots.txt 明确 Allow: /，且允许 ClaudeBot 等 AI 爬虫。

    API: GET https://api-prod.wallstreetcn.com/apiv1/content/lives
         ?channel=global-channel&num=20&cursor=0
    返回: {"code": 20000, "data": {"items": [...]}}

    每条字段：
      id            唯一 ID（用于去重）
      display_time  Unix 时间戳（秒）
      content_text  纯文本内容（适合 LLM 摘要）
      content       HTML（备用）
      score         重要度 1-3（3=重大事件）
      symbols       相关股票列表（常为空，宏观快讯不标注标的）
      channels      所属频道列表
      title         标题（快讯通常为空）
    """

    _API = "https://api-prod.wallstreetcn.com/apiv1/content/lives"
    _CHANNELS = {
        "global":  "global-channel",   # 全球要闻（默认）
        "forex":   "forex-channel",    # 外汇
        "gold":    "goldc-channel",    # 黄金
        "oil":     "oil-channel",      # 原油
        "cn":      "a-channel",        # A 股
        "us":      "us-channel",       # 美股
    }
    _HEADERS = {
        "User-Agent": "Mozilla/5.0 (compatible; trading-bot/1.0)",
        "Accept": "application/json",
    }

    def __init__(
        self,
        universe: Optional[List[str]] = None,
        channels: Optional[List[str]] = None,  # ["global", "forex"] 等
        num: int = 30,
        min_score: int = 1,   # 最低重要度（1-3；3=重大事件）
        timeout: float = 8.0,
    ) -> None:
        self._universe: List[str] = [s.upper() for s in (universe or [])]
        self._channel_ids: List[str] = [
            self._CHANNELS.get(c, c) for c in (channels or ["global"])
        ]
        self._num = num
        self._min_score = min_score
        self._timeout = timeout
        self._seen_ids: Set[int] = set()   # 进程级去重

    def poll(self, since: datetime) -> List[NewsEvent]:
        """拉取 since 之后的新快讯。每轮调用只返回新增条目（内存去重）。"""
        since_ts = since.timestamp()
        events: List[NewsEvent] = []

        for channel_id in self._channel_ids:
            try:
                items = self._fetch_channel(channel_id)
            except Exception as exc:
                logger.warning("WallStreetCN [%s] 请求失败: %s", channel_id, exc)
                continue

            for item in items:
                event = self._to_event(item, since_ts)
                if event is not None:
                    events.append(event)

        if events:
            logger.info("WallStreetCN: +%d 条快讯 (since %s)", len(events),
                        since.strftime("%H:%M UTC"))
        return events

    # ── 内部 ────────────────────────────────────────────────────────────────

    def _fetch_channel(self, channel_id: str) -> List[Dict]:
        url = f"{self._API}?channel={channel_id}&num={self._num}&cursor=0"
        req = _ur.Request(url, headers=self._HEADERS)
        with _ur.urlopen(req, timeout=self._timeout) as resp:
            data = _json.loads(resp.read())
        if data.get("code") != 20000:
            logger.warning("WallStreetCN API code=%s channel=%s", data.get("code"), channel_id)
            return []
        return data.get("data", {}).get("items", [])

    def _to_event(self, item: Dict, since_ts: float) -> Optional[NewsEvent]:
        item_id = item.get("id", 0)
        display_time = item.get("display_time", 0)

        if display_time < since_ts:
            return None
        if item_id in self._seen_ids:
            return None

        score = item.get("score", 1)
        if score < self._min_score:
            return None

        text = (item.get("content_text") or "").strip()
        if not text:
            return None

        # 标的匹配：symbols 字段 + 正文中的 ticker 出现
        item_tickers: List[str] = []
        for sym_obj in (item.get("symbols") or []):
            t = sym_obj.get("ticker", "") if isinstance(sym_obj, dict) else str(sym_obj)
            if t:
                item_tickers.append(t.upper())

        matched: Optional[str] = None
        if self._universe:
            for sym in self._universe:
                if sym in item_tickers or sym in text.upper():
                    matched = sym
                    break
            # 无 universe 匹配的宏观快讯：仍保留（symbol=None），让 agent 判断相关性
        elif item_tickers:
            matched = item_tickers[0]

        title = (item.get("title") or "").strip() or text[:60]
        ts = datetime.fromtimestamp(display_time, tz=timezone.utc)

        self._seen_ids.add(item_id)
        return NewsEvent(
            event_id=new_id(),
            kind="news",
            symbol=matched,
            title=title,
            summary=text[:300],
            url=f"https://wallstreetcn.com/live/{item_id}",
            severity=min(score / 3.0, 1.0),   # score=3 → severity=1.0
            ts=ts,
            source="wallstreetcn",
        )


class NewsSourceStub:
    """占位实现，始终返回空列表。"""

    def poll(self, since: datetime) -> List[NewsEvent]:
        return []
