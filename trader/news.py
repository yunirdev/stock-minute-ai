"""
news.py
新闻/异动模块：
  - PriceMoveSource：从本地 bars 计算涨跌幅异动，生成 price_move 类 NewsEvent。
  - WallStreetCNSource：华尔街见闻 7x24 快讯 API（全球/外汇/黄金等频道）。
  - SECEdgarSource：SEC EDGAR 8-K 实时申报流（无需 API key）。
  - FinnhubSource：Finnhub 公司新闻 API（需 FINNHUB_API_KEY，无 key 则静默跳过）。
"""
from __future__ import annotations

import json as _json
import logging
import os
import time as _time
import urllib.request as _ur
import xml.etree.ElementTree as _ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Set, Tuple

import duckdb

from .data_cache import get_bars
from .models import NewsEvent, new_id, utc_now

logger = logging.getLogger(__name__)

_DEFAULT_THRESHOLD = 0.03   # 3% 涨跌幅触发异动


class PriceMoveSource:
    """实现 NewsSource —— 基于本地 bars 的价格异动侦测。"""

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


class SECEdgarSource:
    """
    SEC EDGAR 8-K 实时申报流。

    轮询各 ticker 最新 8-K 申报（Atom feed），信噪比极高：
    8-K = 重大事件（盈利预警、并购重组、高管变动、诉讼等）。

    完全免费，无需 API key。EDGAR 要求 User-Agent 含联系方式。
    配置：SEC_USER_AGENT=CompanyName your@email.com  （.env 可选）
    """

    _BASE = "https://www.sec.gov/cgi-bin/browse-edgar"
    _ATOM = "http://www.w3.org/2005/Atom"

    def __init__(
        self,
        universe: Optional[List[str]] = None,
        count: int = 5,
        timeout: float = 10.0,
    ) -> None:
        self._universe = [s.upper() for s in (universe or [])]
        self._count = count
        self._timeout = timeout
        self._seen_ids: Set[str] = set()
        ua = os.getenv("SEC_USER_AGENT", "stock-minute-ai/1.0 bot@example.com")
        self._headers = {"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}

    def poll(self, since: datetime) -> List[NewsEvent]:
        since_ts = since.timestamp()
        events: List[NewsEvent] = []
        for symbol in self._universe:
            try:
                batch = self._fetch(symbol, since_ts)
                events.extend(batch)
                _time.sleep(0.15)       # EDGAR 限速：max 10 req/sec
            except Exception as exc:
                logger.warning("SEC EDGAR [%s] 失败: %s", symbol, exc)
        if events:
            logger.info("SEC EDGAR: +%d 条 8-K 申报", len(events))
        return events

    def _fetch(self, symbol: str, since_ts: float) -> List[NewsEvent]:
        url = (
            f"{self._BASE}?action=getcompany&CIK={symbol}"
            f"&type=8-K&dateb=&owner=include&count={self._count}&output=atom"
        )
        req = _ur.Request(url, headers=self._headers)
        with _ur.urlopen(req, timeout=self._timeout) as resp:
            body = resp.read()

        # EDGAR 找不到 ticker 时返回 HTML 而非 Atom XML
        stripped = body.lstrip()
        if not (stripped.startswith(b"<?xml") or stripped.startswith(b"<feed")):
            logger.debug("SEC EDGAR [%s] 返回非 XML（ticker 不存在或限速），跳过", symbol)
            return []

        root = _ET.fromstring(body)
        ns = {"a": self._ATOM}

        # 尝试从 company-info（非 Atom 命名空间）提取公司名
        ci = root.find("company-info")
        company = (
            ci.findtext("conformed-name", default=symbol)
            if ci is not None else symbol
        )

        events: List[NewsEvent] = []
        for entry in root.findall("a:entry", ns):
            entry_id = entry.findtext("a:id", namespaces=ns) or ""
            updated_str = entry.findtext("a:updated", namespaces=ns) or ""
            link_el = entry.find("a:link", ns)
            link_url = link_el.get("href", "") if link_el is not None else ""

            if not entry_id or entry_id in self._seen_ids:
                continue
            try:
                ts = datetime.fromisoformat(updated_str.replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                ts_utc = ts.astimezone(timezone.utc)
            except Exception:
                continue
            if ts_utc.timestamp() < since_ts:
                continue

            self._seen_ids.add(entry_id)
            events.append(NewsEvent(
                event_id=new_id(),
                kind="sec_8k",
                symbol=symbol,
                title=f"{company} 提交 8-K（重大事件申报）",
                summary=(
                    f"{symbol} ({company}) 向 SEC 提交 8-K 申报。"
                    "8-K 涵盖：盈利预警、并购重组、高管变动、诉讼等重大事件。"
                    f"申报时间: {ts_utc.strftime('%Y-%m-%d %H:%M UTC')}"
                ),
                url=link_url,
                severity=0.8,
                ts=ts_utc,
                source="sec_edgar",
            ))
        return events


class FinnhubSource:
    """
    Finnhub 公司新闻 API（免费 tier：60 req/min）。

    需在 .env 配置 FINNHUB_API_KEY（注册 finnhub.io 免费获取）。
    无 key 时静默跳过，不报错，系统其他新闻源正常工作。
    """

    _BASE = "https://finnhub.io/api/v1"

    def __init__(
        self,
        universe: Optional[List[str]] = None,
        api_key: Optional[str] = None,
        timeout: float = 8.0,
    ) -> None:
        self._key = api_key or os.getenv("FINNHUB_API_KEY", "")
        self._universe = [s.upper() for s in (universe or [])]
        self._timeout = timeout
        self._seen_ids: Set[int] = set()
        self._disabled = False
        self._auth_warning_logged = False
        if not self._key:
            logger.debug("FinnhubSource: 未设置 FINNHUB_API_KEY，已跳过")

    def poll(self, since: datetime) -> List[NewsEvent]:
        if not self._key or self._disabled:
            return []
        since_ts = since.timestamp()
        from_date = since.strftime("%Y-%m-%d")
        to_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        events: List[NewsEvent] = []
        for symbol in self._universe:
            try:
                items = self._fetch(symbol, from_date, to_date)
                for item in items:
                    ev = self._to_event(item, since_ts, symbol)
                    if ev:
                        events.append(ev)
                _time.sleep(0.05)       # 60 req/min 充裕
            except Exception as exc:
                if getattr(exc, "code", None) in (401, 403):
                    if not self._auth_warning_logged:
                        logger.warning(
                            "Finnhub API key is unauthorized; disabling Finnhub news source for this process"
                        )
                        self._auth_warning_logged = True
                    self._disabled = True
                    break
                logger.warning("Finnhub [%s] 失败: %s", symbol, exc)
        if events:
            logger.info("Finnhub: +%d 条公司新闻", len(events))
        return events

    def _fetch(self, symbol: str, from_date: str, to_date: str) -> List[Dict]:
        url = (
            f"{self._BASE}/company-news"
            f"?symbol={symbol}&from={from_date}&to={to_date}&token={self._key}"
        )
        req = _ur.Request(url, headers={"Accept": "application/json"})
        with _ur.urlopen(req, timeout=self._timeout) as resp:
            return _json.loads(resp.read()) or []

    def _to_event(self, item: Dict, since_ts: float, symbol: str) -> Optional[NewsEvent]:
        item_id = int(item.get("id", 0))
        dt = item.get("datetime", 0)
        if dt < since_ts or item_id in self._seen_ids:
            return None
        headline = (item.get("headline") or "").strip()
        if not headline:
            return None
        self._seen_ids.add(item_id)
        return NewsEvent(
            event_id=new_id(),
            kind="news",
            symbol=symbol,
            title=headline,
            summary=(item.get("summary") or headline)[:300],
            url=item.get("url", ""),
            severity=0.5,
            ts=datetime.fromtimestamp(dt, tz=timezone.utc),
            source=f"finnhub/{item.get('source', 'unknown')}",
        )


class NewsSource(Protocol):
    def poll(self, since: datetime) -> List[NewsEvent]: ...


class NewsEventStore:
    """Durable, deduplicated storage for polled NewsEvent objects.

    Previously every poll() result was fetched and immediately discarded —
    no persistence, no UI, no downstream consumer. This gives news events a
    real destination so `poll_all_sources` results are actually visible.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        connection = duckdb.connect(self.db_path)
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS news_events (
                    event_id TEXT PRIMARY KEY,
                    kind TEXT,
                    symbol TEXT,
                    title TEXT,
                    summary TEXT,
                    url TEXT,
                    severity DOUBLE,
                    ts TIMESTAMPTZ,
                    source TEXT,
                    recorded_at TIMESTAMPTZ
                )
                """
            )
            connection.commit()
        finally:
            connection.close()

    def record_batch(self, events: List[NewsEvent], *, recorded_at: datetime) -> int:
        """Insert new events, silently skipping ones already recorded (by event_id).

        Returns the number of events attempted (DuckDB's ON CONFLICT DO
        NOTHING doesn't expose an affected-row count for plain INSERT), so
        this is an upper bound on how many were actually new.
        """
        if not events:
            return 0
        connection = duckdb.connect(self.db_path)
        try:
            for event in events:
                connection.execute(
                    """
                    INSERT INTO news_events VALUES (?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT (event_id) DO NOTHING
                    """,
                    [
                        event.event_id,
                        event.kind,
                        event.symbol,
                        event.title,
                        event.summary,
                        event.url,
                        event.severity,
                        event.ts,
                        event.source,
                        recorded_at,
                    ],
                )
            connection.commit()
        finally:
            connection.close()
        return len(events)


def poll_all_sources(
    sources: List[Tuple[str, NewsSource, datetime]],
    store: Optional[NewsEventStore] = None,
    *,
    now: datetime,
) -> List[NewsEvent]:
    """Poll every (name, source, since) tuple, tolerating per-source failures.

    A single failing source (e.g. Finnhub rate-limited) must not block the
    others. Successfully polled events are persisted via `store` when given.
    Callers should invoke this unconditionally every tick — these sources
    are independent of market hours (SEC 8-K filings, wire news, and price
    moves against the local bar cache can all happen pre/post market).
    """
    all_events: List[NewsEvent] = []
    for name, source, since in sources:
        try:
            batch = source.poll(since=since)
        except Exception as exc:
            logger.warning("news.poll [%s] 失败: %s", name, exc)
            continue
        if batch:
            logger.info("新闻 [%s]: %d 条", name, len(batch))
        all_events.extend(batch)
    if store is not None and all_events:
        store.record_batch(all_events, recorded_at=now)
    return all_events
