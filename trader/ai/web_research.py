"""
trader/ai/web_research.py
Agent-Reach CLI 封装层。

Agent-Reach 安装后提供以下 CLI 工具：
  - curl -s "https://r.jina.ai/<URL>"   → 读取任意网页（纯文本）
  - twitter search "query" -n 10        → Twitter/X 搜索
  - opencli reddit search "query" -f yaml → Reddit 帖子搜索
  - feedparser（Python 库）             → RSS/Atom Feed 解析
  - yt-dlp --dump-json URL              → YouTube 视频信息/字幕

用法：
    client = AgentReachClient()
    if client.is_available():
        pages = client.read_url("https://seekingalpha.com/article/...")
        tweets = client.search_twitter("AAPL earnings 2026", n=10)
        posts  = client.search_reddit("NVDA WSB", n=5)
        feed   = client.read_rss("https://feeds.finance.yahoo.com/rss/2.0/headline?s=AAPL")

所有方法 Agent-Reach 未安装时返回 [] 或 "" 并打 warning，不崩溃。
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import yaml
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 15   # 秒


class AgentReachClient:
    """
    调用 Agent-Reach 提供的 CLI 工具抓取网络内容。
    所有方法在工具未安装时优雅降级（返回空列表/空字符串）。
    """

    def __init__(self, timeout: int = _DEFAULT_TIMEOUT) -> None:
        self._timeout = timeout

    # ── 可用性检查 ─────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Agent-Reach 核心（Jina Reader curl）是否可用。"""
        return shutil.which("curl") is not None

    def has_twitter(self) -> bool:
        return shutil.which("twitter") is not None

    def has_reddit(self) -> bool:
        return shutil.which("opencli") is not None

    # ── 网页读取（Jina Reader，无需 API key）─────────────────────────────────

    def read_url(self, url: str, max_chars: int = 3000) -> str:
        """
        用 Jina Reader 读取网页并返回纯文本。
        支持任意 URL：SEC、Seeking Alpha、Bloomberg、公司官网等。
        """
        jina_url = f"https://r.jina.ai/{url}"
        result = self._run(["curl", "-s", "--max-time", str(self._timeout), jina_url])
        return result[:max_chars] if result else ""

    # ── RSS/Atom Feed ─────────────────────────────────────────────────────

    def read_rss(self, feed_url: str, max_items: int = 10) -> List[str]:
        """
        解析 RSS/Atom Feed，返回标题列表。
        使用 feedparser（Python 内置级，无需外部工具）。
        """
        try:
            import feedparser
            feed = feedparser.parse(feed_url)
            return [
                f"{entry.get('title', '')} — {entry.get('summary', '')[:100]}"
                for entry in feed.entries[:max_items]
            ]
        except ImportError:
            # feedparser 未安装，fallback 到 curl 读取
            raw = self._run(["curl", "-s", "--max-time", str(self._timeout), feed_url])
            return [raw[:500]] if raw else []
        except Exception as exc:
            logger.warning("RSS 解析失败 %s: %s", feed_url, exc)
            return []

    # ── Twitter/X 搜索 ───────────────────────────────────────────────────

    def search_twitter(self, query: str, n: int = 10) -> List[str]:
        """
        搜索 Twitter/X（需要 Agent-Reach 安装 twitter-cli）。
        返回推文文本列表。
        """
        if not self.has_twitter():
            logger.debug("twitter CLI 未安装，跳过 Twitter 搜索")
            return []
        raw = self._run(["twitter", "search", query, "-n", str(n)],
                        timeout=20)
        if not raw:
            return []
        try:
            parsed = yaml.safe_load(raw)
            if isinstance(parsed, dict) and parsed.get("ok") and isinstance(parsed.get("data"), list):
                results = []
                for t in parsed["data"][:n]:
                    text = t.get("text", "").replace("\n", " ").strip()
                    author = t.get("author", {}).get("screenName", "")
                    likes = t.get("metrics", {}).get("likes", 0)
                    if text:
                        results.append(f"@{author} (♥{likes}): {text[:200]}")
                return results
        except Exception:
            pass
        return [line.strip() for line in raw.splitlines() if line.strip()][:n]

    # ── Reddit 搜索 ──────────────────────────────────────────────────────

    def search_reddit(self, query: str, n: int = 5) -> List[str]:
        """
        搜索 Reddit（需要 Agent-Reach 安装 opencli）。
        返回帖子标题 + 摘要列表。
        """
        if not self.has_reddit():
            logger.debug("opencli 未安装，跳过 Reddit 搜索")
            return []
        raw = self._run(["opencli", "reddit", "search", query, "-f", "yaml", "--limit", str(n)],
                        timeout=30)
        if not raw:
            return []
        try:
            posts = yaml.safe_load(raw)
            if isinstance(posts, list):
                return [
                    f"{p.get('title', '')} [{p.get('subreddit', '')}] "
                    f"↑{p.get('score', 0)}"
                    for p in posts[:n]
                ]
        except Exception:
            pass
        return [line.strip() for line in raw.splitlines() if line.strip()][:n]

    # ── Fintwit 影响力账号（Twitter 大V）────────────────────────────────────
    #
    # 使用 Twitter 的 from: 运算符搜索指定账号的推文。
    # 分类说明：
    #   options_flow   — 期权/暗池数据驱动
    #   macro          — 宏观视角
    #   news           — 突发新闻
    #   quant_data     — 量化/国会数据
    #   technical      — 技术分析
    #   user_defined   — 用户自选
    #
    FINTWIT_INFLUENCERS: Dict[str, str] = {
        # ── 用户自选 ──────────────────────────────────────────────────
        "aleabitoreddit":  "user_defined",     # 用户关注，交易分析
        # ── 期权流/暗池 ─────────────────────────────────────────────────
        "unusual_whales":  "options_flow",     # 期权流 + 国会交易，全球最火之一
        "OptionsHawk":     "options_flow",     # 异常大单预警，实时扫描
        "quiverquant":     "quant_data",       # 国会持仓/机构/暗池数据
        # ── 宏观/突发新闻 ────────────────────────────────────────────────
        "DeItaone":        "news",             # Bloomberg 第一手消息
        "zerohedge":       "macro",            # 宏观风险/空头视角（高流量）
        "elerianm":        "macro",            # Mohamed El-Erian，前PIMCO CEO
        # ── 技术分析 ────────────────────────────────────────────────────
        "GarethSoloway":   "technical",        # 技术分析，零售端影响力大
    }

    def search_influencer_mentions(
        self,
        symbol: str,
        accounts: Dict[str, str] | None = None,
        n_per_batch: int = 5,
    ) -> List[str]:
        """
        搜索 Fintwit 大V 对指定标的的推文。
        使用 Twitter `from:` 运算符，批量查询（每批最多 3 个账号，避免 query 过长）。
        返回格式：["@username [category]: tweet text", ...]
        """
        if not self.has_twitter():
            return []
        accs = accounts or self.FINTWIT_INFLUENCERS
        if not accs:
            return []

        results: List[str] = []
        account_list = list(accs.items())

        # 分批查询（每批 3 个账号）
        batch_size = 3
        for i in range(0, len(account_list), batch_size):
            batch = account_list[i : i + batch_size]
            from_clause = " OR ".join(f"from:{acc}" for acc, _ in batch)
            query = f"${symbol} ({from_clause})"
            raw_tweets = self.search_twitter(query, n=n_per_batch)
            for tweet in raw_tweets:
                results.append(f"[Fintwit] {tweet}")

        return results[:15]  # 最多返回 15 条

    # ── 预设金融 RSS Feeds ────────────────────────────────────────────────

    FINANCIAL_RSS_FEEDS = {
        "yahoo_general":  "https://finance.yahoo.com/rss/topfinstories",
        "seekingalpha":   "https://seekingalpha.com/market_currents.xml",
        "marketwatch":    "https://feeds.marketwatch.com/marketwatch/topstories/",
        "investing_com":  "https://www.investing.com/rss/news_25.rss",
    }

    def read_financial_rss(self, max_items_per_feed: int = 5) -> List[str]:
        """读取所有预设财经 RSS Feeds，合并返回标题列表。"""
        all_items: List[str] = []
        for name, url in self.FINANCIAL_RSS_FEEDS.items():
            try:
                items = self.read_rss(url, max_items=max_items_per_feed)
                if items:
                    all_items.extend(items)
                    logger.debug("RSS [%s]: %d 条", name, len(items))
            except Exception as exc:
                logger.debug("RSS [%s] 失败: %s", name, exc)
        return all_items

    def read_symbol_news(self, symbol: str, max_items: int = 5) -> List[str]:
        """读取特定股票的 Yahoo Finance RSS 新闻（免费，无需 API key）。"""
        url = f"https://finance.yahoo.com/rss/headline?s={symbol}"
        return self.read_rss(url, max_items=max_items)

    # ── 内部工具 ──────────────────────────────────────────────────────────

    def _run(
        self,
        cmd: List[str],
        timeout: Optional[int] = None,
    ) -> str:
        """运行 CLI 命令并返回 stdout 文本；失败时返回空字符串。"""
        try:
            # Windows 上 npm 全局 CLI 生成 .CMD 文件，需要 cmd.exe 才能执行
            import sys
            actual_cmd = cmd
            if sys.platform == "win32":
                exe = shutil.which(cmd[0])
                if exe and exe.upper().endswith(".CMD"):
                    actual_cmd = ["cmd.exe", "/c"] + cmd
            result = subprocess.run(
                actual_cmd,
                capture_output=True,
                text=True,
                timeout=timeout or self._timeout,
                encoding="utf-8",
                errors="replace",
            )
            if result.returncode == 0:
                return result.stdout.strip()
            logger.debug("CLI %s 返回 code=%d: %s",
                         cmd[0], result.returncode, result.stderr[:100])
            return ""
        except subprocess.TimeoutExpired:
            logger.warning("CLI %s 超时（%ds）", cmd[0], timeout or self._timeout)
            return ""
        except FileNotFoundError:
            logger.debug("CLI %s 未安装", cmd[0])
            return ""
        except Exception as exc:
            logger.error("CLI %s 错误: %s", cmd[0], exc)
            return ""


# ---------------------------------------------------------------------------
# 全局单例
# ---------------------------------------------------------------------------

_client: Optional[AgentReachClient] = None


def get_web_research_client() -> AgentReachClient:
    global _client
    if _client is None:
        _client = AgentReachClient()
    return _client
