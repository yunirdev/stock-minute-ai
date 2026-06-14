"""
universe.py
标的池管理：返回预设/自定义标的列表。
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

_BUILTIN_UNIVERSES: Dict[str, List[str]] = {
    "default":  ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"],
    "mega_cap": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO"],
    "etf":      ["SPY", "QQQ", "IWM", "GLD", "TLT", "XLK", "XLF", "XLE"],
    "watchlist": [],   # 用户自定义，从 conf/watchlist.json 读
}

_WATCHLIST_PATH = Path("conf/watchlist.json")


class ConfigUniverseProvider:
    """实现 UniverseProvider Protocol —— 从内置表 + 文件/env 读取标的池。"""

    def get_universe(self, name: str = "default") -> List[str]:
        if name == "watchlist":
            return self._load_watchlist()
        syms = _BUILTIN_UNIVERSES.get(name)
        if syms is not None:
            return syms
        # 允许直接传逗号分隔 symbol 串
        if "," in name or name.isupper():
            return [s.strip().upper() for s in name.split(",") if s.strip()]
        env_syms = os.getenv("SYMBOLS", "")
        if env_syms:
            return [s.strip().upper() for s in env_syms.split(",") if s.strip()]
        return _BUILTIN_UNIVERSES["default"]

    @staticmethod
    def _load_watchlist() -> List[str]:
        if not _WATCHLIST_PATH.exists():
            return []
        try:
            data = json.loads(_WATCHLIST_PATH.read_text(encoding="utf-8"))
            return [s.upper() for s in data if isinstance(s, str)]
        except Exception as exc:
            logger.warning("watchlist 读取失败: %s", exc)
            return []


# 默认实例
_provider = ConfigUniverseProvider()


def get_universe(name: str = "default") -> List[str]:
    return _provider.get_universe(name)
