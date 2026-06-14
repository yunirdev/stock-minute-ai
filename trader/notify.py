"""
notify.py
推送模块：ConsoleNotifier（打印）+ DiscordNotifier（webhook）。

红线：webhook URL 只从环境变量/config 读，不硬编码；无 URL 时降级到 console。
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict

from .models import Notification

logger = logging.getLogger(__name__)


class ConsoleNotifier:
    """实现 Notifier Protocol —— 打印到控制台/日志（始终成功）。"""

    def send(self, note: Notification) -> bool:
        logger.info("[NOTIFY] [%s] %s — %s", note.kind.upper(), note.title, note.body)
        if note.fields:
            logger.info("         fields=%s", note.fields)
        return True


class DiscordNotifier:
    """实现 Notifier Protocol —— 发送到 Discord webhook；无 URL 时降级到 console。"""

    def __init__(self, webhook_url: str | None = None) -> None:
        self._url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL", "")
        self._console = ConsoleNotifier()

    def send(self, note: Notification) -> bool:
        if not self._url:
            logger.warning("DISCORD_WEBHOOK_URL 未配置，降级到 console")
            return self._console.send(note)
        try:
            import urllib.request
            import json
            embed: Dict[str, Any] = {
                "title": note.title,
                "description": note.body,
                "color": _color_for_kind(note.kind),
            }
            if note.fields:
                embed["fields"] = [
                    {"name": k, "value": str(v), "inline": True}
                    for k, v in note.fields.items()
                ]
            payload = json.dumps({"embeds": [embed]}).encode("utf-8")
            req = urllib.request.Request(
                self._url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                ok = resp.status in (200, 204)
            if ok:
                logger.info("Discord 推送成功: %s", note.title)
            else:
                logger.warning("Discord 推送返回 %s", resp.status)
            return ok
        except Exception as exc:
            logger.error("Discord 推送失败: %s，降级 console", exc)
            return self._console.send(note)


def _color_for_kind(kind: str) -> int:
    return {
        "selection": 0x3498DB,   # 蓝
        "plan":      0x2ECC71,   # 绿
        "review":    0x9B59B6,   # 紫
        "alert":     0xE74C3C,   # 红
        "news":      0xF39C12,   # 橙
    }.get(kind, 0x95A5A6)       # 灰


# 默认单例（供 runtime 使用）
def make_notifier() -> DiscordNotifier:
    """优先 Discord，无 URL 时打印。"""
    return DiscordNotifier()
