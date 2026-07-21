"""
notify.py
推送模块：ConsoleNotifier → DiscordNotifier（Bot Token 优先，Webhook 兜底）。

优先级：
  1. DISCORD_BOT_TOKEN + DISCORD_CHANNEL_ID  → POST /channels/{id}/messages
  2. DISCORD_WEBHOOK_URL                     → POST webhook URL
  3. 降级到 ConsoleNotifier（打印日志）

红线：token/URL 只从环境变量读，不硬编码。
"""
from __future__ import annotations

import json
import logging
import os
import urllib.request
from typing import Any, Dict

from .models import Notification

logger = logging.getLogger(__name__)

_DISCORD_API = "https://discord.com/api/v10"


class ConsoleNotifier:
    """实现 Notifier Protocol —— 打印到控制台/日志（始终成功）。"""

    def send(self, note: Notification) -> bool:
        logger.info("[NOTIFY] [%s] %s — %s", note.kind.upper(), note.title, note.body)
        if note.fields:
            logger.info("         fields=%s", note.fields)
        return True


class DiscordNotifier:
    """
    实现 Notifier Protocol。
    优先使用 Bot Token + Channel ID；无 token 时退回 Webhook URL；两者都没有降级 console。
    """

    def __init__(
        self,
        bot_token: str | None = None,
        channel_id: str | None = None,
        webhook_url: str | None = None,
    ) -> None:
        # None = 未显式传入，读环境变量；"" = 显式禁用，不读环境变量（测试用）
        self._token   = (bot_token   if bot_token   is not None else os.getenv("DISCORD_BOT_TOKEN",   "")).strip()
        self._channel = (channel_id  if channel_id  is not None else os.getenv("DISCORD_CHANNEL_ID",  "")).strip()
        self._webhook = (webhook_url if webhook_url  is not None else os.getenv("DISCORD_WEBHOOK_URL", "")).strip()
        self._console = ConsoleNotifier()

    # ── 公共入口 ─────────────────────────────────────────────────────────────

    def send(self, note: Notification) -> bool:
        if self._token and self._channel:
            return self._send_bot(note)
        if self._webhook:
            return self._send_webhook(note)
        logger.warning("Discord 未配置（无 BOT_TOKEN/CHANNEL_ID 也无 WEBHOOK_URL），降级 console")
        return self._console.send(note)

    # ── Bot Token 方式 ────────────────────────────────────────────────────────

    def _send_bot(self, note: Notification) -> bool:
        url = f"{_DISCORD_API}/channels/{self._channel}/messages"
        payload = json.dumps({"embeds": [self._build_embed(note)]}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Authorization": f"Bot {self._token}",
                "Content-Type": "application/json",
                "User-Agent": "DiscordBot (https://github.com/stock-minute-ai, 1.0)",
            },
            method="POST",
        )
        return self._post(req, note)

    # ── Webhook 方式 ──────────────────────────────────────────────────────────

    def _send_webhook(self, note: Notification) -> bool:
        payload = json.dumps({"embeds": [self._build_embed(note)]}).encode("utf-8")
        req = urllib.request.Request(
            self._webhook,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return self._post(req, note)

    # ── 共用工具 ──────────────────────────────────────────────────────────────

    def _build_embed(self, note: Notification) -> Dict[str, Any]:
        embed: Dict[str, Any] = {
            "title":       note.title,
            "description": note.body,
            "color":       _color_for_kind(note.kind),
        }
        if note.fields:
            embed["fields"] = [
                {"name": k, "value": str(v), "inline": True}
                for k, v in note.fields.items()
            ]
        return embed

    def _post(self, req: urllib.request.Request, note: Notification) -> bool:
        try:
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
        "ai":        0x58A6FF,   # AI 蓝
    }.get(kind, 0x95A5A6)       # 灰


# 默认单例（供 runtime / cockpit 使用）
def make_notifier() -> DiscordNotifier:
    """优先 Bot Token，其次 Webhook，最后 console。"""
    return DiscordNotifier()
