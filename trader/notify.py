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

import hashlib
import json
import logging
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from .discord_delivery import DiscordDeliveryStore
from .discord_limits import fit_notification
from .models import Notification

logger = logging.getLogger(__name__)

_DISCORD_API = "https://discord.com/api/v10"


class ConsoleNotifier:
    """实现 Notifier —— 打印到控制台/日志（始终成功）。"""

    def send(self, note: Notification) -> bool:
        logger.info("[NOTIFY] [%s] %s — %s", note.kind.upper(), note.title, note.body)
        if note.fields:
            logger.info("         fields=%s", note.fields)
        return True


class DiscordNotifier:
    """
    实现 Notifier。
    优先使用 Bot Token + Channel ID；无 token 时退回 Webhook URL；两者都没有降级 console。
    """

    def __init__(
        self,
        bot_token: str | None = None,
        channel_id: str | None = None,
        webhook_url: str | None = None,
        *,
        external_send_enabled: bool | None = None,
        audit_db_path: str | None = None,
    ) -> None:
        # None = 未显式传入，读环境变量；"" = 显式禁用，不读环境变量（测试用）
        self._token   = (bot_token   if bot_token   is not None else os.getenv("DISCORD_BOT_TOKEN",   "")).strip()
        self._channel = (channel_id  if channel_id  is not None else os.getenv("DISCORD_CHANNEL_ID",  "")).strip()
        self._webhook = (webhook_url if webhook_url  is not None else os.getenv("DISCORD_WEBHOOK_URL", "")).strip()
        self._console = ConsoleNotifier()
        explicitly_configured = any(
            value is not None for value in (bot_token, channel_id, webhook_url)
        )
        if external_send_enabled is None:
            external_send_enabled = explicitly_configured or os.getenv(
                "DISCORD_EXTERNAL_SEND_ENABLED", ""
            ).strip().lower() in {"1", "true", "yes", "on"}
        self._external_send_enabled = bool(external_send_enabled)
        root = Path(__file__).resolve().parents[1]
        raw_db = Path(
            audit_db_path or os.getenv("TRADE_DB_PATH", "trade.duckdb")
        )
        self._audit_db_path = str(
            raw_db if raw_db.is_absolute() else root / raw_db
        )

    # ── 公共入口 ─────────────────────────────────────────────────────────────

    def send(self, note: Notification) -> bool:
        configured = bool((self._token and self._channel) or self._webhook)
        if not configured:
            logger.warning(
                "Discord 未配置（无 BOT_TOKEN/CHANNEL_ID 也无 WEBHOOK_URL），降级 console"
            )
        payload_key = "|".join(
            (
                note.kind,
                note.title,
                note.body,
                str(note.plan_id or ""),
                datetime.now(timezone.utc).date().isoformat(),
                f"authorized={self._external_send_enabled}",
            )
        )
        dedupe_key = hashlib.sha256(payload_key.encode()).hexdigest()
        try:
            result = DiscordDeliveryStore(
                self._audit_db_path,
                sender=_DiscordTransport(self),
                external_send_enabled=self._external_send_enabled,
            ).deliver(
                note,
                message_kind=note.kind or "system",
                dedupe_key=dedupe_key,
                dry_run=not configured,
                now=datetime.now(timezone.utc),
            )
        except Exception as exc:
            logger.error(
                "Discord 审计失败，禁止外发: %s",
                type(exc).__name__,
            )
            return self._console.send(note) if not configured else False
        if result["status"] == "DRY_RUN":
            return self._console.send(note)
        return result["status"] == "SENT"

    def _send_configured(self, note: Notification) -> bool:
        """整形到 Discord 允许的尺寸后逐条发出。

        整形放在审计之后、POST 之前：审计库留存的是完整原文（便于事后追溯
        "我们本来打算发什么"），而 Discord 上呈现的是按语义分好页的版本。一
        条业务通知仍然只对应一条审计记录，分页不会污染去重身份。
        """
        pages = fit_notification(note)
        ok = True
        for page in pages:
            if self._token and self._channel:
                sent = self._send_bot(page)
            elif self._webhook:
                sent = self._send_webhook(page)
            else:
                return False
            ok = sent and ok
            if not sent:
                # 分页里有一页失败就别继续刷了，剩下的页对读者也是残缺的
                break
        return ok

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
            self._console.send(note)
            return False


class _DiscordTransport:
    def __init__(self, notifier: DiscordNotifier) -> None:
        self.notifier = notifier

    def send(self, note: Notification) -> bool:
        return self.notifier._send_configured(note)


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
def make_notifier(
    *,
    external_send_enabled: bool | None = None,
) -> DiscordNotifier:
    """优先 Bot Token，其次 Webhook，最后 console。"""
    return DiscordNotifier(external_send_enabled=external_send_enabled)
