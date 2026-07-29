"""Explicitly authorized, redacted, idempotent Discord delivery audit."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb

from .models import Notification

_SECRET_PATTERNS = (
    re.compile(r"(?i)(token|secret|api[_ -]?key)\s*[:=]\s*\S+"),
    re.compile(r"https://discord(?:app)?\.com/api/webhooks/\S+", re.I),
)


def sanitize_notification(note: Notification) -> Notification:
    def clean(value: Any) -> str:
        text = str(value)
        for pattern in _SECRET_PATTERNS:
            text = pattern.sub("[REDACTED]", text)
        return text[:4000]

    return Notification(
        title=clean(note.title),
        body=clean(note.body),
        kind=note.kind,
        fields={clean(key): clean(value) for key, value in note.fields.items()},
        plan_id=note.plan_id,
    )


class DiscordDeliveryStore:
    def __init__(
        self,
        db_path: str | Path,
        *,
        sender: Any,
        external_send_enabled: bool = False,
    ) -> None:
        self.db_path = str(db_path)
        self.sender = sender
        self.external_send_enabled = bool(external_send_enabled)
        connection = duckdb.connect(self.db_path)
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS discord_send_audit (
                    delivery_id TEXT PRIMARY KEY,
                    message_kind TEXT,
                    dedupe_key TEXT,
                    payload_hash TEXT,
                    status TEXT,
                    error_code TEXT,
                    payload_json TEXT,
                    created_at TIMESTAMPTZ,
                    completed_at TIMESTAMPTZ,
                    UNIQUE(message_kind, dedupe_key)
                )
                """
            )
            connection.commit()
        finally:
            connection.close()

    def deliver(
        self,
        note: Notification,
        *,
        message_kind: str,
        dedupe_key: str,
        dry_run: bool,
        now: datetime,
    ) -> dict[str, Any]:
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("DISCORD_DELIVERY_TIME_TZ_REQUIRED")
        kind = message_kind.strip().upper()
        key = dedupe_key.strip()
        if not kind or not key:
            raise ValueError("DISCORD_DELIVERY_IDENTITY_REQUIRED")
        safe = sanitize_notification(note)
        payload = {
            "title": safe.title,
            "body": safe.body,
            "kind": safe.kind,
            "fields": safe.fields,
            "plan_id": safe.plan_id,
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        payload_hash = hashlib.sha256(encoded.encode()).hexdigest()
        delivery_id = "discord-delivery-" + hashlib.sha256(
            f"{kind}|{key}".encode()
        ).hexdigest()[:24]
        existing = self.get(delivery_id)
        if existing is not None:
            if existing["payload_hash"] != payload_hash:
                raise ValueError("DISCORD_DELIVERY_DEDUPE_CONFLICT")
            return existing
        if dry_run:
            status, error = "DRY_RUN", ""
        elif not self.external_send_enabled:
            status, error = "BLOCKED", "DISCORD_EXTERNAL_SEND_NOT_AUTHORIZED"
        else:
            try:
                ok = bool(self.sender.send(safe))
                status, error = (
                    ("SENT", "") if ok else ("FAILED", "DISCORD_SEND_FAILED")
                )
            except Exception:
                status, error = "FAILED", "DISCORD_SEND_EXCEPTION"
        connection = duckdb.connect(self.db_path)
        try:
            connection.execute(
                """
                INSERT INTO discord_send_audit VALUES
                (?,?,?,?,?,?,?,?,?)
                """,
                [
                    delivery_id,
                    kind,
                    key,
                    payload_hash,
                    status,
                    error,
                    encoded,
                    now,
                    now,
                ],
            )
            connection.commit()
        finally:
            connection.close()
        return self.get(delivery_id)

    def get(self, delivery_id: str) -> dict[str, Any] | None:
        connection = duckdb.connect(self.db_path, read_only=True)
        try:
            row = connection.execute(
                "SELECT * FROM discord_send_audit WHERE delivery_id=?",
                [delivery_id],
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        return {
            "delivery_id": str(row[0]),
            "message_kind": str(row[1]),
            "dedupe_key": str(row[2]),
            "payload_hash": str(row[3]),
            "status": str(row[4]),
            "error_code": str(row[5] or ""),
            "payload": json.loads(row[6]),
            "created_at": row[7],
            "completed_at": row[8],
        }
