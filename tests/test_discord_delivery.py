from datetime import datetime, timezone

import pytest

from trader.daily_discord import build_exception_message
from trader.discord_delivery import DiscordDeliveryStore
from trader.models import Notification
from trader.notify import DiscordNotifier

NOW = datetime(2026, 7, 27, 20, 0, tzinfo=timezone.utc)


class Sender:
    def __init__(self, result=True):
        self.result = result
        self.notes = []

    def send(self, note):
        self.notes.append(note)
        return self.result


def _note():
    return Notification(
        title="Daily token=abc",
        body="api_key: SUPERSECRET https://discord.com/api/webhooks/1/private",
        kind="ai",
        fields={"secret": "secret=hidden"},
    )


def test_dry_run_is_redacted_audited_and_does_not_send(tmp_path):
    sender = Sender()
    store = DiscordDeliveryStore(tmp_path / "trade.duckdb", sender=sender)
    result = store.deliver(
        _note(),
        message_kind="DAILY",
        dedupe_key="2026-07-27",
        dry_run=True,
        now=NOW,
    )
    assert result["status"] == "DRY_RUN"
    assert sender.notes == []
    assert "SUPERSECRET" not in str(result["payload"])
    assert "webhooks/1" not in str(result["payload"])


def test_external_send_requires_explicit_authorization(tmp_path):
    sender = Sender()
    store = DiscordDeliveryStore(tmp_path / "trade.duckdb", sender=sender)
    result = store.deliver(
        _note(),
        message_kind="DAILY",
        dedupe_key="2026-07-27",
        dry_run=False,
        now=NOW,
    )
    assert result["status"] == "BLOCKED"
    assert result["error_code"] == "DISCORD_EXTERNAL_SEND_NOT_AUTHORIZED"
    assert sender.notes == []


@pytest.mark.parametrize(
    "sender_result,expected",
    [(True, "SENT"), (False, "FAILED")],
)
def test_authorized_send_records_success_or_failure(
    tmp_path, sender_result, expected
):
    sender = Sender(sender_result)
    store = DiscordDeliveryStore(
        tmp_path / "trade.duckdb",
        sender=sender,
        external_send_enabled=True,
    )
    result = store.deliver(
        _note(),
        message_kind="EXCEPTION",
        dedupe_key="incident-1",
        dry_run=False,
        now=NOW,
    )
    assert result["status"] == expected
    assert len(sender.notes) == 1


def test_delivery_is_idempotent_and_conflicting_rewrite_fails(tmp_path):
    sender = Sender()
    store = DiscordDeliveryStore(
        tmp_path / "trade.duckdb",
        sender=sender,
        external_send_enabled=True,
    )
    values = {
        "message_kind": "DAILY",
        "dedupe_key": "2026-07-27",
        "dry_run": False,
        "now": NOW,
    }
    first = store.deliver(_note(), **values)
    assert store.deliver(_note(), **values) == first
    assert len(sender.notes) == 1
    with pytest.raises(ValueError, match="DEDUPE_CONFLICT"):
        store.deliver(Notification("changed", "body"), **values)


def test_exception_builder_has_code_time_summary_and_evidence():
    note = build_exception_message(
        error_code="BROKER_API_FAILED",
        summary="Paper broker unavailable",
        evidence_refs=["audit-1", "reconcile-1"],
        occurred_at=NOW,
    )
    assert "BROKER_API_FAILED" in note.title
    assert "Paper broker unavailable" in note.body
    assert "audit-1" in note.body


def test_notifier_routes_authorized_send_through_audit(tmp_path, monkeypatch):
    notifier = DiscordNotifier(
        bot_token="",
        channel_id="",
        webhook_url="https://discord.test/webhook",
        external_send_enabled=True,
        audit_db_path=str(tmp_path / "trade.duckdb"),
    )
    calls = []
    monkeypatch.setattr(
        notifier,
        "_send_webhook",
        lambda note: calls.append(note.title) or True,
    )
    note = Notification(title="Daily", body="ok", kind="review")
    first = notifier.send(note)
    assert first and first.status == "SENT"
    # 第二次是幂等命中：仍然报告已送达，但不会再发一遍
    second = notifier.send(note)
    assert second and second.status == "SENT"
    assert calls == ["Daily"]


def test_notifier_fails_closed_without_external_authorization(
    tmp_path,
    monkeypatch,
):
    notifier = DiscordNotifier(
        bot_token="",
        channel_id="",
        webhook_url="https://discord.test/webhook",
        external_send_enabled=False,
        audit_db_path=str(tmp_path / "trade.duckdb"),
    )
    calls = []
    monkeypatch.setattr(
        notifier,
        "_send_webhook",
        lambda note: calls.append(note.title) or True,
    )
    outcome = notifier.send(Notification(title="Daily", body="ok", kind="review"))
    assert not outcome
    assert outcome.status == "BLOCKED"
    assert calls == []
