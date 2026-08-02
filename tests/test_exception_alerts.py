"""系统故障必须送到人眼前，且不能把频道刷爆。

在这之前，引擎的故障只写日志：watchdog 每轮都在喊"心跳超时 194222s"，
风控熔断和 broker 断连也记录得很完整，但没有任何一条会到达读者。
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from trader.daily_discord import build_exception_message
from trader.models import Notification
from trader.notify import DiscordNotifier

#: 取整点，这样下面的"同一小时内"偏移量一眼就能看出没跨桶。
NOW = datetime(2026, 8, 3, 14, 0, tzinfo=timezone.utc)


def _message(code="BROKER_SNAPSHOT_FAILED", at=NOW) -> Notification:
    return build_exception_message(
        error_code=code,
        summary="无法从券商读取权益/持仓",
        evidence_refs=["tick=42"],
        occurred_at=at,
    )


def test_exception_carries_code_time_and_evidence():
    note = _message()
    assert "BROKER_SNAPSHOT_FAILED" in note.title
    assert "2026-08-03" in note.body
    assert "tick=42" in note.body
    assert note.kind == "alert"


def test_urgent_prefix_survives_single_channel():
    """单频道下 🚨 是唯一的紧急度过滤手段，读者靠它扫描。"""
    assert _message().title.startswith("🚨")


def test_same_error_within_the_hour_is_throttled(tmp_path, monkeypatch):
    """引擎 30 秒一轮，一个持续故障一小时能复现 120 次。"""
    notifier = DiscordNotifier(
        bot_token="",
        channel_id="",
        webhook_url="https://discord.test/webhook",
        external_send_enabled=True,
        audit_db_path=str(tmp_path / "audit.duckdb"),
    )
    sent = []
    monkeypatch.setattr(
        notifier, "_send_webhook", lambda note: sent.append(note.title) or True
    )

    # 14:00 ~ 14:59，全部落在同一个小时桶里
    for seconds in (0, 30, 60, 1800, 3540):
        notifier.send(_message(at=NOW + timedelta(seconds=seconds)))

    assert len(sent) == 1, "同一小时内的同一个错误码应该只推一条"


def test_next_hour_alerts_again(tmp_path, monkeypatch):
    """节流不是永久静音——故障跨过整点仍要再响一次。

    用的是固定小时窗口而不是滑动窗口，所以 14:59 报过一次后 15:00 会再报，
    间隔可能只有一分钟。对持续性故障来说，多提醒一次远好过漏报。
    """
    notifier = DiscordNotifier(
        bot_token="",
        channel_id="",
        webhook_url="https://discord.test/webhook",
        external_send_enabled=True,
        audit_db_path=str(tmp_path / "audit.duckdb"),
    )
    sent = []
    monkeypatch.setattr(
        notifier, "_send_webhook", lambda note: sent.append(note.title) or True
    )
    notifier.send(_message(at=NOW))
    notifier.send(_message(at=NOW + timedelta(hours=1)))
    assert len(sent) == 2


def test_different_codes_are_not_throttled_together(tmp_path, monkeypatch):
    notifier = DiscordNotifier(
        bot_token="",
        channel_id="",
        webhook_url="https://discord.test/webhook",
        external_send_enabled=True,
        audit_db_path=str(tmp_path / "audit.duckdb"),
    )
    sent = []
    monkeypatch.setattr(
        notifier, "_send_webhook", lambda note: sent.append(note.title) or True
    )
    notifier.send(_message(code="RISK_HALTED"))
    notifier.send(_message(code="MARKET_DATA_UNAVAILABLE"))
    assert len(sent) == 2


def test_research_failure_routes_through_on_error_callback():
    """研究批次的故障以前被吞在日志里。"""
    from trader.daily_runtime_support import DailyRuntimeSupport

    seen = []

    class _Config:
        daily_research_enabled = True
        daily_research_db = ":memory:"
        ai_score_db = ":memory:"
        timeframe = "5m"

    support = DailyRuntimeSupport.__new__(DailyRuntimeSupport)
    support._on_error = lambda code, summary: seen.append((code, summary))
    support._notify_error("DAILY_RESEARCH_ALL_FAILED", "全部标的失败")
    assert seen == [("DAILY_RESEARCH_ALL_FAILED", "全部标的失败")]


def test_alert_failure_never_breaks_the_caller():
    """告警链路自己出问题，不能反过来打断交易主循环。"""
    from trader.daily_runtime_support import DailyRuntimeSupport

    def _boom(code, summary):
        raise RuntimeError("discord down")

    support = DailyRuntimeSupport.__new__(DailyRuntimeSupport)
    support._on_error = _boom
    support._notify_error("X", "y")  # 不抛异常即通过
