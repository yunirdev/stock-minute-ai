"""推送必须如实报告结果，且同一份报告一天只出现一次。

这两条都是从实际事故里长出来的：
- 监控台长期把"没配 Discord"显示成"✓ 已发送"。
- 2026-08-02 06:04:57 与 06:05:27，同一份晨报被推了两遍；四条里只有一条
  因为正文碰巧一字不差被内容哈希挡下，其余三条因为行情数字动了几位就被
  当成新消息放行了。
"""
from __future__ import annotations

from datetime import datetime, timezone

from trader.models import Notification
from trader.notify import DeliveryOutcome, DiscordNotifier, summarize


def _notifier(tmp_path, **kwargs):
    return DiscordNotifier(
        bot_token="",
        channel_id="",
        webhook_url=kwargs.pop("webhook_url", "https://discord.test/webhook"),
        external_send_enabled=kwargs.pop("external_send_enabled", True),
        audit_db_path=str(tmp_path / "audit.duckdb"),
        **kwargs,
    )


# ── 诚实性 ───────────────────────────────────────────────────────────────────


def test_unconfigured_send_is_not_reported_as_success(tmp_path):
    notifier = _notifier(tmp_path, webhook_url="")
    outcome = notifier.send(Notification(title="晨报", body="内容"))
    assert not outcome, "没配 Discord 却报告成功，监控台会显示假的『已发送』"
    assert outcome.status == "DRY_RUN"


def test_blocked_send_is_distinguishable_from_failure(tmp_path):
    notifier = _notifier(tmp_path, external_send_enabled=False)
    outcome = notifier.send(Notification(title="晨报", body="内容"))
    assert outcome.status == "BLOCKED"
    assert outcome.status != "FAILED", "未授权和发送失败要能分辨，处理方式不同"


def test_summarize_reports_the_worst_of_a_batch():
    """晨报四条里有一条没发出去，整体就不能报告成功。"""
    assert summarize([DeliveryOutcome("SENT")] * 4).status == "SENT"
    assert summarize(
        [DeliveryOutcome("SENT"), DeliveryOutcome("FAILED"), DeliveryOutcome("SENT")]
    ).status == "FAILED"
    assert summarize(
        [DeliveryOutcome("SENT"), DeliveryOutcome("DRY_RUN")]
    ).status == "DRY_RUN"
    assert summarize(
        [DeliveryOutcome("DRY_RUN"), DeliveryOutcome("BLOCKED")]
    ).status == "BLOCKED"


# ── 去重 ─────────────────────────────────────────────────────────────────────


def test_business_identity_blocks_the_real_duplicate(tmp_path, monkeypatch):
    """复现真实事故：同一天的晨报重算，正文因行情变动而不同。

    内容哈希会把它当成新消息放行；业务身份能挡住。
    """
    notifier = _notifier(tmp_path)
    sent = []
    monkeypatch.setattr(
        notifier, "_send_webhook", lambda note: sent.append(note.title) or True
    )

    first = Notification(
        title="🎯 08/02 周日 · 今日交易作战卡",
        body="SPY 747.03（+0.72%）",
        kind="review",
        dedupe_key="morning_brief:action:2026-08-02",
    )
    # 31 秒后重算：同一份报告，但价格动了
    second = Notification(
        title="🎯 08/02 周日 · 今日交易作战卡",
        body="SPY 747.11（+0.73%）",
        kind="review",
        dedupe_key="morning_brief:action:2026-08-02",
    )

    assert notifier.send(first).status == "SENT"
    assert notifier.send(second).status == "SENT"  # 幂等命中，不是错误
    assert sent == ["🎯 08/02 周日 · 今日交易作战卡"], "同一天的晨报被推了两遍"


def test_different_days_are_not_deduped(tmp_path, monkeypatch):
    notifier = _notifier(tmp_path)
    sent = []
    monkeypatch.setattr(
        notifier, "_send_webhook", lambda note: sent.append(note.body) or True
    )
    for day in ("2026-08-03", "2026-08-04"):
        notifier.send(
            Notification(
                title="晨报",
                body=day,
                kind="review",
                dedupe_key=f"morning_brief:action:{day}",
            )
        )
    assert len(sent) == 2


def test_content_drift_without_business_identity_still_conflicts(tmp_path, monkeypatch):
    """没声明业务身份时保持旧行为：内容哈希模式下同 key 不同内容仍是异常。"""
    from trader.discord_delivery import DiscordDeliveryStore

    class _Sender:
        def send(self, note):
            return True

    store = DiscordDeliveryStore(
        str(tmp_path / "a.duckdb"), sender=_Sender(), external_send_enabled=True
    )
    args = dict(
        message_kind="REVIEW",
        dedupe_key="fixed-key",
        dry_run=False,
        now=datetime(2026, 8, 3, tzinfo=timezone.utc),
    )
    store.deliver(Notification(title="a", body="a"), **args)
    import pytest

    with pytest.raises(ValueError, match="DEDUPE_CONFLICT"):
        store.deliver(Notification(title="b", body="b"), **args)


