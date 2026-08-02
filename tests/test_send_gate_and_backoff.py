"""外发总闸 + 限流退避。

两个都是"看起来有、实际没有"的机制：
- DISCORD_EXTERNAL_SEND_ENABLED 被五个调用点各自硬编码 True 绕过，而
  .env.example 里写着 false，配置文件在描述一件和实际相反的事。
- 429 限流和 DNS 失败走同一条路径、同样返回 False，而 FAILED 在投递层是终
  态：一次限流就等于永久丢一条消息。
"""
from __future__ import annotations

import urllib.error
import urllib.request

import pytest

from trader.models import Notification
from trader.notify import DiscordNotifier, _external_send_allowed, _retry_after_seconds


def _notifier(tmp_path, **kwargs):
    return DiscordNotifier(
        bot_token="",
        channel_id="",
        webhook_url="https://discord.test/webhook",
        audit_db_path=str(tmp_path / "audit.duckdb"),
        **kwargs,
    )


# ── 外发总闸 ─────────────────────────────────────────────────────────────────


def test_gate_defaults_to_allowing(monkeypatch):
    """实际部署的 .env 里没有这一项，默认拦截会让升级后推送全部停掉。"""
    monkeypatch.delenv("DISCORD_EXTERNAL_SEND_ENABLED", raising=False)
    assert _external_send_allowed() is True


@pytest.mark.parametrize("value", ["false", "FALSE", "0", "no", "off", " off "])
def test_gate_blocks_when_explicitly_disabled(monkeypatch, value):
    monkeypatch.setenv("DISCORD_EXTERNAL_SEND_ENABLED", value)
    assert _external_send_allowed() is False


@pytest.mark.parametrize("value", ["true", "1", "yes", "on", ""])
def test_gate_allows_for_anything_else(monkeypatch, value):
    monkeypatch.setenv("DISCORD_EXTERNAL_SEND_ENABLED", value)
    assert _external_send_allowed() is True


def test_gate_actually_stops_a_send(tmp_path, monkeypatch):
    """关掉总闸后，连引擎自己的推送也发不出去——这正是它该有的效果。"""
    monkeypatch.setenv("DISCORD_EXTERNAL_SEND_ENABLED", "false")
    notifier = _notifier(tmp_path)
    sent = []
    monkeypatch.setattr(
        notifier, "_send_webhook", lambda note: sent.append(note.title) or True
    )
    outcome = notifier.send(Notification(title="晨报", body="x", kind="review"))
    assert outcome.status == "BLOCKED"
    assert sent == []


def test_credentials_no_longer_imply_authorization(tmp_path, monkeypatch):
    """旧逻辑里"传了凭据参数"就算已授权，连测试传空字符串都被当成授权。"""
    monkeypatch.setenv("DISCORD_EXTERNAL_SEND_ENABLED", "false")
    notifier = DiscordNotifier(
        bot_token="tok",
        channel_id="123",
        webhook_url="",
        audit_db_path=str(tmp_path / "a.duckdb"),
    )
    assert notifier._external_send_enabled is False


def test_no_production_call_site_hardcodes_authorization():
    """五处硬编码 external_send_enabled=True 让总闸形同虚设。"""
    from pathlib import Path

    for name in (
        "trader/morning_brief.py",
        "trader/runtime.py",
        "trader/manual_push.py",
        "trader/monitor_nice.py",
        "trader/teams/maintenance.py",
    ):
        source = Path(name).read_text(encoding="utf-8")
        # 只看真正的代码：注释里解释"以前为什么这么写"是允许的
        for lineno, line in enumerate(source.splitlines(), start=1):
            if line.strip().startswith("#"):
                continue
            assert "external_send_enabled=True" not in line, (
                f"{name}:{lineno} 仍在绕过总闸"
            )


# ── 限流退避 ─────────────────────────────────────────────────────────────────


class _Resp:
    status = 204

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _http_error(code, retry_after=None):
    headers = {}
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError(
        "https://discord.test", code, "err", headers, None
    )


def test_retry_after_is_read_from_the_header():
    assert _retry_after_seconds(_http_error(429, "2.5")) == 2.5


def test_retry_after_falls_back_when_header_is_missing_or_junk():
    assert _retry_after_seconds(_http_error(429)) == 1.0
    assert _retry_after_seconds(_http_error(429, "not-a-number")) == 1.0


def test_retry_after_is_capped():
    """Discord 偶尔给很长的冷却，但把交易引擎的 tick 卡在那里毫无意义。"""
    assert _retry_after_seconds(_http_error(429, "600")) <= 10.0


def test_rate_limited_send_is_retried(tmp_path, monkeypatch):
    notifier = _notifier(tmp_path, external_send_enabled=True)
    monkeypatch.setattr(notifier, "_throttle", lambda: None)
    monkeypatch.setattr("time.sleep", lambda _s: None)

    calls = []

    def _urlopen(req, timeout=None):
        calls.append(1)
        if len(calls) == 1:
            raise _http_error(429, "0.1")
        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    assert notifier.send(Notification(title="计划", body="x", kind="plan")).status == "SENT"
    assert len(calls) == 2, "429 之后应该重试一次"


def test_server_errors_are_retried(tmp_path, monkeypatch):
    notifier = _notifier(tmp_path, external_send_enabled=True)
    monkeypatch.setattr(notifier, "_throttle", lambda: None)
    monkeypatch.setattr("time.sleep", lambda _s: None)

    calls = []

    def _urlopen(req, timeout=None):
        calls.append(1)
        if len(calls) < 3:
            raise _http_error(503)
        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    assert notifier.send(Notification(title="计划", body="x", kind="plan")).status == "SENT"
    assert len(calls) == 3


def test_client_errors_are_not_retried(tmp_path, monkeypatch):
    """400 是我们自己把请求构造错了，重试多少次都一样。"""
    notifier = _notifier(tmp_path, external_send_enabled=True)
    monkeypatch.setattr(notifier, "_throttle", lambda: None)
    monkeypatch.setattr("time.sleep", lambda _s: None)

    calls = []

    def _urlopen(req, timeout=None):
        calls.append(1)
        raise _http_error(400)

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    assert notifier.send(Notification(title="计划", body="x", kind="plan")).status == "FAILED"
    assert len(calls) == 1


def test_retries_give_up_eventually(tmp_path, monkeypatch):
    notifier = _notifier(tmp_path, external_send_enabled=True)
    monkeypatch.setattr(notifier, "_throttle", lambda: None)
    monkeypatch.setattr("time.sleep", lambda _s: None)

    calls = []

    def _urlopen(req, timeout=None):
        calls.append(1)
        raise _http_error(429, "0.1")

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    assert notifier.send(Notification(title="计划", body="x", kind="plan")).status == "FAILED"
    assert len(calls) == 3


def test_throttle_spaces_out_a_burst(tmp_path, monkeypatch):
    """晨报四条背靠背发出，实测一秒内打完，很容易撞上频道限流。"""
    import trader.notify as notify_mod

    notifier = _notifier(tmp_path, external_send_enabled=True)
    monkeypatch.setattr(notify_mod, "_MIN_SEND_INTERVAL", 0.05)

    slept = []
    monkeypatch.setattr("time.sleep", lambda s: slept.append(s))
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: _Resp())

    for i in range(3):
        notifier.send(
            Notification(title=f"第{i}条", body="x", kind="plan", dedupe_key=f"k{i}")
        )
    assert any(s > 0 for s in slept), "连续发送之间应该有间隔"
