"""盘中信号播报：填掉开盘到收盘之间的 6.5 小时空白。

SignalPublisher 此前全仓库没有一处实例化，READY/ENTERED/EXIT/CLOSED 这些
"现在要不要动手"的判断全部只落库、不到人眼前。
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from trader.models import Side
from trader.signal_reports import (
    PUBLISHABLE_STATES,
    SignalPublisher,
    SignalReport,
    SignalState,
    SignalStore,
)

NOW = datetime(2026, 8, 3, 15, 0, tzinfo=timezone.utc)


class _Notifier:
    def __init__(self):
        self.sent = []

    def send(self, note):
        self.sent.append(note)
        return True


def _report(state=SignalState.READY, *, signal_id="sig-1", version=1, symbol="NVDA"):
    return SignalReport(
        signal_id=signal_id,
        version=version,
        symbol=symbol,
        state=state,
        side=Side.BUY,
        strategy="test",
        timeframe="5m",
        market_regime="neutral",
        market_price=101.0,
        market_data_at=NOW,
        generated_at=NOW,
        valid_until=NOW + timedelta(hours=6),
        entry_low=100.0,
        entry_high=102.0,
        chase_limit=103.0,
        stop_loss=98.0,
        take_profit=110.0,
        risk_reward=2.5,
        model_weight_pct=5.0,
        model_risk_pct=0.5,
        ai_score=78.0,
        ai_run_id="run-1",
        ai_contributors=["quant"],
        decision_id="dec-1",
        plan_id="plan-1",
    )


@pytest.fixture
def store(tmp_path):
    return SignalStore(str(tmp_path / "signals.duckdb"))


# ── 播报范围 ─────────────────────────────────────────────────────────────────


def test_actionable_states_are_published():
    """每一条都对应一个"现在要不要动手"的判断。"""
    for state in (
        SignalState.READY,
        SignalState.ENTERED,
        SignalState.EXIT,
        SignalState.CLOSED,
    ):
        assert state in PUBLISHABLE_STATES


def test_hold_is_not_published():
    """HOLD 是持仓心跳，每轮都可能重复，7 个标的会把频道变成噪音流。"""
    assert SignalState.HOLD not in PUBLISHABLE_STATES


def test_invalidated_is_not_published():
    """"没等到就过期了"是事后信息，归收盘报告的未成交计划一节。"""
    assert SignalState.INVALIDATED not in PUBLISHABLE_STATES


def test_publisher_skips_non_actionable_states(store):
    notifier = _Notifier()
    publisher = SignalPublisher(store, notifier)
    assert publisher.publish(_report(SignalState.HOLD)) is True
    assert notifier.sent == []


# ── 幂等 ─────────────────────────────────────────────────────────────────────


def test_same_version_is_published_once(store):
    notifier = _Notifier()
    publisher = SignalPublisher(store, notifier)
    report = _report()
    publisher.publish(report)
    publisher.publish(report)
    assert len(notifier.sent) == 1


def test_each_version_gets_its_own_message(store):
    """READY → ENTERED 是两个不同的行动信号，不能被当成重复。"""
    notifier = _Notifier()
    publisher = SignalPublisher(store, notifier)
    publisher.publish(_report(SignalState.READY, version=1))
    publisher.publish(_report(SignalState.ENTERED, version=2))
    assert len(notifier.sent) == 2


def test_message_carries_a_versioned_dedupe_identity():
    from trader.daily_discord import build_signal_report_message

    note = build_signal_report_message(_report(version=3))
    assert note.dedupe_key == "signal:sig-1:3"


# ── pending_publications ─────────────────────────────────────────────────────


def test_intermediate_states_are_not_skipped(store):
    """recent() 每个信号只给最新版本，READY 会被 ENTERED 覆盖掉——
    而 READY 恰恰是最有行动价值的一条。"""
    store._insert(_report(SignalState.READY, version=1))
    store._insert(_report(SignalState.ENTERED, version=2))

    pending = store.pending_publications(since=NOW - timedelta(minutes=30))
    states = {r.state for r in pending}
    assert SignalState.READY in states, "READY 被漏掉了"
    assert SignalState.ENTERED in states


def test_backfill_window_blocks_stale_events(store):
    """推送中断几天后恢复，不该把积压的历史事件倾泻到频道里。"""
    old = _report(SignalState.READY, signal_id="old")
    old.generated_at = NOW - timedelta(days=3)
    store._insert(old)

    pending = store.pending_publications(since=NOW - timedelta(minutes=30))
    assert all(r.signal_id != "old" for r in pending)


def test_published_events_stop_showing_up_as_pending(store):
    notifier = _Notifier()
    publisher = SignalPublisher(store, notifier)
    store._insert(_report(SignalState.READY))

    first = publisher.publish_pending(since=NOW - timedelta(minutes=30))
    second = publisher.publish_pending(since=NOW - timedelta(minutes=30))
    assert first == 1
    assert second == 0
    assert len(notifier.sent) == 1


def test_failed_delivery_stays_pending_for_retry(store):
    """投递失败的事件下一轮还要再试——否则一次网络抖动就永久丢一条信号。"""

    class _Flaky:
        def __init__(self):
            self.calls = 0

        def send(self, note):
            self.calls += 1
            return self.calls > 1  # 第一次失败

    notifier = _Flaky()
    publisher = SignalPublisher(store, notifier)
    store._insert(_report(SignalState.READY))

    assert publisher.publish_pending(since=NOW - timedelta(minutes=30)) == 0
    assert publisher.publish_pending(since=NOW - timedelta(minutes=30)) == 1
