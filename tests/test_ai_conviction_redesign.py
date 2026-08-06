"""AI 分数体系重构：把"方向"（recommendation）和"强度"（conviction）拆开，
所有"够不够格"的判断都比方向无关的 conviction，不再直接比双极的 ai_score。

背景：ai_score 是双极量表（50=中性，BUY 落在 [50,100]，SELL 落在 [0,50]）。
选股/风控门槛原来直接拿这个数字跟一个 >=55 的阈值比，导致 SELL 无论 AI 信心
多足，分数上限就是 50，永远过不了门槛——AI 判断该平仓/该做空的信号在选股这
一步就被结构性清空了，从未真正驱动过任何一次平仓。
"""
from datetime import datetime, timezone

from trader.ai.safety import (
    AIScorePolicy,
    AIScoreSnapshot,
    AIScoreValidator,
    conviction_of,
)
from trader.models import Bar
from trader.selection import AICandidateSelector

NOW = datetime(2026, 8, 5, 15, 0, tzinfo=timezone.utc)


def _snapshot(**overrides) -> AIScoreSnapshot:
    base = dict(
        symbol="AAPL",
        score=85.0,
        created_at=NOW,
        run_id="research-1",
        provider="tradingagents",
        model="model-v1",
        source="daily_research",
        recommendation="BUY",
        confidence=0.7,
    )
    base.update(overrides)
    return AIScoreSnapshot(**base)


# ---------------------------------------------------------------------------
# conviction_of()
# ---------------------------------------------------------------------------

def test_conviction_prefers_explicit_confidence_over_derived_score():
    snap = _snapshot(score=85.0, confidence=0.4)
    assert conviction_of(snap) == 0.4


def test_conviction_falls_back_to_bipolar_inverse_for_buy():
    snap = _snapshot(score=85.0, confidence=None)
    assert conviction_of(snap) == 0.7  # (85-50)/50


def test_conviction_falls_back_to_bipolar_inverse_for_sell():
    # 这正是原来那个 bug 的核心：SELL 信心越足，双极 score 越低（不是越高）。
    snap = _snapshot(score=15.0, recommendation="SELL", confidence=None)
    assert conviction_of(snap) == 0.7  # (50-15)/50 == 0.7，跟同等信心的 BUY 相等


def test_conviction_none_when_nothing_usable():
    snap = _snapshot(score=None, confidence=None)
    assert conviction_of(snap) is None


# ---------------------------------------------------------------------------
# AIScoreValidator：方向无关的门槛
# ---------------------------------------------------------------------------

def test_validator_passes_high_conviction_sell_not_just_buy():
    policy = AIScorePolicy(min_ai_score=65.0, max_age_minutes=30.0)
    sell = _snapshot(
        score=7.5, recommendation="SELL", confidence=0.85,
    )
    result = AIScoreValidator(policy, lambda: NOW).validate(sell)
    assert result.valid, result.reason_code
    assert result.confidence == 0.85


def test_validator_rejects_low_conviction_regardless_of_direction():
    policy = AIScorePolicy(min_ai_score=65.0, max_age_minutes=30.0)
    weak_sell = _snapshot(score=40.0, recommendation="SELL", confidence=0.2)
    result = AIScoreValidator(policy, lambda: NOW).validate(weak_sell)
    assert not result.valid
    assert result.reason_code == "AI_SCORE_BELOW_THRESHOLD"


# ---------------------------------------------------------------------------
# AICandidateSelector：选股这一步不再系统性清空 SELL
# ---------------------------------------------------------------------------

def _bars(n=10, price=100.0):
    return [
        Bar(symbol="AAPL", timestamp=NOW, open=price, high=price + 1,
            low=price - 1, close=price, volume=1000)
        for _ in range(n)
    ]


def test_selector_admits_high_conviction_sell_candidate():
    selector = AICandidateSelector(confirm_window=5)
    snapshot = _snapshot(
        symbol="AAPL", score=7.5, recommendation="SELL", confidence=0.85,
    )
    candidates = selector.select(
        ai_scores={"AAPL": snapshot},
        bars={"AAPL": _bars(price=90.0)},  # 现价低于均线，跟 SELL 方向一致
        score_threshold=55.0,
    )
    assert len(candidates) == 1
    assert candidates[0].symbol == "AAPL"


def test_selector_still_rejects_low_conviction_sell():
    selector = AICandidateSelector(confirm_window=5)
    snapshot = _snapshot(
        symbol="AAPL", score=45.0, recommendation="SELL", confidence=0.1,
    )
    candidates = selector.select(
        ai_scores={"AAPL": snapshot},
        bars={"AAPL": _bars(price=90.0)},
        score_threshold=55.0,
    )
    assert candidates == []
