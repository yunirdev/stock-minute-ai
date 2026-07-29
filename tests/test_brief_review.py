from __future__ import annotations


def test_evaluate_bullish_direction_call_scores_aligned_day():
    from trader.brief_review import evaluate_direction_call, format_brief_call_review

    review = evaluate_direction_call(
        bias="偏多",
        session_open=100.0,
        session_close=101.0,
        session_high=102.0,
        session_low=99.7,
    )

    assert review.bias == "bullish"
    assert review.verdict == "方向判断有效"
    assert review.score == 85
    assert review.session_return_pct == 1.0
    assert "顺向空间" in format_brief_call_review(review)


def test_evaluate_bearish_direction_call_marks_wrong_way_day():
    from trader.brief_review import evaluate_direction_call

    review = evaluate_direction_call(
        bias="偏空",
        session_open=100.0,
        session_close=101.0,
        session_high=101.5,
        session_low=99.5,
    )

    assert review.bias == "bearish"
    assert review.verdict == "方向判断失效"
    assert review.score == 30


def test_evaluate_neutral_direction_call_rewards_chop():
    from trader.brief_review import evaluate_direction_call

    review = evaluate_direction_call(
        bias="中性",
        session_open=100.0,
        session_close=100.1,
        session_high=100.3,
        session_low=99.8,
    )

    assert review.bias == "neutral"
    assert review.verdict == "中性判断有效"
    assert review.score == 80
