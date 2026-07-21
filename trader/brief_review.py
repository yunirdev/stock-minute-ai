from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class BriefCallReview:
    bias: str
    verdict: str
    score: int
    session_return_pct: float
    max_favorable_pct: Optional[float]
    max_adverse_pct: Optional[float]
    notes: list[str] = field(default_factory=list)


def evaluate_direction_call(
    bias: str,
    session_open: float,
    session_close: float,
    session_high: Optional[float] = None,
    session_low: Optional[float] = None,
    threshold_pct: float = 0.35,
) -> BriefCallReview:
    normalized = _normalize_bias(bias)
    session_return = _pct(session_close, session_open)
    max_favorable = _max_favorable_pct(normalized, session_open, session_high, session_low)
    max_adverse = _max_adverse_pct(normalized, session_open, session_high, session_low)

    verdict, score, notes = _score_call(
        normalized,
        session_return,
        max_favorable,
        max_adverse,
        threshold_pct,
    )
    return BriefCallReview(
        bias=normalized,
        verdict=verdict,
        score=score,
        session_return_pct=session_return,
        max_favorable_pct=max_favorable,
        max_adverse_pct=max_adverse,
        notes=notes,
    )


def format_brief_call_review(review: BriefCallReview) -> str:
    fav = "n/a" if review.max_favorable_pct is None else f"{review.max_favorable_pct:+.2f}%"
    adverse = "n/a" if review.max_adverse_pct is None else f"{review.max_adverse_pct:+.2f}%"
    note = "；".join(review.notes) if review.notes else "无补充说明"
    return (
        f"晨报方向复盘：{review.verdict}（{review.score}/100）；"
        f"收盘相对开盘 {review.session_return_pct:+.2f}%；"
        f"最大顺向 {fav}；最大逆向 {adverse}；{note}"
    )


def _normalize_bias(bias: str) -> str:
    text = str(bias or "").lower()
    if any(token in text for token in ("bull", "long", "多", "看多", "偏多")):
        return "bullish"
    if any(token in text for token in ("bear", "short", "空", "看空", "防守", "偏空")):
        return "bearish"
    return "neutral"


def _pct(value: float, base: float) -> float:
    if base <= 0:
        raise ValueError("base price must be positive")
    return (value - base) / base * 100


def _max_favorable_pct(
    bias: str,
    session_open: float,
    session_high: Optional[float],
    session_low: Optional[float],
) -> Optional[float]:
    if bias == "bullish" and session_high is not None:
        return _pct(session_high, session_open)
    if bias == "bearish" and session_low is not None:
        return -_pct(session_low, session_open)
    if bias == "neutral" and session_high is not None and session_low is not None:
        return max(abs(_pct(session_high, session_open)), abs(_pct(session_low, session_open)))
    return None


def _max_adverse_pct(
    bias: str,
    session_open: float,
    session_high: Optional[float],
    session_low: Optional[float],
) -> Optional[float]:
    if bias == "bullish" and session_low is not None:
        return _pct(session_low, session_open)
    if bias == "bearish" and session_high is not None:
        return -_pct(session_high, session_open)
    if bias == "neutral" and session_high is not None and session_low is not None:
        return max(abs(_pct(session_high, session_open)), abs(_pct(session_low, session_open)))
    return None


def _score_call(
    bias: str,
    session_return: float,
    max_favorable: Optional[float],
    max_adverse: Optional[float],
    threshold_pct: float,
) -> tuple[str, int, list[str]]:
    notes: list[str] = []

    if bias == "neutral":
        if abs(session_return) <= threshold_pct:
            return "中性判断有效", 80, ["市场收盘仍接近开盘"]
        return "中性判断偏保守", 55, ["市场走出明确方向"]

    expected_positive = bias == "bullish"
    aligned = session_return >= threshold_pct if expected_positive else session_return <= -threshold_pct
    wrong = session_return <= -threshold_pct if expected_positive else session_return >= threshold_pct

    if aligned:
        score = 85
        verdict = "方向判断有效"
    elif wrong:
        score = 30
        verdict = "方向判断失效"
    else:
        score = 60
        verdict = "方向判断中性"

    if max_favorable is not None and max_favorable >= threshold_pct:
        notes.append("盘中曾给出顺向空间")
    if max_adverse is not None and max_adverse <= -threshold_pct:
        notes.append("盘中逆向波动较大，纪律重要")
    if not notes:
        notes.append("波动不足，结论参考价值有限")
    return verdict, score, notes
