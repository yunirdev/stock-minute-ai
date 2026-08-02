from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


class BriefCallStore:
    """记下晨报当天给的方向判断，供收盘时对账。

    闭环的前提。晨报每天都在给可证伪的判断——"中性震荡，先等开盘区间给方
    向"、"偏多，但只做回踩确认后的多头"——但判断一旦推送出去就没有任何东西
    记得它，收盘时也就无从谈起"今天早上说对了没有"。evaluate_direction_call
    早就写好了打分逻辑，却只能挂在一个要人手动选方向的按钮上，因为程序自己
    不知道早上说过什么。
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = str(db_path)
        self._init_db()

    def _connect(self):
        import duckdb

        return duckdb.connect(self.db_path)

    def _init_db(self) -> None:
        try:
            con = self._connect()
        except Exception as exc:  # noqa: BLE001 - 建表失败不该拖垮晨报
            logger.warning("晨报方向存储不可用: %s", exc)
            return
        try:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS morning_brief_calls (
                    trading_date TEXT PRIMARY KEY,
                    bias TEXT,
                    reasons TEXT,
                    created_at TIMESTAMPTZ
                )
                """
            )
            con.commit()
        finally:
            con.close()

    def record(
        self,
        *,
        trading_date: str,
        bias: str,
        reasons: Optional[list[str]] = None,
        at: Optional[datetime] = None,
    ) -> None:
        """记录当天的方向判断。同一天重复调用保留第一次的判断。

        保留第一次是刻意的：收盘要对账的是"开盘前说了什么"。手动重发晨报会
        用当时的行情重算出一个不同的方向，如果覆盖掉早上那次，等于让判断跟
        着结果走，复盘就没有意义了。
        """
        try:
            con = self._connect()
        except Exception as exc:  # noqa: BLE001
            logger.warning("晨报方向记录失败: %s", exc)
            return
        try:
            con.execute(
                """
                INSERT INTO morning_brief_calls VALUES (?,?,?,?)
                ON CONFLICT (trading_date) DO NOTHING
                """,
                [
                    trading_date,
                    str(bias or ""),
                    "；".join(reasons or []),
                    at or datetime.now(timezone.utc),
                ],
            )
            con.commit()
        except Exception as exc:  # noqa: BLE001
            logger.warning("晨报方向记录失败: %s", exc)
        finally:
            con.close()

    def get(self, trading_date: str) -> Optional[dict[str, Any]]:
        try:
            con = self._connect()
        except Exception as exc:  # noqa: BLE001
            logger.warning("晨报方向读取失败: %s", exc)
            return None
        try:
            row = con.execute(
                "SELECT trading_date, bias, reasons, created_at "
                "FROM morning_brief_calls WHERE trading_date=?",
                [trading_date],
            ).fetchone()
        except Exception as exc:  # noqa: BLE001
            logger.warning("晨报方向读取失败: %s", exc)
            return None
        finally:
            con.close()
        if row is None:
            return None
        return {
            "trading_date": str(row[0]),
            "bias": str(row[1] or ""),
            "reasons": [r for r in str(row[2] or "").split("；") if r],
            "created_at": row[3],
        }


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
    # 观望类措辞必须先判：晨报在重大事件日给出的是"事件前观望，不抢多空；
    # 事件后只跟随确认方向"，这句里"多"和"空"同时出现，按下面的子串顺序会
    # 先撞上"多"从而判成看多——把一个明确的观望建议记成了看涨，收盘对账时
    # 就会拿错误的方向去打分。
    if any(token in text for token in ("观望", "不抢", "中性", "震荡", "flat")):
        return "neutral"
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
