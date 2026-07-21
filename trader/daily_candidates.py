from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

import pandas as pd


_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_STORE = _ROOT / "conf" / "daily_candidates.json"

_ANCHORS = {"SPY", "QQQ"}
_MEGA_CAP = {"AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO"}
_BROAD_ETFS = {
    "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "IVV", "RSP",
    "XLK", "XLC", "XLY", "XLF", "XLE", "XLI", "XLV", "XLB", "XLP", "XLU", "XLRE",
}


@dataclass
class DailyCandidate:
    symbol: str
    rank: int
    score: float
    status: str
    source_quality_score: float
    ai_score: Optional[float]
    tactical_score: Optional[float]
    data_confidence: str
    reasons: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    as_of: str = ""


def build_daily_candidates(
    universe: Iterable[str],
    timeframe: str = "5m",
    ai_db_path: Optional[str] = None,
    limit: int = 12,
    include_anchors: bool = True,
    progress_callback: Optional[Callable[[dict], None]] = None,
) -> list[DailyCandidate]:
    symbols = _normalize_universe(universe, include_anchors=include_anchors)
    ai_scores = _load_ai_scores(ai_db_path)

    rows = []
    now_s = datetime.now(timezone.utc).isoformat()
    _progress(progress_callback, 0, len(symbols), "开始计算每日决策池")
    for idx, symbol in enumerate(symbols, start=1):
        source_score, source_reasons, source_risks = _source_quality(symbol)
        tactical = _tactical_score(symbol, timeframe)
        ai_score = ai_scores.get(symbol)

        score, confidence, reasons, risks = _combine_scores(
            symbol=symbol,
            source_score=source_score,
            ai_score=ai_score,
            tactical=tactical,
            source_reasons=source_reasons,
            source_risks=source_risks,
        )
        rows.append(DailyCandidate(
            symbol=symbol,
            rank=0,
            score=round(score, 1),
            status=_status_for(score, risks, symbol),
            source_quality_score=round(source_score, 1),
            ai_score=round(ai_score, 1) if ai_score is not None else None,
            tactical_score=round(tactical["score"], 1) if tactical else None,
            data_confidence=confidence,
            reasons=reasons[:5],
            risk_flags=risks[:5],
            as_of=now_s,
        ))
        if idx == len(symbols) or idx % 10 == 0:
            _progress(progress_callback, idx, len(symbols), f"{symbol}: {rows[-1].status}")

    rows.sort(key=lambda row: (_status_priority(row.status), -row.score, row.symbol))
    selected = rows[: max(1, int(limit))]
    for idx, row in enumerate(selected, start=1):
        row.rank = idx
    _progress(progress_callback, len(symbols), len(symbols), f"选出 {len(selected)} 只")
    return selected


def save_daily_candidates(
    candidates: list[DailyCandidate],
    path: Path | str = _DEFAULT_STORE,
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "candidates": [asdict(candidate) for candidate in candidates],
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_daily_candidates(path: Path | str = _DEFAULT_STORE) -> list[DailyCandidate]:
    src = Path(path)
    if not src.exists():
        return []
    try:
        payload = json.loads(src.read_text(encoding="utf-8"))
        return [DailyCandidate(**item) for item in payload.get("candidates", [])]
    except Exception:
        return []


def daily_candidate_symbols(
    path: Path | str = _DEFAULT_STORE,
    statuses: Optional[set[str]] = None,
    limit: int = 8,
) -> list[str]:
    statuses = statuses or {"ENTRY_READY", "WAIT_TRIGGER", "WAIT_PULLBACK", "WAIT_BREAKOUT", "WATCH"}
    rows = [
        row for row in load_daily_candidates(path)
        if row.status in statuses
    ]
    rows.sort(key=lambda row: row.rank or 999)
    return [row.symbol for row in rows[:limit]]


def format_daily_candidates(candidates: list[DailyCandidate], max_rows: int = 10) -> str:
    if not candidates:
        return "今日候选池为空。"
    lines = []
    for row in candidates[:max_rows]:
        reason = "；".join(row.reasons[:2]) if row.reasons else "无明确理由"
        risk = f" 风险：{'；'.join(row.risk_flags[:2])}" if row.risk_flags else ""
        lines.append(
            f"{row.rank}. {row.symbol} {row.score:.1f} "
            f"[{row.status}] {row.data_confidence} - {reason}{risk}"
        )
    return "\n".join(lines)


def _normalize_universe(universe: Iterable[str], include_anchors: bool) -> list[str]:
    out = []
    for item in universe:
        for raw in str(item or "").split(","):
            symbol = raw.strip().upper()
            if symbol and symbol not in out:
                out.append(symbol)
    if include_anchors:
        for anchor in ("SPY", "QQQ"):
            if anchor not in out:
                out.insert(0, anchor)
    return out


def _load_ai_scores(ai_db_path: Optional[str]) -> dict[str, float]:
    if not ai_db_path:
        return {}
    try:
        from .ai.manager import get_composite_scores_from_db
        return get_composite_scores_from_db(ai_db_path)
    except Exception:
        return {}


def _source_quality(symbol: str) -> tuple[float, list[str], list[str]]:
    score = 55.0
    reasons = []
    risks = []

    if symbol in _ANCHORS:
        score = 62.0
        reasons.append("指数锚点，用于判断市场方向")
        risks.append("指数/ETF 不等同于个股机会")
    elif symbol in _MEGA_CAP:
        score += 15
        reasons.append("高流动性大型权重股")
    elif symbol in _BROAD_ETFS:
        score += 5
        reasons.append("高流动性 ETF")
        risks.append("ETF 更适合作为环境/对冲参考")
    else:
        reasons.append("来自当前候选 universe")

    if len(symbol) > 5 and "-" not in symbol:
        score -= 8
        risks.append("代码异常或非普通美股，需人工确认")

    return max(0.0, min(100.0, score)), reasons, risks


def _tactical_score(symbol: str, timeframe: str) -> Optional[dict]:
    df = _load_bars(symbol, timeframe)
    if df is None or len(df) < 40:
        return None

    df = _clean_bars(df)
    if len(df) < 40:
        return None

    close = df["close"]
    last = float(close.iloc[-1])
    ma20 = float(close.tail(20).mean())
    ma50 = float(close.tail(50).mean()) if len(close) >= 50 else ma20
    ret_1 = _safe_pct(last, float(close.iloc[-2]))
    ret_20 = _safe_pct(last, float(close.iloc[-21])) if len(close) >= 21 else 0.0
    vol_ratio = _volume_ratio(df)
    consensus = _consensus_score(symbol, timeframe)

    score = 50.0
    reasons = []
    risks = []

    if last >= ma20 >= ma50:
        score += 12
        reasons.append("价格在 20/50 均线上方")
    elif last < ma20 and last < ma50:
        score -= 12
        risks.append("价格低于 20/50 均线")

    if ret_20 > 3:
        score += min(12, ret_20 * 1.2)
        reasons.append(f"近 20 根动量 {ret_20:+.1f}%")
    elif ret_20 < -3:
        score += max(-12, ret_20 * 1.2)
        risks.append(f"近 20 根动量 {ret_20:+.1f}%")

    if vol_ratio is not None:
        if vol_ratio >= 1.4 and ret_1 > 0:
            score += 8
            reasons.append(f"放量上涨，量比 {vol_ratio:.1f}x")
        elif vol_ratio >= 1.4 and ret_1 < 0:
            score -= 8
            risks.append(f"放量下跌，量比 {vol_ratio:.1f}x")

    if consensus is not None:
        score = score * 0.55 + consensus * 0.45
        reasons.append(f"策略共识 {consensus:.0f}/100")

    return {
        "score": max(0.0, min(100.0, score)),
        "reasons": reasons,
        "risks": risks,
    }


def _load_bars(symbol: str, timeframe: str) -> pd.DataFrame:
    try:
        from .data_cache import get_bars
        return get_bars(symbol, timeframe)
    except Exception:
        return pd.DataFrame()


def _clean_bars(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    clean.columns = [str(col).lower() for col in clean.columns]
    required = {"close", "high", "low", "volume"}
    if not required.issubset(clean.columns):
        return pd.DataFrame()
    for col in required:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
    return clean.dropna(subset=["close", "high", "low"]).reset_index(drop=True)


def _consensus_score(symbol: str, timeframe: str) -> Optional[float]:
    try:
        from .selection import ConsensusSelector
        candidate = ConsensusSelector()._score(symbol, timeframe, datetime.now(timezone.utc))
        return candidate.score if candidate is not None else None
    except Exception:
        return None


def _safe_pct(value: float, base: float) -> float:
    return (value - base) / base * 100 if base else 0.0


def _volume_ratio(df: pd.DataFrame) -> Optional[float]:
    if "volume" not in df.columns or len(df) < 40:
        return None
    long_avg = float(df["volume"].mean())
    if long_avg <= 0:
        return None
    return float(df["volume"].tail(20).mean() / long_avg)


def _progress(
    callback: Optional[Callable[[dict], None]],
    current: int,
    total: int,
    message: str,
) -> None:
    if not callback:
        return
    try:
        callback({
            "current": int(current),
            "total": int(total),
            "message": message,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
    except Exception:
        pass


def _combine_scores(
    symbol: str,
    source_score: float,
    ai_score: Optional[float],
    tactical: Optional[dict],
    source_reasons: list[str],
    source_risks: list[str],
) -> tuple[float, str, list[str], list[str]]:
    reasons = list(source_reasons)
    risks = list(source_risks)
    weighted = source_score * 0.25
    total_weight = 0.25

    if ai_score is not None:
        weighted += ai_score * 0.35
        total_weight += 0.35
        reasons.append(f"AI 综合分 {ai_score:.1f}")
    else:
        risks.append("缺少最新 AI 综合分")

    if tactical is not None:
        tactical_score = float(tactical["score"])
        weighted += tactical_score * 0.40
        total_weight += 0.40
        reasons.extend(tactical.get("reasons", []))
        risks.extend(tactical.get("risks", []))
    else:
        risks.append("缺少本地 K 线或技术共识")

    score = weighted / total_weight if total_weight else 50.0
    if symbol in _ANCHORS:
        score = min(score, 72.0)

    confidence = _confidence_label(ai_score is not None, tactical is not None)
    return score, confidence, _unique(reasons), _unique(risks)


def _confidence_label(has_ai: bool, has_tactical: bool) -> str:
    if has_ai and has_tactical:
        return "高"
    if has_ai or has_tactical:
        return "中"
    return "低"


def _status_for(score: float, risks: list[str], symbol: str) -> str:
    if symbol in _ANCHORS:
        return "MARKET_ANCHOR"
    if _has_hard_risk(risks):
        return "BENCH" if score >= 50 else "AVOID_NOW"
    if score >= 75:
        return "ENTRY_READY"
    if score >= 68:
        return "WAIT_BREAKOUT"
    if score >= 60:
        return "WATCH"
    if score >= 50:
        return "BENCH"
    return "AVOID_NOW"


def _has_hard_risk(risks: list[str]) -> bool:
    hard_tokens = ("缺少本地 K 线", "代码异常")
    return any(any(token in risk for token in hard_tokens) for risk in risks)


def _status_priority(status: str) -> int:
    order = {
        "ENTRY_READY": 0,
        "WAIT_TRIGGER": 1,
        "WAIT_BREAKOUT": 2,
        "WATCH": 2,
        "BENCH": 3,
        "MARKET_ANCHOR": 4,
        "AVOID_NOW": 5,
    }
    return order.get(status, 9)


def _unique(items: list[str]) -> list[str]:
    out = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return out
