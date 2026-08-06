from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

import pandas as pd

from .daily_candidates import (
    DailyCandidate,
    save_daily_candidates,
)


_ROOT = Path(__file__).resolve().parents[1]
_STORE = _ROOT / "conf" / "selection_pools.json"
_DECISION_REPORT_STORE = _ROOT / "conf" / "decision_pool_report.json"
_BARS_DIR = _ROOT / "data" / "bars"

LONG_TERM = "long_term"
WEEKLY_FOCUS = "weekly_focus"
DAILY_DECISION = "daily_decision"

DECISION_MIN_SIZE = 3
DECISION_TARGET_SIZE = 5
DECISION_MAX_SIZE = 7
DECISION_STYLE_STANDARD = "standard"
DECISION_STYLE_AGGRESSIVE = "aggressive"

_ANCHORS = {"SPY", "QQQ"}
_MEGA_CAP = {"AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO"}
_BROAD_ETFS = {
    "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "IVV", "RSP",
    "XLK", "XLC", "XLY", "XLF", "XLE", "XLI", "XLV", "XLB", "XLP", "XLU", "XLRE",
}
_DECISION_ETFS = ("SPY", "QQQ", "IWM", "SMH", "XLK", "XLF", "TLT", "GLD")

_CURATED_LARGE_CAP = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "AVGO", "TSLA", "BRK.B",
    "LLY", "JPM", "V", "UNH", "MA", "NFLX", "XOM", "COST", "WMT", "HD",
    "PG", "JNJ", "ABBV", "BAC", "KO", "PM", "CRM", "ORCL", "AMD", "CSCO",
    "CVX", "WFC", "ABT", "MCD", "GE", "IBM", "MRK", "DIS", "NOW", "INTU",
    "QCOM", "TXN", "AMAT", "UBER", "CAT", "GS", "AXP", "ISRG", "TMO", "VZ",
    "MU", "LRCX", "PANW", "ADBE", "PEP", "NEE", "RTX", "SPGI", "BKNG", "PGR",
]


@dataclass
class PoolItem:
    symbol: str
    rank: int
    score: float
    status: str
    data_confidence: str
    layer: str
    component_scores: dict[str, float | None] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    as_of: str = ""


@dataclass
class PoolResult:
    layer: str
    updated_at: str
    source_size: int
    selected_size: int
    items: list[PoolItem]
    warnings: list[str] = field(default_factory=list)


@dataclass
class DecisionChange:
    symbol: str
    change: str
    decision_type: str
    direction: str
    score: float
    reasons: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)


@dataclass
class DecisionPoolReport:
    updated_at: str
    previous_symbols: list[str]
    current_symbols: list[str]
    decision_style: str = DECISION_STYLE_STANDARD
    added: list[DecisionChange] = field(default_factory=list)
    removed: list[DecisionChange] = field(default_factory=list)
    kept: list[DecisionChange] = field(default_factory=list)
    slot_usage: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def rebuild_selection_pipeline(
    source: Iterable[str] | str | None = None,
    *,
    long_limit: int = 100,
    daily_limit: int = DECISION_MAX_SIZE,
    decision_style: str = DECISION_STYLE_STANDARD,
    ai_db_path: Optional[str] = None,
    save: bool = True,
    download_missing_decision_etfs: bool = False,
    manual_symbols: Iterable[str] = (),
    progress_callback: Optional[Callable[[dict], None]] = None,
) -> dict[str, PoolResult]:
    long_pool = build_long_term_pool(
        source,
        limit=long_limit,
        ai_db_path=ai_db_path,
        progress_callback=progress_callback,
    )
    daily_pool = build_daily_decision_pool(
        pool_symbols_from_result(
            long_pool,
            statuses={"CORE", "WATCH", "RESEARCH"},
            limit=max(long_limit, DECISION_MAX_SIZE * 8),
        ) or _symbols(long_pool.items),
        limit=daily_limit,
        decision_style=decision_style,
        ai_db_path=ai_db_path,
        sync_daily_store=save,
        base_items=long_pool.items,
        download_missing_decision_etfs=download_missing_decision_etfs,
        manual_symbols=manual_symbols,
        progress_callback=progress_callback,
    )
    results = {
        LONG_TERM: long_pool,
        DAILY_DECISION: daily_pool,
    }
    if save:
        save_selection_pools(results)
    return results


def build_long_term_pool(
    source: Iterable[str] | str | None = None,
    *,
    limit: int = 100,
    ai_db_path: Optional[str] = None,
    progress_callback: Optional[Callable[[dict], None]] = None,
) -> PoolResult:
    symbols = build_source_universe(source)
    ai_scores = _load_ai_scores(ai_db_path)
    now_s = _now_s()
    items: list[PoolItem] = []
    warnings = []

    _progress(progress_callback, LONG_TERM, 0, len(symbols), "scoring long-term pool")
    for idx, symbol in enumerate(symbols, start=1):
        quality_score, quality_reasons, quality_risks = _quality_score(symbol)
        features = _feature_score(symbol, "1d", mode=LONG_TERM)
        ai_score = ai_scores.get(symbol)
        score, confidence, reasons, risks = _combine_weighted(
            parts={
                "quality": (quality_score, 0.30),
                "trend": ((features or {}).get("score"), 0.50),
                "ai": (ai_score, 0.20),
            },
            required_missing=[
                ("trend", features is None, "缺少日线趋势数据"),
            ],
        )
        reasons = quality_reasons + reasons + ((features or {}).get("reasons", []))
        risks = quality_risks + risks + ((features or {}).get("risks", []))
        if ai_score is not None:
            reasons.append(f"AI 综合分 {ai_score:.1f}")
        status = _long_status(score, risks)
        items.append(PoolItem(
            symbol=symbol,
            rank=0,
            score=round(score, 1),
            status=status,
            data_confidence=confidence,
            layer=LONG_TERM,
            component_scores={
                "quality": round(quality_score, 1),
                "trend": _round_or_none((features or {}).get("score")),
                "ai": _round_or_none(ai_score),
            },
            reasons=_unique(reasons)[:6],
            risk_flags=_unique(risks)[:6],
            as_of=now_s,
        ))
        if idx == len(symbols) or idx % 10 == 0:
            _progress(progress_callback, LONG_TERM, idx, len(symbols), f"{symbol}: {status}")

    items.sort(key=lambda row: (_status_order(LONG_TERM, row.status), -row.score, row.symbol))
    selected = items[: max(1, int(limit))]
    _rank(selected)
    if not symbols:
        warnings.append("没有可用候选源")
    _progress(progress_callback, LONG_TERM, len(symbols), len(symbols), f"selected {len(selected)}")
    return PoolResult(
        layer=LONG_TERM,
        updated_at=now_s,
        source_size=len(symbols),
        selected_size=len(selected),
        items=selected,
        warnings=warnings,
    )




def build_daily_decision_pool(
    source: Iterable[str] | str | None = None,
    *,
    limit: int = DECISION_MAX_SIZE,
    decision_style: str = DECISION_STYLE_STANDARD,
    ai_db_path: Optional[str] = None,
    sync_daily_store: bool = True,
    base_items: Optional[list[PoolItem]] = None,
    download_missing_decision_etfs: bool = False,
    manual_symbols: Iterable[str] = (),
    progress_callback: Optional[Callable[[dict], None]] = None,
) -> PoolResult:
    symbols = _normalize_symbols(source) or pool_symbols(
        LONG_TERM,
        statuses={"CORE", "WATCH", "RESEARCH"},
        limit=max(100, limit * 12),
    )
    if not symbols:
        symbols = build_source_universe(None)[: max(limit, 1)]
    symbols = _with_decision_etfs(symbols)
    manual = _normalize_symbols(list(manual_symbols)) if manual_symbols else []
    if manual:
        symbols = list(dict.fromkeys([*symbols, *manual]))
    if download_missing_decision_etfs:
        _ensure_decision_etf_bars(symbols, progress_callback=progress_callback)

    style = _normalize_decision_style(decision_style)
    cfg = _decision_style_config(style)
    min_size = min(cfg["min_size"], int(limit or DECISION_MAX_SIZE))
    max_size = max(min_size, min(DECISION_MAX_SIZE, int(limit or DECISION_MAX_SIZE)))
    ai_scores = _load_ai_scores(ai_db_path)
    base_scores = {item.symbol: item.score for item in (base_items or load_selection_pool(LONG_TERM).items)}
    previous = load_selection_pool(DAILY_DECISION)
    previous_symbols = [item.symbol for item in previous.items]
    now_s = _now_s()
    scored: list[PoolItem] = []

    _progress(progress_callback, DAILY_DECISION, 0, len(symbols), "scoring decision pool")
    for idx, symbol in enumerate(symbols, start=1):
        item = _score_decision_symbol(
            symbol,
            ai_score=ai_scores.get(symbol),
            base_score=base_scores.get(symbol),
            decision_style=style,
            now_s=now_s,
        )
        scored.append(item)
        if idx == len(symbols) or idx % 10 == 0:
            _progress(progress_callback, DAILY_DECISION, idx, len(symbols), f"{symbol}: {item.status}")

    selected = _select_decision_items(scored, min_size=min_size, max_size=max_size, decision_style=style)

    # 自选标的是用户明确要求观察的，不该被 AI 评分/名额挤掉——AI 淘汰的理由
    # 对自选没有意义，所以这里无条件补回，不受 min_score/max_size/类别名额限制。
    if manual:
        scored_by_symbol = {item.symbol: item for item in scored}
        selected_symbols = {item.symbol for item in selected}
        for symbol in manual:
            item = scored_by_symbol.get(symbol)
            if item is None or symbol in selected_symbols:
                continue
            item.reasons = _unique(["自选（用户手动追加，不受评分/名额限制）", *item.reasons])[:9]
            selected.append(item)
            selected_symbols.add(symbol)
        selected.sort(key=lambda row: (-row.score, row.symbol))

    _rank(selected)

    daily_rows = [_daily_candidate_from_pool_item(item) for item in selected]
    if sync_daily_store:
        save_daily_candidates(daily_rows)

    report = _build_decision_report(previous_symbols, selected, scored, now_s, decision_style=style)
    save_decision_pool_report(report)
    try:
        from .decision_trade_plans import build_decision_trade_plan_report
        build_decision_trade_plan_report(selected, decision_style=style, save=True)
    except Exception as exc:
        report.warnings.append(f"决策交易计划生成失败: {exc}")
        save_decision_pool_report(report)
    _progress(progress_callback, DAILY_DECISION, len(symbols), len(symbols), f"selected {len(selected)}")
    return PoolResult(
        layer=DAILY_DECISION,
        updated_at=now_s,
        source_size=len(symbols),
        selected_size=len(selected),
        items=selected,
        warnings=report.warnings,
    )


def _score_decision_symbol(
    symbol: str,
    *,
    ai_score: Optional[float],
    base_score: Optional[float],
    decision_style: str,
    now_s: str,
) -> PoolItem:
    profile = _decision_profile(symbol, decision_style=decision_style)
    quality_score, quality_reasons, quality_risks = _quality_score(symbol)
    setup_score = profile.get("score") if profile else None
    cfg = _decision_style_config(decision_style)
    score, confidence, reasons, risks = _combine_weighted(
        parts={
            "long_pool": (base_score, cfg["base_weight"]),
            "setup": (setup_score, cfg["setup_weight"]),
            "quality": (quality_score, cfg["quality_weight"]),
            "ai": (ai_score, cfg["ai_weight"]),
        },
        required_missing=[
            ("setup", profile is None, "缺少决策池所需日线数据"),
        ],
    )
    reasons = quality_reasons + reasons + list((profile or {}).get("reasons", []))
    risks = quality_risks + risks + list((profile or {}).get("risks", []))
    if base_score is not None:
        reasons.append(f"长期池分 {base_score:.1f}")
    if ai_score is not None:
        reasons.append(f"AI 综合分 {ai_score:.1f}")

    decision_type = str((profile or {}).get("decision_type", "WATCH"))
    direction = str((profile or {}).get("direction", "WATCH"))
    status = _decision_status(score, risks)
    reasons = [f"类型 {decision_type}", f"方向 {direction}", f"风格 {decision_style}"] + reasons
    for text in list((profile or {}).get("trade_notes", [])):
        reasons.append(text)

    return PoolItem(
        symbol=symbol,
        rank=0,
        score=round(score, 1),
        status=status,
        data_confidence=confidence,
        layer=DAILY_DECISION,
        component_scores={
            "long_pool": _round_or_none(base_score),
            "setup": _round_or_none(setup_score),
            "quality": round(quality_score, 1),
            "ai": _round_or_none(ai_score),
        },
        reasons=_unique(reasons)[:9],
        risk_flags=_unique(risks)[:7],
        as_of=now_s,
    )


def _decision_profile(symbol: str, *, decision_style: str = DECISION_STYLE_STANDARD) -> Optional[dict]:
    df = _clean_bars(_load_bars(symbol, "1d"))
    if len(df) < 60:
        return None
    close = df["close"]
    volume = df["volume"] if "volume" in df.columns else pd.Series(dtype=float)
    last = float(close.iloc[-1])
    ma20 = float(close.tail(20).mean())
    ma50 = float(close.tail(50).mean())
    ma200 = float(close.tail(200).mean()) if len(close) >= 200 else ma50
    ret5 = _pct(last, float(close.iloc[-6])) if len(close) >= 6 else 0.0
    ret20 = _pct(last, float(close.iloc[-21])) if len(close) >= 21 else 0.0
    ret60 = _pct(last, float(close.iloc[-61])) if len(close) >= 61 else ret20
    high60 = float(close.tail(min(60, len(close))).max())
    drawdown = _pct(last, high60)
    vol20 = float(close.pct_change().tail(20).std() * 100) if len(close) >= 25 else 0.0
    vol_ratio = _volume_ratio(volume)
    cfg = _decision_style_config(decision_style)

    long_score = 50.0
    short_score = 45.0
    reversal_score = 42.0
    aggressive_score = 44.0
    reasons: list[str] = []
    risks: list[str] = []
    notes: list[str] = []

    if last > ma20 > ma50:
        long_score += 16
        aggressive_score += 10
        reasons.append("20/50 日结构偏多")
    elif last > ma50:
        long_score += 8
        aggressive_score += 5
        reasons.append("仍在 50 日均线上")
    else:
        short_score += 8
        risks.append("低于 50 日均线")

    if ma50 > ma200:
        long_score += 8
        reasons.append("50 日均线高于长期均线")
    elif last < ma50:
        short_score += 8

    if ret20 > 4:
        long_score += min(12, ret20 * 0.8)
        aggressive_score += min(18, ret20 * 1.0)
        reasons.append(f"20 日动量 {ret20:+.1f}%")
    elif ret20 < -6:
        short_score += min(14, abs(ret20) * 0.8)
        risks.append(f"20 日走弱 {ret20:+.1f}%")

    if ret60 > 8:
        long_score += min(10, ret60 * 0.35)
        aggressive_score += min(14, ret60 * 0.35)
        reasons.append(f"60 日趋势 {ret60:+.1f}%")
    elif ret60 < -10:
        short_score += min(12, abs(ret60) * 0.4)
        risks.append(f"60 日趋势 {ret60:+.1f}%")

    if drawdown < -15:
        reversal_score += min(22, abs(drawdown + 10) * 1.1)
        risks.append(f"距 60 日高点 {drawdown:.1f}%")
        if ret5 > 1:
            reversal_score += 8
            reasons.append("超跌后短线修复")

    if vol_ratio is not None and vol_ratio >= 1.25:
        long_score += 4 if ret5 >= 0 else 0
        short_score += 4 if ret5 < 0 else 0
        aggressive_score += 8 if ret5 >= 0 else 2
        reasons.append(f"近期量能 {vol_ratio:.1f}x")

    if vol20 > 5.0:
        risks.append("波动偏高，自动交易需降仓")
        long_score -= cfg["high_vol_penalty"]
        short_score -= max(2, cfg["high_vol_penalty"] - 1)
        reversal_score -= cfg["high_vol_penalty"] + 1
        aggressive_score -= max(0, cfg["high_vol_penalty"] - 5)
    if decision_style == DECISION_STYLE_AGGRESSIVE and ret5 > 2 and ret20 > 8:
        aggressive_score += 8
        reasons.append("小资金进攻：短线动量延续")

    scores = {
        "LONG_TREND": long_score,
        "SHORT_TREND": short_score,
        "REVERSAL": reversal_score,
    }
    if decision_style == DECISION_STYLE_AGGRESSIVE:
        scores["AGGRESSIVE_MOMENTUM"] = aggressive_score
    decision_type = max(scores, key=scores.get)
    direction = {
        "LONG_TREND": "LONG",
        "SHORT_TREND": "SHORT",
        "REVERSAL": "LONG",
        "AGGRESSIVE_MOMENTUM": "LONG",
    }.get(decision_type, "WATCH")
    score = scores[decision_type]
    if symbol in _BROAD_ETFS or symbol in _ANCHORS:
        decision_type = "ETF_MACRO"
        direction = "LONG" if last >= ma20 else "HEDGE"
        score = max(score, 62.0 if last >= ma50 else 56.0)
        reasons.append("ETF/宏观表达名额")

    if direction == "LONG":
        notes.append(f"触发参考：站稳 ${last:.2f} 或回踩 20 日线附近")
        notes.append(f"失效参考：跌破 ${ma20:.2f} 后重新评估")
        if decision_type == "AGGRESSIVE_MOMENTUM":
            notes.append("仓位提示：进攻候选单笔仓位应小于标准趋势候选")
    elif direction == "SHORT":
        notes.append(f"触发参考：反弹不过 20 日线 ${ma20:.2f}")
        notes.append(f"失效参考：重新站回 50 日线 ${ma50:.2f}")
    else:
        notes.append("作为市场表达或对冲参考，不强制开仓")

    return {
        "score": max(0.0, min(100.0, score)),
        "decision_type": decision_type,
        "direction": direction,
        "reasons": reasons,
        "risks": risks,
        "trade_notes": notes,
    }


def _select_decision_items(
    items: list[PoolItem],
    *,
    min_size: int,
    max_size: int,
    decision_style: str,
) -> list[PoolItem]:
    cfg = _decision_style_config(decision_style)
    viable = [
        item for item in items
        if item.status != "AVOID_NOW" and item.score >= cfg["min_score"] and item.symbol not in _ANCHORS
    ]
    viable.sort(key=lambda row: (-row.score, _decision_type(row), row.symbol))
    caps = cfg["caps"]
    selected: list[PoolItem] = []
    usage: dict[str, int] = {}
    for item in viable:
        kind = _decision_type(item)
        if usage.get(kind, 0) >= caps.get(kind, 2):
            continue
        selected.append(item)
        usage[kind] = usage.get(kind, 0) + 1
        if len(selected) >= max_size:
            break

    if len(selected) < min_size:
        selected_symbols = {item.symbol for item in selected}
        for item in viable:
            if item.symbol in selected_symbols:
                continue
            selected.append(item)
            selected_symbols.add(item.symbol)
            if len(selected) >= min_size:
                break

    selected.sort(key=lambda row: (-row.score, row.symbol))
    return selected[:max_size]


def _decision_status(score: float, risks: list[str]) -> str:
    if _has_hard_risk(risks):
        return "BENCH" if score >= 55 else "AVOID_NOW"
    if score >= 74:
        return "ENTRY_READY"
    if score >= 66:
        return "WAIT_TRIGGER"
    if score >= 58:
        return "WATCH"
    return "BENCH"


def _normalize_decision_style(style: str | None) -> str:
    raw = str(style or "").strip().lower()
    aliases = {
        "积极": DECISION_STYLE_AGGRESSIVE,
        "进攻": DECISION_STYLE_AGGRESSIVE,
        "小资金进攻": DECISION_STYLE_AGGRESSIVE,
        "aggressive": DECISION_STYLE_AGGRESSIVE,
        "standard": DECISION_STYLE_STANDARD,
        "标准": DECISION_STYLE_STANDARD,
        "稳健": DECISION_STYLE_STANDARD,
    }
    return aliases.get(raw, DECISION_STYLE_STANDARD)


def _decision_style_config(style: str) -> dict:
    if _normalize_decision_style(style) == DECISION_STYLE_AGGRESSIVE:
        return {
            "min_size": 5,
            "target_size": 6,
            "min_score": 50.0,
            "base_weight": 0.20,
            "setup_weight": 0.56,
            "quality_weight": 0.10,
            "ai_weight": 0.14,
            "high_vol_penalty": 2,
            "caps": {
                "AGGRESSIVE_MOMENTUM": 3,
                "LONG_TREND": 4,
                "SHORT_TREND": 2,
                "REVERSAL": 2,
                "ETF_MACRO": 2,
            },
        }
    return {
        "min_size": DECISION_MIN_SIZE,
        "target_size": DECISION_TARGET_SIZE,
        "min_score": 55.0,
        "base_weight": 0.28,
        "setup_weight": 0.46,
        "quality_weight": 0.14,
        "ai_weight": 0.12,
        "high_vol_penalty": 4,
        "caps": {
            "ETF_MACRO": 2,
            "LONG_TREND": 3,
            "SHORT_TREND": 2,
            "REVERSAL": 1,
            "AGGRESSIVE_MOMENTUM": 1,
        },
    }


def _with_decision_etfs(symbols: list[str]) -> list[str]:
    out = list(symbols)
    for symbol in _DECISION_ETFS:
        if symbol not in out:
            out.append(symbol)
    return out


def _ensure_decision_etf_bars(
    symbols: list[str],
    *,
    progress_callback: Optional[Callable[[dict], None]] = None,
) -> None:
    """Download missing daily history for the ETFs injected into the decision pool."""
    etfs = [symbol for symbol in symbols if symbol in _DECISION_ETFS]
    for idx, symbol in enumerate(etfs, start=1):
        if len(_clean_bars(_load_bars(symbol, "1d"))) >= 60:
            continue
        _progress(
            progress_callback,
            DAILY_DECISION,
            idx - 1,
            len(etfs),
            f"{symbol}: downloading missing daily bars",
        )
        _download_bars(symbol, "1d")


def _decision_type(item: PoolItem) -> str:
    for reason in item.reasons:
        if reason.startswith("类型 "):
            return reason.replace("类型 ", "", 1)
    return "WATCH"


def _decision_direction(item: PoolItem) -> str:
    for reason in item.reasons:
        if reason.startswith("方向 "):
            return reason.replace("方向 ", "", 1)
    return "WATCH"


def _daily_candidate_from_pool_item(item: PoolItem) -> DailyCandidate:
    return DailyCandidate(
        symbol=item.symbol,
        rank=item.rank,
        score=item.score,
        status=item.status,
        source_quality_score=float(item.component_scores.get("quality") or 0.0),
        ai_score=item.component_scores.get("ai"),
        tactical_score=item.component_scores.get("setup"),
        data_confidence=item.data_confidence,
        reasons=list(item.reasons),
        risk_flags=list(item.risk_flags),
        as_of=item.as_of,
    )


def _build_decision_report(
    previous_symbols: list[str],
    current_items: list[PoolItem],
    scored_items: list[PoolItem],
    now_s: str,
    decision_style: str = DECISION_STYLE_STANDARD,
) -> DecisionPoolReport:
    current_symbols = [item.symbol for item in current_items]
    current_by_symbol = {item.symbol: item for item in current_items}
    scored_by_symbol = {item.symbol: item for item in scored_items}
    added = [
        _decision_change(current_by_symbol[symbol], "added", ["新进入决策池，当前 setup 排名靠前"])
        for symbol in current_symbols
        if symbol not in previous_symbols
    ]
    kept = [
        _decision_change(current_by_symbol[symbol], "kept", ["核心条件仍成立，继续保留"])
        for symbol in current_symbols
        if symbol in previous_symbols
    ]
    removed: list[DecisionChange] = []
    current_floor = min((item.score for item in current_items), default=0.0)
    for symbol in previous_symbols:
        if symbol in current_symbols:
            continue
        item = scored_by_symbol.get(symbol)
        if item is None:
            removed.append(DecisionChange(
                symbol=symbol,
                change="removed",
                decision_type="UNKNOWN",
                direction="WATCH",
                score=0.0,
                reasons=["不在本轮长期池/候选输入中"],
                risk_flags=["stale_no_source"],
            ))
            continue
        reasons = ["被更高质量候选替代"] if item.score < current_floor else ["组合名额约束，暂时移出"]
        removed.append(_decision_change(item, "removed", reasons))

    slot_usage: dict[str, int] = {}
    for item in current_items:
        kind = _decision_type(item)
        slot_usage[kind] = slot_usage.get(kind, 0) + 1
    warnings = []
    if len(current_items) < DECISION_MIN_SIZE:
        warnings.append("高质量可交易候选不足，决策池未强行凑满 3 个")
    return DecisionPoolReport(
        updated_at=now_s,
        previous_symbols=previous_symbols,
        current_symbols=current_symbols,
        decision_style=decision_style,
        added=added,
        removed=removed,
        kept=kept,
        slot_usage=slot_usage,
        warnings=warnings,
    )


def _decision_change(item: PoolItem, change: str, extra_reasons: list[str]) -> DecisionChange:
    return DecisionChange(
        symbol=item.symbol,
        change=change,
        decision_type=_decision_type(item),
        direction=_decision_direction(item),
        score=item.score,
        reasons=_unique(extra_reasons + list(item.reasons))[:8],
        risk_flags=list(item.risk_flags)[:6],
    )


def save_decision_pool_report(
    report: DecisionPoolReport,
    path: Path | str = _DECISION_REPORT_STORE,
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8")


def load_decision_pool_report(path: Path | str = _DECISION_REPORT_STORE) -> DecisionPoolReport:
    src = Path(path)
    if not src.exists():
        return DecisionPoolReport(updated_at="", previous_symbols=[], current_symbols=[])
    try:
        payload = json.loads(src.read_text(encoding="utf-8"))
        return DecisionPoolReport(
            updated_at=str(payload.get("updated_at", "") or ""),
            previous_symbols=list(payload.get("previous_symbols", []) or []),
            current_symbols=list(payload.get("current_symbols", []) or []),
            decision_style=str(payload.get("decision_style", DECISION_STYLE_STANDARD) or DECISION_STYLE_STANDARD),
            added=[DecisionChange(**item) for item in payload.get("added", []) if isinstance(item, dict)],
            removed=[DecisionChange(**item) for item in payload.get("removed", []) if isinstance(item, dict)],
            kept=[DecisionChange(**item) for item in payload.get("kept", []) if isinstance(item, dict)],
            slot_usage=dict(payload.get("slot_usage", {}) or {}),
            warnings=list(payload.get("warnings", []) or []),
        )
    except Exception:
        return DecisionPoolReport(updated_at="", previous_symbols=[], current_symbols=[])


def save_selection_pools(results: dict[str, PoolResult], path: Path | str = _STORE) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": _now_s(),
        "pools": {
            layer: asdict(result)
            for layer, result in results.items()
        },
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_selection_pool(result: PoolResult, path: Path | str = _STORE) -> None:
    existing = load_selection_pools(path)
    existing[result.layer] = result
    save_selection_pools(existing, path)


def load_selection_pools(path: Path | str = _STORE) -> dict[str, PoolResult]:
    src = Path(path)
    if not src.exists():
        return {}
    try:
        payload = json.loads(src.read_text(encoding="utf-8"))
        pools = payload.get("pools", {})
        return {layer: _pool_result_from_dict(data) for layer, data in pools.items()}
    except Exception:
        return {}


def load_selection_pool(layer: str, path: Path | str = _STORE) -> PoolResult:
    pools = load_selection_pools(path)
    return pools.get(layer, PoolResult(layer=layer, updated_at="", source_size=0, selected_size=0, items=[]))


def pool_symbols(
    layer: str,
    *,
    statuses: Optional[set[str]] = None,
    limit: int = 20,
    path: Path | str = _STORE,
) -> list[str]:
    return pool_symbols_from_result(load_selection_pool(layer, path), statuses=statuses, limit=limit)


def pool_symbols_from_result(
    result: PoolResult,
    *,
    statuses: Optional[set[str]] = None,
    limit: int = 20,
) -> list[str]:
    rows = result.items
    if statuses:
        rows = [row for row in rows if row.status in statuses]
    rows = [row for row in rows if row.symbol not in _ANCHORS and row.status != "AVOID"]
    rows.sort(key=lambda row: row.rank or 999)
    return [row.symbol for row in rows[:limit]]


def decision_symbols(limit: int = 8, path: Path | str = _STORE) -> list[str]:
    preferred = pool_symbols(
        DAILY_DECISION,
        statuses={"ENTRY_READY", "WAIT_TRIGGER", "WAIT_BREAKOUT", "WATCH"},
        limit=limit,
        path=path,
    )
    if preferred:
        return preferred
    return pool_symbols(
        DAILY_DECISION,
        statuses={"BENCH", "ENTRY_READY", "WAIT_TRIGGER", "WAIT_BREAKOUT", "WATCH"},
        limit=limit,
        path=path,
    )


def build_source_universe(source: Iterable[str] | str | None = None, *, max_symbols: int = 3000) -> list[str]:
    explicit = _normalize_symbols(source)
    if explicit:
        return explicit[:max_symbols]

    symbols: list[str] = []
    try:
        from .market_scan import HOT, KEEP, WATCH, market_scan_symbols
        symbols.extend(market_scan_symbols(statuses={KEEP, WATCH, HOT}, limit=max_symbols))
    except Exception:
        pass
    if symbols:
        return _normalize_symbols(symbols)[:max_symbols]

    try:
        from .universe import get_universe
        for name in ("default", "mega_cap", "etf", "watchlist"):
            symbols.extend(get_universe(name))
    except Exception:
        pass
    symbols.extend(_CURATED_LARGE_CAP)
    symbols.extend(_cached_symbols())
    return _normalize_symbols(symbols)[:max_symbols]


def parse_symbol_text(text: str | None) -> list[str]:
    return _normalize_symbols(text)


def _feature_score(symbol: str, timeframe: str, mode: str) -> Optional[dict]:
    df = _load_bars(symbol, timeframe)
    clean = _clean_bars(df)
    min_bars = 80 if mode == LONG_TERM else 35
    if len(clean) < min_bars:
        return None

    close = clean["close"]
    volume = clean["volume"] if "volume" in clean.columns else pd.Series(dtype=float)
    last = float(close.iloc[-1])
    ma20 = float(close.tail(20).mean())
    ma50 = float(close.tail(50).mean())
    ma200 = float(close.tail(200).mean()) if len(close) >= 200 else ma50
    ret_5 = _pct(last, float(close.iloc[-6])) if len(close) >= 6 else 0.0
    ret_20 = _pct(last, float(close.iloc[-21])) if len(close) >= 21 else 0.0
    ret_60 = _pct(last, float(close.iloc[-61])) if len(close) >= 61 else ret_20
    ret_120 = _pct(last, float(close.iloc[-121])) if len(close) >= 121 else ret_60
    high_120 = float(close.tail(min(120, len(close))).max())
    drawdown = _pct(last, high_120)
    vol20 = float(close.pct_change().tail(20).std() * 100) if len(close) >= 25 else 0.0
    vol_ratio = _volume_ratio(volume)

    score = 50.0
    reasons: list[str] = []
    risks: list[str] = []

    if mode == LONG_TERM:
        if last > ma50:
            score += 8
            reasons.append("站上 50 日均线")
        else:
            score -= 8
            risks.append("低于 50 日均线")
        if last > ma200:
            score += 10
            reasons.append("长期趋势仍在 200 日均线上")
        else:
            score -= 10
            risks.append("低于长期均线")
        if ma50 > ma200:
            score += 7
            reasons.append("中长期均线结构偏强")
        if ret_60 > 8:
            score += min(12, ret_60 * 0.45)
            reasons.append(f"近 3 个月动量 {ret_60:+.1f}%")
        elif ret_60 < -8:
            score += max(-12, ret_60 * 0.45)
            risks.append(f"近 3 个月动量 {ret_60:+.1f}%")
        if ret_120 > 12:
            score += min(8, ret_120 * 0.20)
            reasons.append(f"近半年趋势 {ret_120:+.1f}%")
        if drawdown < -25:
            score -= 8
            risks.append(f"距阶段高点 {drawdown:.1f}%")
        if vol20 > 4.0:
            score -= 4
            risks.append("波动偏高")
    else:
        if last > ma20:
            score += 8
            reasons.append("站上 20 日均线")
        else:
            score -= 6
            risks.append("低于 20 日均线")
        if last > ma50:
            score += 6
            reasons.append("仍在 50 日均线上")
        if ret_5 > 1.5:
            score += min(10, ret_5 * 1.5)
            reasons.append(f"近 5 日动量 {ret_5:+.1f}%")
        elif ret_5 < -4:
            score += max(-12, ret_5 * 1.4)
            risks.append(f"近 5 日走弱 {ret_5:+.1f}%")
        if 0 < ret_20 < 12:
            score += min(8, ret_20 * 0.6)
            reasons.append(f"近 20 日趋势 {ret_20:+.1f}%")
        elif ret_20 >= 18:
            score -= 5
            risks.append("短线涨幅过快，等待回踩")
        elif ret_20 < -8:
            score -= 8
            risks.append(f"近 20 日趋势 {ret_20:+.1f}%")
        if vol_ratio is not None and vol_ratio >= 1.25 and ret_5 > 0:
            score += 6
            reasons.append(f"近期量能放大 {vol_ratio:.1f}x")
        if drawdown < -18:
            risks.append(f"距阶段高点 {drawdown:.1f}%")

    return {
        "score": max(0.0, min(100.0, score)),
        "reasons": reasons,
        "risks": risks,
    }


def _quality_score(symbol: str) -> tuple[float, list[str], list[str]]:
    score = 52.0
    reasons: list[str] = []
    risks: list[str] = []
    if symbol in _ANCHORS:
        score = 60.0
        reasons.append("市场锚点")
        risks.append("指数/ETF 用于环境判断，不直接代表个股机会")
    elif symbol in _MEGA_CAP:
        score += 18
        reasons.append("大型权重股，流动性较好")
    elif symbol in _CURATED_LARGE_CAP:
        score += 12
        reasons.append("高质量大盘候选源")
    elif symbol in _BROAD_ETFS:
        score += 6
        reasons.append("高流动性 ETF")
        risks.append("ETF 更适合做环境或对冲参考")
    else:
        reasons.append("来自候选源或本地缓存")

    if len(symbol) > 5 and "." not in symbol and "-" not in symbol:
        score -= 10
        risks.append("代码形态异常，需人工确认")
    return max(0.0, min(100.0, score)), reasons, risks


def _combine_weighted(
    *,
    parts: dict[str, tuple[Optional[float], float]],
    required_missing: list[tuple[str, bool, str]],
) -> tuple[float, str, list[str], list[str]]:
    weighted = 0.0
    total = 0.0
    reasons: list[str] = []
    risks: list[str] = []
    available = 0
    for name, (value, weight) in parts.items():
        if value is None:
            continue
        weighted += float(value) * weight
        total += weight
        available += 1
        reasons.append(f"{name} {float(value):.1f}")
    for _name, missing, msg in required_missing:
        if missing:
            risks.append(msg)
    score = weighted / total if total else 50.0
    confidence = "高" if available >= 3 and not risks else ("中" if available >= 2 else "低")
    return score, confidence, reasons, risks


def _from_daily_candidate(row: DailyCandidate) -> PoolItem:
    return PoolItem(
        symbol=row.symbol,
        rank=row.rank,
        score=row.score,
        status=row.status,
        data_confidence=row.data_confidence,
        layer=DAILY_DECISION,
        component_scores={
            "quality": row.source_quality_score,
            "ai": row.ai_score,
            "tactical": row.tactical_score,
        },
        reasons=list(row.reasons),
        risk_flags=list(row.risk_flags),
        as_of=row.as_of,
    )


def _long_status(score: float, risks: list[str]) -> str:
    if _has_hard_risk(risks):
        return "BENCH" if score >= 50 else "AVOID"
    if score >= 76:
        return "CORE"
    if score >= 65:
        return "WATCH"
    if score >= 55:
        return "RESEARCH"
    if score >= 45:
        return "BENCH"
    return "AVOID"


def _weekly_status(score: float, risks: list[str]) -> str:
    if _has_hard_risk(risks):
        return "COOL_DOWN" if score < 55 else "WATCH"
    if score >= 74:
        return "FOCUS_READY"
    if score >= 66:
        return "SETUP"
    if score >= 56:
        return "WATCH"
    return "COOL_DOWN"


def _status_order(layer: str, status: str) -> int:
    orders = {
        LONG_TERM: {"CORE": 0, "WATCH": 1, "RESEARCH": 2, "BENCH": 3, "AVOID": 4},
        WEEKLY_FOCUS: {"FOCUS_READY": 0, "SETUP": 1, "WATCH": 2, "COOL_DOWN": 3},
        DAILY_DECISION: {
            "ENTRY_READY": 0, "WAIT_TRIGGER": 1, "WAIT_BREAKOUT": 2, "WATCH": 3,
            "BENCH": 3, "MARKET_ANCHOR": 4, "AVOID_NOW": 5,
        },
    }
    return orders.get(layer, {}).get(status, 9)


def _pool_result_from_dict(data: dict) -> PoolResult:
    items = [
        PoolItem(**item)
        for item in data.get("items", [])
        if isinstance(item, dict)
    ]
    return PoolResult(
        layer=data.get("layer", ""),
        updated_at=data.get("updated_at", ""),
        source_size=int(data.get("source_size", 0) or 0),
        selected_size=int(data.get("selected_size", len(items)) or 0),
        items=items,
        warnings=list(data.get("warnings", [])),
    )


def _normalize_symbols(source: Iterable[str] | str | None) -> list[str]:
    if source is None:
        return []
    raw_items: list[str] = []
    if isinstance(source, str):
        text = source.replace("\n", ",").replace(";", ",").replace("，", ",")
        raw_items = [part for part in text.split(",")]
    else:
        raw_items = [str(item) for item in source]

    out: list[str] = []
    for raw in raw_items:
        symbol = str(raw or "").strip().upper()
        if not symbol:
            continue
        symbol = symbol.replace("/", ".")
        if symbol and symbol not in out:
            out.append(symbol)
    return out


def _cached_symbols() -> list[str]:
    if not _BARS_DIR.exists():
        return []
    symbols: list[str] = []
    for path in _BARS_DIR.glob("*.parquet"):
        stem = path.stem
        if "_" not in stem:
            continue
        symbol = stem.rsplit("_", 1)[0].upper()
        if symbol:
            symbols.append(symbol)
    return _normalize_symbols(symbols)


def _load_ai_scores(ai_db_path: Optional[str]) -> dict[str, float]:
    if not ai_db_path:
        return {}
    try:
        from .ai.manager import get_composite_scores_from_db
        return get_composite_scores_from_db(ai_db_path)
    except Exception:
        return {}


def _load_bars(symbol: str, timeframe: str) -> pd.DataFrame:
    try:
        from .data_cache import get_bars
        return get_bars(symbol, timeframe)
    except Exception:
        return pd.DataFrame()


def _download_bars(symbol: str, timeframe: str) -> pd.DataFrame:
    try:
        from .data_cache import fetch_and_save
        return fetch_and_save(symbol, timeframe)
    except Exception:
        return pd.DataFrame()


def _clean_bars(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    clean = df.copy()
    clean.columns = [str(col).lower() for col in clean.columns]
    required = {"close", "volume"}
    if not required.issubset(clean.columns):
        return pd.DataFrame()
    for col in ("close", "volume"):
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
    return clean.dropna(subset=["close"]).reset_index(drop=True)


def _volume_ratio(volume: pd.Series) -> Optional[float]:
    if volume.empty or len(volume) < 25:
        return None
    avg = float(volume.tail(60).mean())
    if avg <= 0:
        return None
    return float(volume.tail(10).mean() / avg)


def _pct(value: float, base: float) -> float:
    return (value - base) / base * 100 if base else 0.0


def _rank(items: list[PoolItem]) -> None:
    for idx, item in enumerate(items, start=1):
        item.rank = idx


def _symbols(items: list[PoolItem]) -> list[str]:
    return [item.symbol for item in items if item.symbol not in _ANCHORS]


def _has_hard_risk(risks: list[str]) -> bool:
    hard_tokens = ("缺少日线趋势数据", "缺少周级技术数据", "缺少决策池所需日线数据", "代码形态异常")
    return any(any(token in risk for token in hard_tokens) for risk in risks)


def _round_or_none(value: Optional[float]) -> Optional[float]:
    return round(float(value), 1) if value is not None else None


def _unique(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return out


def _now_s() -> str:
    return datetime.now(timezone.utc).isoformat()


def _progress(callback: Optional[Callable[[dict], None]], stage: str, current: int, total: int, message: str) -> None:
    if not callback:
        return
    try:
        callback({
            "stage": stage,
            "current": int(current),
            "total": int(total),
            "message": message,
            "updated_at": _now_s(),
        })
    except Exception:
        pass
