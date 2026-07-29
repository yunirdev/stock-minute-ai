from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from .data_cache import get_bars
from .selection_pools import (
    DAILY_DECISION,
    DECISION_STYLE_AGGRESSIVE,
    DECISION_STYLE_STANDARD,
    PoolItem,
    load_decision_pool_report,
    load_selection_pool,
)


_ROOT = Path(__file__).resolve().parents[1]
_STORE = _ROOT / "conf" / "decision_trade_plans.json"


@dataclass
class DecisionTradePlan:
    symbol: str
    rank: int
    score: float
    status: str
    decision_type: str
    direction: str
    action: str
    latest_price: Optional[float] = None
    atr14: Optional[float] = None
    atr_pct: Optional[float] = None
    entry_trigger: str = ""
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    invalidation: str = ""
    suggested_weight_pct: float = 0.0
    risk_per_trade_pct: float = 0.0
    max_position_pct: float = 0.0
    time_horizon: str = "intraday_to_swing"
    source_status: str = "UNKNOWN"
    blocked_reason: str = ""
    reasons: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    as_of: str = ""


@dataclass
class DecisionTradePlanReport:
    updated_at: str
    decision_style: str
    source_symbols: list[str]
    plans: list[DecisionTradePlan]
    ready_count: int = 0
    wait_count: int = 0
    blocked_count: int = 0
    warnings: list[str] = field(default_factory=list)


def build_decision_trade_plan_report(
    items: Optional[Iterable[PoolItem]] = None,
    *,
    decision_style: str | None = None,
    save: bool = True,
    path: Path | str = _STORE,
) -> DecisionTradePlanReport:
    pool_items = list(items) if items is not None else list(load_selection_pool(DAILY_DECISION).items)
    report = load_decision_pool_report()
    style = _normalize_style(decision_style or report.decision_style)
    now_s = _now_s()
    plans = [_build_plan(item, style=style, now_s=now_s) for item in pool_items]
    result = DecisionTradePlanReport(
        updated_at=now_s,
        decision_style=style,
        source_symbols=[item.symbol for item in pool_items],
        plans=plans,
        ready_count=sum(1 for plan in plans if plan.action == "TRADE_READY"),
        wait_count=sum(1 for plan in plans if plan.action in {"WAIT_TRIGGER", "HEDGE_ONLY", "REVIEW_ONLY"}),
        blocked_count=sum(1 for plan in plans if plan.action == "BLOCKED"),
        warnings=_report_warnings(plans),
    )
    if save:
        save_decision_trade_plan_report(result, path)
    return result


def save_decision_trade_plan_report(
    report: DecisionTradePlanReport,
    path: Path | str = _STORE,
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8")


def load_decision_trade_plan_report(path: Path | str = _STORE) -> DecisionTradePlanReport:
    src = Path(path)
    if not src.exists():
        return DecisionTradePlanReport(updated_at="", decision_style=DECISION_STYLE_STANDARD, source_symbols=[], plans=[])
    try:
        payload = json.loads(src.read_text(encoding="utf-8"))
        plans = [
            DecisionTradePlan(**item)
            for item in payload.get("plans", [])
            if isinstance(item, dict)
        ]
        return DecisionTradePlanReport(
            updated_at=str(payload.get("updated_at", "") or ""),
            decision_style=_normalize_style(str(payload.get("decision_style", DECISION_STYLE_STANDARD) or "")),
            source_symbols=list(payload.get("source_symbols", []) or []),
            plans=plans,
            ready_count=int(payload.get("ready_count", 0) or 0),
            wait_count=int(payload.get("wait_count", 0) or 0),
            blocked_count=int(payload.get("blocked_count", 0) or 0),
            warnings=list(payload.get("warnings", []) or []),
        )
    except Exception:
        return DecisionTradePlanReport(updated_at="", decision_style=DECISION_STYLE_STANDARD, source_symbols=[], plans=[])


def executable_symbols(limit: int = 7, path: Path | str = _STORE) -> list[str]:
    report = load_decision_trade_plan_report(path)
    rows = [plan for plan in report.plans if plan.action != "BLOCKED"]
    rows.sort(key=lambda plan: plan.rank or 999)
    return [plan.symbol for plan in rows[: max(1, int(limit or 7))]]


def _build_plan(item: PoolItem, *, style: str, now_s: str) -> DecisionTradePlan:
    reasons = list(item.reasons or [])
    risk_flags = list(item.risk_flags or [])
    decision_type = _reason_value(reasons, "类型") or "WATCH"
    direction = _reason_value(reasons, "方向") or "WATCH"
    status = str(item.status or "")
    bars = _clean_bars(get_bars(item.symbol, "1d"))
    base_plan = DecisionTradePlan(
        symbol=item.symbol,
        rank=int(item.rank or 0),
        score=float(item.score or 0.0),
        status=status,
        decision_type=decision_type,
        direction=direction,
        action="BLOCKED",
        source_status="NO_BARS",
        blocked_reason="缺少可验证日线数据，不能生成交易计划",
        reasons=reasons[:8],
        risk_flags=risk_flags[:8],
        as_of=now_s,
    )
    if len(bars) < 20:
        return base_plan

    latest = float(bars["close"].iloc[-1])
    atr = _atr14(bars)
    if latest <= 0 or atr <= 0:
        base_plan.source_status = "BAD_PRICE"
        base_plan.blocked_reason = "价格或波动率异常，不能生成交易计划"
        return base_plan

    ma20 = float(bars["close"].tail(20).mean())
    ma50 = float(bars["close"].tail(min(50, len(bars))).mean())
    atr_pct = atr / latest * 100
    side = "SHORT" if direction == "SHORT" else ("HEDGE" if direction == "HEDGE" else "LONG")
    stop_distance = _stop_multiplier(decision_type, style) * atr
    rr = 1.8 if decision_type in {"AGGRESSIVE_MOMENTUM", "REVERSAL"} else 2.0

    if side == "SHORT":
        stop = latest + stop_distance
        target = latest - stop_distance * rr
        entry_text = f"反弹不过 ${ma20:.2f}，或跌破 ${latest:.2f} 后回抽失败"
        invalidation = f"重新站回 50 日线 ${ma50:.2f} 或放量收复止损位"
    elif side == "HEDGE":
        stop = latest - stop_distance
        target = latest + stop_distance * 1.4
        entry_text = f"仅作指数/行业对冲参考，等待盘中确认 ${latest:.2f}"
        invalidation = "个股信号更清晰或市场风险回落时取消对冲优先级"
    else:
        stop = latest - stop_distance
        target = latest + stop_distance * rr
        entry_text = f"站稳 ${latest:.2f}，或回踩 20 日线 ${ma20:.2f} 附近企稳"
        invalidation = f"收盘跌破 20 日线 ${ma20:.2f} 或触及止损后重新评估"

    action = _action_for(status, side)
    blocked_reason = ""
    risks = list(risk_flags[:8])
    if atr_pct >= 8:
        risks.append(f"ATR14/价格 {atr_pct:.1f}%：波动过高，自动交易需显著降仓")
    elif atr_pct >= 5:
        risks.append(f"ATR14/价格 {atr_pct:.1f}%：波动偏高，避免追单")
    if _is_stale(bars):
        risks.append("本地日线可能不是最新交易日，盘前需刷新行情")
    if action == "BLOCKED":
        blocked_reason = "当前状态不进入可交易/等待触发队列"

    suggested_weight = _suggested_weight(decision_type, status, atr_pct, style)
    risk_pct = _risk_per_trade(decision_type, atr_pct, style)
    return DecisionTradePlan(
        symbol=item.symbol,
        rank=int(item.rank or 0),
        score=round(float(item.score or 0.0), 1),
        status=status,
        decision_type=decision_type,
        direction=direction,
        action=action,
        latest_price=round(latest, 4),
        atr14=round(atr, 4),
        atr_pct=round(atr_pct, 2),
        entry_trigger=entry_text,
        stop_loss=round(max(0.01, stop), 4),
        take_profit=round(max(0.01, target), 4),
        invalidation=invalidation,
        suggested_weight_pct=round(suggested_weight, 1),
        risk_per_trade_pct=round(risk_pct, 2),
        max_position_pct=round(min(30.0 if style == DECISION_STYLE_AGGRESSIVE else 25.0, suggested_weight * 1.35), 1),
        time_horizon="intraday_to_3d" if decision_type == "AGGRESSIVE_MOMENTUM" else "intraday_to_swing",
        source_status="OK",
        blocked_reason=blocked_reason,
        reasons=reasons[:8],
        risk_flags=_unique(risks)[:8],
        as_of=now_s,
    )


def _clean_bars(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out.columns = [str(col).lower() for col in out.columns]
    if "close" not in out.columns:
        return pd.DataFrame()
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    if "high" not in out.columns:
        out["high"] = out["close"]
    if "low" not in out.columns:
        out["low"] = out["close"]
    out["high"] = pd.to_numeric(out["high"], errors="coerce").fillna(out["close"])
    out["low"] = pd.to_numeric(out["low"], errors="coerce").fillna(out["close"])
    out = out.dropna(subset=["close"])
    return out[out["close"] > 0].reset_index(drop=True)


def _atr14(df: pd.DataFrame) -> float:
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    prev = close.shift(1)
    tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    atr = float(tr.tail(14).mean())
    if not pd.notna(atr) or atr <= 0:
        atr = float(close.pct_change().tail(20).std() * close.iloc[-1])
    return max(atr, float(close.iloc[-1]) * 0.005)


def _action_for(status: str, side: str) -> str:
    if status == "ENTRY_READY":
        return "HEDGE_ONLY" if side == "HEDGE" else "TRADE_READY"
    if status in {"WAIT_TRIGGER", "WAIT_BREAKOUT", "WATCH"}:
        return "HEDGE_ONLY" if side == "HEDGE" else "WAIT_TRIGGER"
    if status == "BENCH":
        return "REVIEW_ONLY"
    return "BLOCKED"


def _suggested_weight(decision_type: str, status: str, atr_pct: float, style: str) -> float:
    base = {
        "ETF_MACRO": 20.0,
        "LONG_TREND": 18.0,
        "AGGRESSIVE_MOMENTUM": 12.0,
        "REVERSAL": 10.0,
        "SHORT_TREND": 8.0,
    }.get(decision_type, 8.0)
    if style == DECISION_STYLE_AGGRESSIVE:
        base *= 1.18
    if status in {"WAIT_TRIGGER", "WATCH"}:
        base *= 0.75
    elif status == "BENCH":
        base *= 0.5
    if atr_pct >= 8:
        base *= 0.45
    elif atr_pct >= 5:
        base *= 0.65
    return max(2.0, min(base, 24.0))


def _risk_per_trade(decision_type: str, atr_pct: float, style: str) -> float:
    risk = 0.75 if style == DECISION_STYLE_STANDARD else 0.95
    if decision_type in {"AGGRESSIVE_MOMENTUM", "REVERSAL", "SHORT_TREND"}:
        risk -= 0.15
    if atr_pct >= 8:
        risk *= 0.55
    elif atr_pct >= 5:
        risk *= 0.75
    return max(0.25, min(risk, 1.2))


def _stop_multiplier(decision_type: str, style: str) -> float:
    if decision_type == "AGGRESSIVE_MOMENTUM":
        return 1.25 if style == DECISION_STYLE_AGGRESSIVE else 1.15
    if decision_type == "REVERSAL":
        return 1.2
    if decision_type == "SHORT_TREND":
        return 1.35
    return 1.5


def _is_stale(df: pd.DataFrame) -> bool:
    for col in ("timestamp_utc", "timestamp", "date"):
        if col not in df.columns:
            continue
        try:
            ts = pd.to_datetime(df[col].iloc[-1], utc=True)
            return (datetime.now(timezone.utc) - ts.to_pydatetime()).days > 10
        except Exception:
            return False
    return False


def _reason_value(reasons: list[str], prefix: str) -> str:
    head = f"{prefix} "
    for reason in reasons:
        if str(reason).startswith(head):
            return str(reason).replace(head, "", 1)
    return ""


def _normalize_style(style: str | None) -> str:
    raw = str(style or "").strip().lower()
    if raw in {DECISION_STYLE_AGGRESSIVE, "小资金进攻", "进攻", "aggressive"}:
        return DECISION_STYLE_AGGRESSIVE
    return DECISION_STYLE_STANDARD


def _report_warnings(plans: list[DecisionTradePlan]) -> list[str]:
    warnings = []
    if not plans:
        warnings.append("决策池为空，未生成交易计划")
    if any(plan.source_status != "OK" for plan in plans):
        warnings.append("部分标的缺少可验证 K 线，已阻断自动交易计划")
    if any((plan.atr_pct or 0) >= 8 for plan in plans):
        warnings.append("存在超高波动标的，仓位建议已下调")
    return warnings


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _now_s() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
