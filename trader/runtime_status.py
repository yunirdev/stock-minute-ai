"""Lock-free runtime status sidecar used by the live NiceGUI monitor."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .ai.safety import AIScoreSnapshot
from .models import Bar, Position, TradePlan

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATUS_PATH = _ROOT / "logs" / "runtime_status.json"


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _enum(value: Any) -> str:
    return str(getattr(value, "value", value or ""))


def build_runtime_status(
    *,
    now: datetime,
    tick_count: int,
    session: str,
    equity: float,
    reconciliation_blocked: bool,
    kill_switch: bool,
    bars: Mapping[str, Bar] | None = None,
    positions: Mapping[str, Position] | None = None,
    plans: Mapping[str, TradePlan] | None = None,
    research_snapshots: Mapping[str, AIScoreSnapshot] | None = None,
    research_run: Any | None = None,
    open_orders: Iterable[Any] = (),
    message: str = "",
    auto_trade_paper: bool = False,
) -> dict[str, Any]:
    bars = bars or {}
    positions = positions or {}
    plans = plans or {}
    research_snapshots = research_snapshots or {}
    symbols = sorted(
        set(bars)
        | set(positions)
        | set(plans)
        | set(research_snapshots)
    )
    rows = []
    for symbol in symbols:
        bar = bars.get(symbol)
        position = positions.get(symbol)
        plan = plans.get(symbol)
        snapshot = research_snapshots.get(symbol)
        price = float(bar.close) if bar is not None else None
        entry = float(plan.entry_price) if plan is not None else None
        if price is not None and entry:
            distance = (price / entry - 1.0) * 100
        else:
            distance = None
        if position is not None and float(position.qty) != 0:
            state = "HOLDING"
        elif plan is not None:
            state = "READY"
        elif snapshot is not None:
            state = "WATCHING"
        else:
            state = "OBSERVED"
        rows.append(
            {
                "symbol": symbol,
                "state": state,
                "price": price,
                "bar_at": _iso(bar.timestamp) if bar is not None else None,
                "entry_price": entry,
                "distance_to_entry_pct": (
                    round(distance, 3) if distance is not None else None
                ),
                "stop_loss": (
                    float(plan.stop_loss) if plan is not None else None
                ),
                "take_profit": (
                    float(plan.take_profit) if plan is not None else None
                ),
                "position_qty": (
                    float(position.qty) if position is not None else 0.0
                ),
                "unrealized_pnl": (
                    float(position.unrealized_pnl) if position is not None else 0.0
                ),
                "research_score": (
                    float(snapshot.score)
                    if snapshot is not None and snapshot.score is not None
                    else None
                ),
                "research_run_id": snapshot.run_id if snapshot is not None else None,
            }
        )
    orders = []
    for order in open_orders:
        orders.append(
            {
                "symbol": str(getattr(order, "symbol", "")),
                "side": _enum(getattr(order, "side", "")),
                "qty": float(getattr(order, "qty", 0.0) or 0.0),
                "limit_price": getattr(order, "limit_price", None),
                "intent_id": str(getattr(order, "intent_id", "")),
            }
        )
    return {
        "schema_version": 1,
        "updated_at": _iso(now),
        "tick_count": int(tick_count),
        "session": str(session),
        "equity": float(equity),
        "reconciliation_blocked": bool(reconciliation_blocked),
        "kill_switch": bool(kill_switch),
        # 引擎进程和 Dashboard 是两个独立进程——Dashboard 自己的 _cfg 只反映
        # 它上次用来启动引擎的参数，不代表真正在跑的那个引擎当前的实际配置
        # （引擎也可能是手动起的）。这个字段由引擎自己写出来，是唯一准确的
        # 来源，Dashboard 顶栏的 AutoTrade 指示灯必须读这个，不能读自己的 _cfg。
        "auto_trade_paper": bool(auto_trade_paper),
        "message": str(message),
        "daily_research": (
            {
                "run_id": getattr(research_run, "run_id", ""),
                "trading_date": getattr(research_run, "trading_date", ""),
                "status": getattr(research_run, "status", ""),
                "completed_symbols": int(
                    getattr(research_run, "completed_symbols", 0) or 0
                ),
                "failed_symbols": int(
                    getattr(research_run, "failed_symbols", 0) or 0
                ),
                "completed_at": _iso(getattr(research_run, "completed_at", None)),
            }
            if research_run is not None
            else None
        ),
        "candidates": rows,
        "open_orders": orders,
    }


def write_runtime_status(
    payload: Mapping[str, Any],
    path: Path | str = DEFAULT_STATUS_PATH,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def read_runtime_status(
    path: Path | str = DEFAULT_STATUS_PATH,
) -> dict[str, Any] | None:
    source = Path(path)
    if not source.exists():
        return None
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except Exception:
        return None
