"""Canonical research-signal lifecycle and durable event store."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable

import duckdb

from .ai.safety import AIScoreSnapshot
from .models import Bar, Fill, Side, TradePlan


class SignalState(str, Enum):
    READY = "READY"
    ENTERED = "ENTERED"
    HOLD = "HOLD"
    EXIT = "EXIT"
    INVALIDATED = "INVALIDATED"
    CLOSED = "CLOSED"


_TERMINAL = {SignalState.INVALIDATED, SignalState.CLOSED}
_TRANSITIONS = {
    SignalState.READY: {SignalState.ENTERED, SignalState.EXIT, SignalState.INVALIDATED},
    SignalState.ENTERED: {SignalState.HOLD, SignalState.EXIT, SignalState.CLOSED},
    SignalState.HOLD: {SignalState.HOLD, SignalState.EXIT, SignalState.CLOSED},
    SignalState.EXIT: {SignalState.CLOSED},
    SignalState.INVALIDATED: set(),
    SignalState.CLOSED: set(),
}


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _iso(value: datetime | None) -> str | None:
    return _utc(value).isoformat() if value is not None else None


def _dt(value: str | datetime | None) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return _utc(value) if value is not None else None
    parsed = datetime.fromisoformat(value)
    return _utc(parsed)


@dataclass
class SignalReport:
    signal_id: str
    version: int
    symbol: str
    state: SignalState
    side: Side
    strategy: str
    timeframe: str
    market_regime: str
    market_price: float
    market_data_at: datetime
    generated_at: datetime
    valid_until: datetime
    entry_low: float
    entry_high: float
    chase_limit: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    model_weight_pct: float
    model_risk_pct: float
    ai_score: float | None
    ai_run_id: str | None
    ai_contributors: list[str]
    decision_id: str
    plan_id: str
    reasons: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    invalidation: str = ""
    quantity: float = 0.0
    entry_fill_price: float | None = None
    exit_fill_price: float | None = None
    realized_pnl: float | None = None
    realized_return_pct: float | None = None
    closed_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["state"] = self.state.value
        value["side"] = self.side.value
        for key in ("market_data_at", "generated_at", "valid_until", "closed_at"):
            value[key] = _iso(value[key])
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SignalReport":
        payload = dict(value)
        payload["state"] = SignalState(payload["state"])
        payload["side"] = Side(payload["side"])
        for key in ("market_data_at", "generated_at", "valid_until", "closed_at"):
            payload[key] = _dt(payload.get(key))
        return cls(**payload)


def build_ready_signal_report(
    plan: TradePlan,
    decision: Any,
    snapshot: AIScoreSnapshot | None,
    latest_bar: Bar,
    *,
    equity: float,
    now: datetime,
    timeframe: str,
) -> SignalReport:
    now = _utc(now)
    entry = float(plan.entry_price)
    stop_distance = abs(entry - float(plan.stop_loss))
    reward_distance = abs(float(plan.take_profit) - entry)
    risk_reward = reward_distance / stop_distance if stop_distance > 0 else 0.0
    quantity = max(0.0, float(plan.qty or 0.0))
    model_risk = stop_distance * quantity / equity * 100 if equity > 0 else 0.0
    contributors = [
        str(item.get("agent_name"))
        for item in (snapshot.contributors if snapshot else []) or []
        if item.get("agent_name") and not item.get("is_fallback")
    ]
    decision_id = str(getattr(decision, "decision_id", "") or plan.metadata.get("decision_id", ""))
    strategy = str(getattr(decision, "strategy", "") or plan.metadata.get("strategy", ""))
    regime = str(getattr(decision, "market_regime", "") or "unknown")
    valid_until = _dt(getattr(decision, "valid_until", None)) or now
    reasons = list(getattr(decision, "reason_codes", ()) or ())
    reasons.append(f"STRATEGY:{strategy}")
    risks = list(plan.metadata.get("risk_flags", []) or [])
    signal_key = f"{decision_id}|{plan.plan_id}|{plan.symbol}|{now.isoformat()}"
    signal_id = "sig-" + hashlib.sha256(signal_key.encode()).hexdigest()[:24]
    band = entry * 0.005
    chase_limit = entry * (1.01 if plan.side == Side.BUY else 0.99)
    return SignalReport(
        signal_id=signal_id,
        version=1,
        symbol=plan.symbol,
        state=SignalState.READY,
        side=plan.side,
        strategy=strategy,
        timeframe=timeframe,
        market_regime=regime,
        market_price=float(latest_bar.close),
        market_data_at=_utc(latest_bar.timestamp),
        generated_at=now,
        valid_until=valid_until,
        entry_low=round(entry - band, 4),
        entry_high=round(entry + band, 4),
        chase_limit=round(chase_limit, 4),
        stop_loss=round(float(plan.stop_loss), 4),
        take_profit=round(float(plan.take_profit), 4),
        risk_reward=round(risk_reward, 2),
        model_weight_pct=round(float(plan.target_weight or 0.0) * 100, 2),
        model_risk_pct=round(model_risk, 3),
        ai_score=(round(float(snapshot.score), 1) if snapshot and snapshot.score is not None else None),
        ai_run_id=(snapshot.run_id if snapshot else None),
        ai_contributors=contributors,
        decision_id=decision_id,
        plan_id=plan.plan_id,
        reasons=reasons[:8],
        risks=risks[:8],
        invalidation=f"Signal expires at {_iso(valid_until)} or the model stop is reached.",
        quantity=quantity,
    )


class SignalStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._init_db()

    def _connect(self):
        for attempt in range(5):
            try:
                return duckdb.connect(self.db_path)
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(0.1 * (attempt + 1))

    def _init_db(self) -> None:
        con = self._connect()
        try:
            con.execute("""
                CREATE TABLE IF NOT EXISTS signal_events (
                    event_id VARCHAR PRIMARY KEY,
                    signal_id VARCHAR,
                    version INTEGER,
                    symbol VARCHAR,
                    state VARCHAR,
                    side VARCHAR,
                    strategy VARCHAR,
                    timeframe VARCHAR,
                    decision_id VARCHAR,
                    plan_id VARCHAR,
                    payload_json VARCHAR,
                    created_at TIMESTAMPTZ,
                    UNIQUE(signal_id, version)
                )
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS signal_plan_links (
                    plan_id VARCHAR PRIMARY KEY,
                    signal_id VARCHAR,
                    created_at TIMESTAMPTZ
                )
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS signal_publications (
                    publication_id VARCHAR PRIMARY KEY,
                    signal_id VARCHAR,
                    version INTEGER,
                    status VARCHAR,
                    attempts INTEGER,
                    last_error VARCHAR,
                    created_at TIMESTAMPTZ,
                    sent_at TIMESTAMPTZ,
                    UNIQUE(signal_id, version)
                )
            """)
            con.commit()
        finally:
            con.close()

    def register_ready(self, report: SignalReport) -> tuple[SignalReport, bool]:
        active = self.find_active(report.symbol, report.side)
        if active is not None:
            self.link_plan(report.plan_id, active.signal_id)
            return active, False
        self._insert(report)
        self.link_plan(report.plan_id, report.signal_id)
        return report, True

    def transition(
        self,
        signal_id: str,
        state: SignalState,
        *,
        at: datetime | None = None,
        **updates: Any,
    ) -> SignalReport | None:
        current = self.latest(signal_id)
        if current is None:
            return None
        if current.state == state and state != SignalState.HOLD:
            return current
        if state not in _TRANSITIONS[current.state]:
            raise ValueError(f"INVALID_SIGNAL_TRANSITION:{current.state.value}->{state.value}")
        now = _utc(at or datetime.now(timezone.utc))
        payload = {**updates, "state": state, "version": current.version + 1, "generated_at": now}
        if state in _TERMINAL:
            payload["closed_at"] = now
        updated = replace(current, **payload)
        self._insert(updated)
        return updated

    def apply_fill(self, plan_id: str, fill: Fill, *, final: bool = True) -> SignalReport | None:
        report = self.find_by_plan(plan_id)
        if report is None:
            return None
        if fill.side == Side.BUY:
            if report.state == SignalState.READY:
                return self.transition(
                    report.signal_id,
                    SignalState.ENTERED,
                    at=fill.fill_time,
                    quantity=float(fill.filled_qty),
                    entry_fill_price=float(fill.avg_price),
                )
            return report
        if not final:
            return report
        entry = report.entry_fill_price or report.entry_high
        quantity = min(float(fill.filled_qty), report.quantity or float(fill.filled_qty))
        pnl = (float(fill.avg_price) - float(entry)) * quantity
        return_pct = (float(fill.avg_price) / float(entry) - 1) * 100 if entry else 0.0
        target_state = SignalState.CLOSED if report.state in {SignalState.ENTERED, SignalState.HOLD, SignalState.EXIT} else SignalState.INVALIDATED
        return self.transition(
            report.signal_id,
            target_state,
            at=fill.fill_time,
            exit_fill_price=float(fill.avg_price),
            realized_pnl=round(pnl, 2),
            realized_return_pct=round(return_pct, 3),
        )

    def mark_exit(self, symbol: str, *, plan_id: str, at: datetime | None = None) -> SignalReport | None:
        current = self.find_active(symbol)
        if current is None or current.state not in {SignalState.ENTERED, SignalState.HOLD}:
            return current
        updated = self.transition(current.signal_id, SignalState.EXIT, at=at)
        if updated is not None:
            self.link_plan(plan_id, updated.signal_id)
        return updated

    def invalidate_expired(self, now: datetime) -> list[SignalReport]:
        expired = []
        for report in self.active():
            if report.state == SignalState.READY and report.valid_until < _utc(now):
                updated = self.transition(report.signal_id, SignalState.INVALIDATED, at=now)
                if updated is not None:
                    expired.append(updated)
        return expired

    def latest(self, signal_id: str) -> SignalReport | None:
        return self._one(
            "SELECT payload_json FROM signal_events WHERE signal_id=? ORDER BY version DESC LIMIT 1",
            [signal_id],
        )

    def find_by_plan(self, plan_id: str) -> SignalReport | None:
        con = self._connect()
        try:
            row = con.execute(
                "SELECT signal_id FROM signal_plan_links WHERE plan_id=?",
                [plan_id],
            ).fetchone()
        finally:
            con.close()
        return self.latest(row[0]) if row else None

    def find_active(self, symbol: str, side: Side | None = None) -> SignalReport | None:
        candidates = [item for item in self.active() if item.symbol == symbol]
        if side is not None:
            candidates = [item for item in candidates if item.side == side]
        return candidates[0] if candidates else None

    def active(self) -> list[SignalReport]:
        return [
            item
            for item in self.recent(limit=500)
            if item.state not in _TERMINAL
        ]

    def recent(self, limit: int = 100) -> list[SignalReport]:
        con = self._connect()
        try:
            rows = con.execute(
                """
                SELECT payload_json FROM signal_events e
                WHERE version=(SELECT MAX(version) FROM signal_events x WHERE x.signal_id=e.signal_id)
                ORDER BY created_at DESC LIMIT ?
                """,
                [limit],
            ).fetchall()
        finally:
            con.close()
        return [SignalReport.from_dict(json.loads(row[0])) for row in rows]

    def pending_publications(
        self,
        *,
        since: datetime,
        limit: int = 50,
    ) -> list["SignalReport"]:
        """还没成功播报出去的信号事件（按发生顺序）。

        为什么不能用 recent()：那个查询每个 signal 只取最新版本。一个信号在
        两次 tick 之间走完 READY → ENTERED，recent() 只会给出 ENTERED，
        READY 那条"可以进场了"就永远没人看见了——而它恰恰是最有行动价值的
        一条。这里按 (signal_id, version) 逐条比对播报记录，中间态不会丢。

        since 是防补发闸门：推送中断几天后重新接上，不该把积压的历史事件一
        次性倾泻到频道里，那些行情早就过时了。
        """
        con = self._connect()
        try:
            rows = con.execute(
                """
                SELECT e.payload_json
                FROM signal_events e
                LEFT JOIN signal_publications p
                  ON p.signal_id = e.signal_id AND p.version = e.version
                WHERE e.created_at >= ?
                  AND (p.status IS NULL OR p.status <> 'SENT')
                ORDER BY e.created_at
                LIMIT ?
                """,
                [_utc(since), limit],
            ).fetchall()
        finally:
            con.close()
        return [SignalReport.from_dict(json.loads(row[0])) for row in rows]

    def begin_publication(self, report: SignalReport) -> bool:
        publication_id = f"pub-{report.signal_id}-{report.version}"
        con = self._connect()
        try:
            existing = con.execute(
                "SELECT status FROM signal_publications WHERE signal_id=? AND version=?",
                [report.signal_id, report.version],
            ).fetchone()
            if existing and existing[0] == "SENT":
                return False
            if existing:
                con.execute(
                    "UPDATE signal_publications SET status='PENDING', attempts=attempts+1, last_error=NULL WHERE signal_id=? AND version=?",
                    [report.signal_id, report.version],
                )
            else:
                con.execute(
                    "INSERT INTO signal_publications VALUES (?,?,?,?,?,?,?,?)",
                    [publication_id, report.signal_id, report.version, "PENDING", 1, None, datetime.now(timezone.utc), None],
                )
            con.commit()
            return True
        finally:
            con.close()

    def finish_publication(self, report: SignalReport, ok: bool, error: str = "") -> None:
        con = self._connect()
        try:
            con.execute(
                "UPDATE signal_publications SET status=?, last_error=?, sent_at=? WHERE signal_id=? AND version=?",
                ["SENT" if ok else "FAILED", error or None, datetime.now(timezone.utc) if ok else None, report.signal_id, report.version],
            )
            con.commit()
        finally:
            con.close()

    def performance_summary(self) -> dict[str, Any]:
        reports = self.recent(limit=5000)
        closed = [item for item in reports if item.state == SignalState.CLOSED and item.realized_return_pct is not None]
        invalidated = [item for item in reports if item.state == SignalState.INVALIDATED]
        wins = [item for item in closed if (item.realized_pnl or 0.0) > 0]
        losses = [item for item in closed if (item.realized_pnl or 0.0) < 0]
        by_strategy: dict[str, dict[str, float]] = {}
        by_contributor: dict[str, dict[str, float]] = {}
        for report in closed:
            self._accumulate(by_strategy, report.strategy or "unknown", report)
            for contributor in report.ai_contributors:
                self._accumulate(by_contributor, contributor, report)
        return {
            "active": sum(1 for item in reports if item.state not in _TERMINAL),
            "closed": len(closed),
            "invalidated": len(invalidated),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate_pct": round(len(wins) / len(closed) * 100, 1) if closed else 0.0,
            "total_pnl": round(sum(item.realized_pnl or 0.0 for item in closed), 2),
            "average_return_pct": round(sum(item.realized_return_pct or 0.0 for item in closed) / len(closed), 3) if closed else 0.0,
            "by_strategy": self._finalize(by_strategy),
            "by_contributor": self._finalize(by_contributor),
        }

    @staticmethod
    def _accumulate(target: dict[str, dict[str, float]], key: str, report: SignalReport) -> None:
        row = target.setdefault(key, {"count": 0.0, "wins": 0.0, "pnl": 0.0, "return": 0.0})
        row["count"] += 1
        row["wins"] += 1 if (report.realized_pnl or 0.0) > 0 else 0
        row["pnl"] += report.realized_pnl or 0.0
        row["return"] += report.realized_return_pct or 0.0

    @staticmethod
    def _finalize(target: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        result = {}
        for key, row in target.items():
            count = row["count"]
            result[key] = {
                "count": int(count),
                "win_rate_pct": round(row["wins"] / count * 100, 1) if count else 0.0,
                "total_pnl": round(row["pnl"], 2),
                "average_return_pct": round(row["return"] / count, 3) if count else 0.0,
            }
        return result

    def link_plan(self, plan_id: str, signal_id: str) -> None:
        if not plan_id:
            return
        con = self._connect()
        try:
            con.execute(
                "INSERT INTO signal_plan_links VALUES (?,?,?) ON CONFLICT(plan_id) DO UPDATE SET signal_id=excluded.signal_id",
                [plan_id, signal_id, datetime.now(timezone.utc)],
            )
            con.commit()
        finally:
            con.close()

    def _insert(self, report: SignalReport) -> None:
        event_id = f"evt-{report.signal_id}-{report.version}"
        con = self._connect()
        try:
            con.execute(
                "INSERT INTO signal_events VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                [
                    event_id,
                    report.signal_id,
                    report.version,
                    report.symbol,
                    report.state.value,
                    report.side.value,
                    report.strategy,
                    report.timeframe,
                    report.decision_id,
                    report.plan_id,
                    json.dumps(report.to_dict(), ensure_ascii=False),
                    report.generated_at,
                ],
            )
            con.commit()
        finally:
            con.close()

    def _one(self, sql: str, params: Iterable[Any]) -> SignalReport | None:
        con = self._connect()
        try:
            row = con.execute(sql, list(params)).fetchone()
        finally:
            con.close()
        return SignalReport.from_dict(json.loads(row[0])) if row else None


#: 值得打断读者的状态——每一条都对应一个"现在要不要动手"的判断。
#:
#: 刻意排除两个：
#: - HOLD 是持仓心跳，每轮都可能重复，7 个标的会把频道变成噪音流。
#: - INVALIDATED 是"计划没等到就过期了"，属于事后信息而非行动信号，
#:   并入收盘报告的"未成交计划结局"一节更合适。
PUBLISHABLE_STATES = frozenset(
    {
        SignalState.READY,      # 可以进场了
        SignalState.ENTERED,    # 已成交
        SignalState.EXIT,       # 该退了
        SignalState.CLOSED,     # 结算，带已实现盈亏
    }
)


class SignalPublisher:
    def __init__(self, store: SignalStore, notifier: Any) -> None:
        self.store = store
        self.notifier = notifier

    def should_publish(self, report: SignalReport) -> bool:
        return report.state in PUBLISHABLE_STATES

    def publish(self, report: SignalReport) -> bool:
        if not self.should_publish(report):
            return True
        if not self.store.begin_publication(report):
            return True
        from .daily_discord import build_signal_report_message
        try:
            ok = bool(self.notifier.send(build_signal_report_message(report)))
            self.store.finish_publication(report, ok, "" if ok else "DELIVERY_FAILED")
            return ok
        except Exception as exc:
            self.store.finish_publication(report, False, type(exc).__name__)
            return False

    def publish_pending(self, *, since: datetime, limit: int = 50) -> int:
        """播报所有待发的信号事件，返回实际发出的条数。

        集中在一处轮询，而不是在 register_ready / apply_fill / mark_exit 等
        五个跃迁点各插一次调用——那样每加一个新的状态变更点就得记得补一次推
        送，迟早会漏。
        """
        sent = 0
        for report in self.store.pending_publications(since=since, limit=limit):
            if not self.should_publish(report):
                continue
            if self.publish(report):
                sent += 1
        return sent
