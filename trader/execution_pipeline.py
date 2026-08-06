"""Versioned CandidatePlan -> FinalTradePlan -> OrderIntent state machine."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import duckdb

from .models import (
    CandidatePlan,
    CandidatePlanStatus,
    FinalTradePlan,
    FinalTradePlanStatus,
    OrderIntent,
    RiskVerdict,
    Side,
    TradePlan,
)
from .order_lifecycle import client_order_id, idempotency_key

# Marketable-limit 缓冲——LMT 单是硬性红线（AlpacaBroker 拒绝任何非 LMT 提交），
# 但一口价挂在 plan 生成那一刻的价位上、零缓冲，会让相当一部分信号因为价格
# 已经走开而整天不成交。这里给限价单主动让一点价换成交概率，同时仍然保留
# "最坏价格上限"——不是市价单。
#
# 出场单（止损/平仓/减仓）用更大的缓冲：入场没成交只是错过一次机会，出场
# 没成交等于风控失效、敞口继续暴露，两者的代价不对称，缓冲也不该一样大。
_ENTRY_LIMIT_BUFFER_PCT = 0.0015   # 入场：0.15%
_EXIT_LIMIT_BUFFER_PCT = 0.005     # 出场（CLOSE/REDUCE）：0.5%


def marketable_limit_price(reference_price: float, side: Side, *, action: str) -> float:
    """给定计划价和方向，算出实际提交给 broker 的限价（含缓冲）。

    BUY 单向上让价（愿意多付一点换成交），SELL 单向下让价（愿意少收一点换
    成交）——这条规则对入场和出场都成立，出场只是把同一个 side 的规则套用到
    "平仓方向"上（平多是 SELL，平空是 BUY），缓冲幅度按 action 是否为
    CLOSE/REDUCE 区分紧急程度。
    """
    buffer_pct = (
        _EXIT_LIMIT_BUFFER_PCT
        if str(action or "").upper() in {"CLOSE", "REDUCE"}
        else _ENTRY_LIMIT_BUFFER_PCT
    )
    multiplier = (1 + buffer_pct) if side == Side.BUY else (1 - buffer_pct)
    return round(float(reference_price) * multiplier, 4)


class ExecutionPipelineStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._migrate()

    def _connect(self, *, read_only: bool = False):
        return duckdb.connect(self.db_path, read_only=read_only)

    def _migrate(self) -> None:
        connection = self._connect()
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS candidate_plans (
                    candidate_plan_id TEXT PRIMARY KEY,
                    symbol TEXT, side TEXT, action TEXT,
                    entry_price DOUBLE, stop_loss DOUBLE,
                    take_profit DOUBLE, proposed_qty DOUBLE,
                    decision_id TEXT, strategy_version TEXT,
                    data_version TEXT, evidence_refs_json TEXT,
                    created_at TIMESTAMPTZ, valid_until TIMESTAMPTZ,
                    status TEXT, updated_at TIMESTAMPTZ
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS final_trade_plans (
                    final_plan_id TEXT PRIMARY KEY,
                    version INTEGER, candidate_plan_id TEXT UNIQUE,
                    symbol TEXT, side TEXT, action TEXT,
                    entry_price DOUBLE, stop_loss DOUBLE,
                    take_profit DOUBLE, qty DOUBLE, decision_id TEXT,
                    risk_check_id TEXT, risk_config_version TEXT,
                    strategy_version TEXT, data_version TEXT,
                    evidence_refs_json TEXT, created_at TIMESTAMPTZ,
                    valid_until TIMESTAMPTZ, status TEXT,
                    updated_at TIMESTAMPTZ
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS execution_plan_intents (
                    final_plan_id TEXT PRIMARY KEY,
                    intent_id TEXT UNIQUE,
                    idempotency_key TEXT UNIQUE,
                    created_at TIMESTAMPTZ
                )
                """
            )
            connection.commit()
        finally:
            connection.close()

    def register_candidate(self, candidate: CandidatePlan) -> CandidatePlan:
        if candidate.status != CandidatePlanStatus.DRAFT:
            raise ValueError("CANDIDATE_PLAN_INITIAL_STATUS_INVALID")
        connection = self._connect()
        try:
            connection.execute(
                """
                INSERT INTO candidate_plans VALUES
                (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                [
                    candidate.candidate_plan_id,
                    candidate.symbol,
                    candidate.side.value,
                    candidate.action,
                    candidate.entry_price,
                    candidate.stop_loss,
                    candidate.take_profit,
                    candidate.proposed_qty,
                    candidate.decision_id,
                    candidate.strategy_version,
                    candidate.data_version,
                    json.dumps(list(candidate.evidence_refs)),
                    candidate.created_at,
                    candidate.valid_until,
                    candidate.status.value,
                    candidate.created_at,
                ],
            )
            connection.commit()
        finally:
            connection.close()
        return candidate

    def validate_candidate(
        self,
        candidate_plan_id: str,
        *,
        now: datetime,
    ) -> CandidatePlan:
        candidate = self.get_candidate(candidate_plan_id)
        if candidate is None:
            raise KeyError(candidate_plan_id)
        if candidate.status != CandidatePlanStatus.DRAFT:
            raise ValueError("CANDIDATE_PLAN_TRANSITION_INVALID")
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("EXECUTION_PIPELINE_TIME_TZ_REQUIRED")
        if now > candidate.valid_until:
            self._set_candidate_status(
                candidate_plan_id,
                CandidatePlanStatus.EXPIRED,
                now,
            )
            raise ValueError("CANDIDATE_PLAN_EXPIRED")
        self._set_candidate_status(
            candidate_plan_id,
            CandidatePlanStatus.VALIDATED,
            now,
        )
        return replace(candidate, status=CandidatePlanStatus.VALIDATED)

    def reject_candidate(
        self,
        candidate_plan_id: str,
        *,
        now: datetime,
    ) -> CandidatePlan:
        candidate = self.get_candidate(candidate_plan_id)
        if candidate is None:
            raise KeyError(candidate_plan_id)
        if candidate.status not in {
            CandidatePlanStatus.DRAFT,
            CandidatePlanStatus.VALIDATED,
        }:
            raise ValueError("CANDIDATE_PLAN_TRANSITION_INVALID")
        self._set_candidate_status(
            candidate_plan_id,
            CandidatePlanStatus.REJECTED,
            now,
        )
        return replace(candidate, status=CandidatePlanStatus.REJECTED)

    def finalize(
        self,
        candidate_plan_id: str,
        *,
        risk_verdict: RiskVerdict,
        risk_check_id: str,
        risk_config_version: str,
        now: datetime,
    ) -> FinalTradePlan:
        candidate = self.get_candidate(candidate_plan_id)
        if candidate is None:
            raise KeyError(candidate_plan_id)
        if candidate.status != CandidatePlanStatus.VALIDATED:
            raise ValueError("CANDIDATE_PLAN_NOT_VALIDATED")
        if now > candidate.valid_until:
            raise ValueError("CANDIDATE_PLAN_EXPIRED")
        if (
            not risk_verdict.approved
            or not risk_check_id.strip()
            or not risk_config_version.strip()
        ):
            raise ValueError("FINAL_TRADE_PLAN_RISK_NOT_APPROVED")
        quantity = float(
            risk_verdict.suggested_qty or candidate.proposed_qty
        )
        if quantity <= 0 or quantity > candidate.proposed_qty:
            raise ValueError("FINAL_TRADE_PLAN_RISK_QUANTITY_INVALID")
        final_id = self._stable_id(
            "final-plan",
            candidate.candidate_plan_id,
            risk_check_id,
            risk_config_version,
            f"{quantity:.8f}",
        )
        final = FinalTradePlan(
            final_plan_id=final_id,
            version=1,
            candidate_plan_id=candidate.candidate_plan_id,
            symbol=candidate.symbol,
            side=candidate.side,
            action=candidate.action,
            entry_price=candidate.entry_price,
            stop_loss=candidate.stop_loss,
            take_profit=candidate.take_profit,
            qty=quantity,
            decision_id=candidate.decision_id,
            risk_check_id=risk_check_id,
            risk_config_version=risk_config_version,
            strategy_version=candidate.strategy_version,
            data_version=candidate.data_version,
            evidence_refs=candidate.evidence_refs,
            created_at=now,
            valid_until=candidate.valid_until,
        )
        connection = self._connect()
        try:
            connection.execute("BEGIN TRANSACTION")
            changed = connection.execute(
                """
                UPDATE candidate_plans
                SET status=?, updated_at=?
                WHERE candidate_plan_id=? AND status=?
                RETURNING candidate_plan_id
                """,
                [
                    CandidatePlanStatus.FINALIZED.value,
                    now,
                    candidate.candidate_plan_id,
                    CandidatePlanStatus.VALIDATED.value,
                ],
            ).fetchone()
            if changed is None:
                raise ValueError("CANDIDATE_PLAN_TRANSITION_INVALID")
            connection.execute(
                """
                INSERT INTO final_trade_plans VALUES
                (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                [
                    final.final_plan_id,
                    final.version,
                    final.candidate_plan_id,
                    final.symbol,
                    final.side.value,
                    final.action,
                    final.entry_price,
                    final.stop_loss,
                    final.take_profit,
                    final.qty,
                    final.decision_id,
                    final.risk_check_id,
                    final.risk_config_version,
                    final.strategy_version,
                    final.data_version,
                    json.dumps(list(final.evidence_refs)),
                    final.created_at,
                    final.valid_until,
                    final.status.value,
                    final.created_at,
                ],
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        return final

    def create_order_intent(
        self,
        final_plan_id: str,
        *,
        now: datetime,
    ) -> OrderIntent:
        final = self.get_final(final_plan_id)
        if final is None:
            raise KeyError(final_plan_id)
        if final.status != FinalTradePlanStatus.FINAL_EXECUTABLE:
            raise ValueError("FINAL_TRADE_PLAN_TRANSITION_INVALID")
        if now > final.valid_until:
            self._set_final_status(
                final_plan_id,
                FinalTradePlanStatus.EXPIRED,
                now,
            )
            raise ValueError("FINAL_TRADE_PLAN_EXPIRED")
        key = idempotency_key(
            final.final_plan_id,
            final.symbol,
            final.side.value,
            final.qty,
            final.entry_price,
            final.action,
        )
        intent_id = self._stable_id("intent", key)
        intent = self._intent_from_final(
            final,
            intent_id=intent_id,
            key=key,
        )
        connection = self._connect()
        try:
            connection.execute("BEGIN TRANSACTION")
            connection.execute(
                """
                INSERT INTO execution_plan_intents VALUES (?,?,?,?)
                """,
                [final.final_plan_id, intent.intent_id, key, now],
            )
            changed = connection.execute(
                """
                UPDATE final_trade_plans
                SET status=?, updated_at=?
                WHERE final_plan_id=? AND status=?
                RETURNING final_plan_id
                """,
                [
                    FinalTradePlanStatus.ORDER_INTENT_CREATED.value,
                    now,
                    final.final_plan_id,
                    FinalTradePlanStatus.FINAL_EXECUTABLE.value,
                ],
            ).fetchone()
            if changed is None:
                raise ValueError("FINAL_TRADE_PLAN_TRANSITION_INVALID")
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        return intent

    def get_candidate(self, candidate_plan_id: str) -> CandidatePlan | None:
        connection = self._connect(read_only=True)
        try:
            row = connection.execute(
                """
                SELECT * FROM candidate_plans WHERE candidate_plan_id=?
                """,
                [candidate_plan_id],
            ).fetchone()
        finally:
            connection.close()
        return self._candidate_from_row(row) if row else None

    def get_final(self, final_plan_id: str) -> FinalTradePlan | None:
        connection = self._connect(read_only=True)
        try:
            row = connection.execute(
                """
                SELECT * FROM final_trade_plans WHERE final_plan_id=?
                """,
                [final_plan_id],
            ).fetchone()
        finally:
            connection.close()
        return self._final_from_row(row) if row else None

    def get_final_by_candidate(
        self,
        candidate_plan_id: str,
    ) -> FinalTradePlan | None:
        connection = self._connect(read_only=True)
        try:
            row = connection.execute(
                """
                SELECT * FROM final_trade_plans
                WHERE candidate_plan_id=?
                """,
                [candidate_plan_id],
            ).fetchone()
        finally:
            connection.close()
        return self._final_from_row(row) if row else None

    def get_intent_for_final(
        self,
        final_plan_id: str,
    ) -> OrderIntent | None:
        final = self.get_final(final_plan_id)
        if final is None:
            return None
        connection = self._connect(read_only=True)
        try:
            row = connection.execute(
                """
                SELECT intent_id, idempotency_key
                FROM execution_plan_intents
                WHERE final_plan_id=?
                """,
                [final_plan_id],
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        return self._intent_from_final(
            final,
            intent_id=str(row[0]),
            key=str(row[1]),
        )

    def _set_candidate_status(
        self,
        candidate_plan_id: str,
        status: CandidatePlanStatus,
        now: datetime,
    ) -> None:
        connection = self._connect()
        try:
            connection.execute(
                """
                UPDATE candidate_plans SET status=?, updated_at=?
                WHERE candidate_plan_id=?
                """,
                [status.value, now, candidate_plan_id],
            )
            connection.commit()
        finally:
            connection.close()

    def _set_final_status(
        self,
        final_plan_id: str,
        status: FinalTradePlanStatus,
        now: datetime,
    ) -> None:
        connection = self._connect()
        try:
            connection.execute(
                """
                UPDATE final_trade_plans SET status=?, updated_at=?
                WHERE final_plan_id=?
                """,
                [status.value, now, final_plan_id],
            )
            connection.commit()
        finally:
            connection.close()

    @staticmethod
    def _candidate_from_row(row: tuple) -> CandidatePlan:
        return CandidatePlan(
            candidate_plan_id=str(row[0]),
            symbol=str(row[1]),
            side=Side(str(row[2])),
            action=str(row[3]),
            entry_price=float(row[4]),
            stop_loss=float(row[5]),
            take_profit=float(row[6]),
            proposed_qty=float(row[7]),
            decision_id=str(row[8]),
            strategy_version=str(row[9]),
            data_version=str(row[10]),
            evidence_refs=tuple(json.loads(row[11] or "[]")),
            created_at=row[12],
            valid_until=row[13],
            status=CandidatePlanStatus(str(row[14])),
        )

    @staticmethod
    def _final_from_row(row: tuple) -> FinalTradePlan:
        return FinalTradePlan(
            final_plan_id=str(row[0]),
            version=int(row[1]),
            candidate_plan_id=str(row[2]),
            symbol=str(row[3]),
            side=Side(str(row[4])),
            action=str(row[5]),
            entry_price=float(row[6]),
            stop_loss=float(row[7]),
            take_profit=float(row[8]),
            qty=float(row[9]),
            decision_id=str(row[10]),
            risk_check_id=str(row[11]),
            risk_config_version=str(row[12]),
            strategy_version=str(row[13]),
            data_version=str(row[14]),
            evidence_refs=tuple(json.loads(row[15] or "[]")),
            created_at=row[16],
            valid_until=row[17],
            status=FinalTradePlanStatus(str(row[18])),
        )

    @staticmethod
    def _stable_id(prefix: str, *parts: str) -> str:
        return prefix + "-" + hashlib.sha256(
            "|".join(parts).encode()
        ).hexdigest()[:24]

    @staticmethod
    def _intent_from_final(
        final: FinalTradePlan,
        *,
        intent_id: str,
        key: str,
    ) -> OrderIntent:
        return OrderIntent(
            intent_id=intent_id,
            signal_id=final.decision_id,
            symbol=final.symbol,
            side=final.side,
            qty=final.qty,
            order_type="LMT",
            limit_price=marketable_limit_price(
                final.entry_price, final.side, action=final.action
            ),
            # reference_price 保留计划里那个"干净"的价格（AI/ATR 算出来的，
            # 没有掺执行层缓冲）——止损风险计算、审计、回放都锚定这个值，只有
            # 真正提交给 broker 的 limit_price 加了成交概率缓冲。
            reference_price=final.entry_price,
            idempotency_key=key,
            client_order_id=client_order_id(key),
            decision_id=final.decision_id,
            plan_id=final.final_plan_id,
            candidate_plan_id=final.candidate_plan_id,
            final_plan_id=final.final_plan_id,
            final_plan_version=final.version,
            risk_check_id=final.risk_check_id,
            evidence_refs=final.evidence_refs,
        )


def candidate_from_trade_plan(
    plan: TradePlan,
    *,
    decision_id: str,
    strategy_version: str,
    data_version: str,
    evidence_refs: tuple[str, ...],
    valid_until: datetime,
) -> CandidatePlan:
    raw = "|".join(
        (
            plan.plan_id,
            decision_id,
            strategy_version,
            data_version,
            plan.symbol,
            plan.side.value,
            plan.action,
        )
    )
    candidate_id = "candidate-plan-" + hashlib.sha256(
        raw.encode()
    ).hexdigest()[:24]
    return CandidatePlan(
        candidate_plan_id=candidate_id,
        symbol=plan.symbol,
        side=plan.side,
        action=plan.action,
        entry_price=plan.entry_price,
        stop_loss=plan.stop_loss,
        take_profit=plan.take_profit,
        proposed_qty=plan.qty,
        decision_id=decision_id,
        strategy_version=strategy_version,
        data_version=data_version,
        evidence_refs=evidence_refs,
        created_at=plan.created_at,
        valid_until=valid_until,
    )
