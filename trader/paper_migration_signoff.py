"""Immutable architecture and FINAL_REAL Paper migration sign-offs."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb

_SIGNOFF_TYPES = {"ARCHITECTURE", "FINAL_REAL"}
_CALL_GRAPH_NODES = {
    "ResearchSnapshot",
    "OrderIntent",
    "PositionPlan",
    "EpisodeReview",
    "StrategyCandidate",
}
_REQUIRED_DOCS = {
    "task_board",
    "closed_loop_acceptance",
    "maturity_runbook",
    "signoff_report",
}


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


class PaperMigrationSignoffStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        connection = duckdb.connect(self.db_path)
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS paper_migration_signoffs (
                    signoff_id TEXT PRIMARY KEY,
                    signoff_type TEXT,
                    maturity_report_id TEXT,
                    maturity_evidence_type TEXT,
                    resilience_run_id TEXT,
                    resilience_evidence_type TEXT,
                    closed_loop_evidence_id TEXT,
                    verification_json TEXT,
                    documents_json TEXT,
                    call_graph TEXT,
                    limitations_json TEXT,
                    status TEXT,
                    content_hash TEXT,
                    created_at TIMESTAMPTZ
                )
                """
            )
            connection.commit()
        finally:
            connection.close()

    def record(
        self,
        *,
        signoff_type: str,
        maturity_report: dict[str, Any],
        resilience_report: dict[str, Any],
        closed_loop_evidence: dict[str, Any],
        verification: dict[str, Any],
        documents: dict[str, str],
        call_graph: str,
        limitations: list[str],
        live_authorized: bool,
        created_at: datetime,
    ) -> dict[str, Any]:
        normalized = signoff_type.strip().upper()
        if normalized not in _SIGNOFF_TYPES:
            raise ValueError("PAPER_SIGNOFF_TYPE_INVALID")
        if created_at.tzinfo is None or created_at.utcoffset() is None:
            raise ValueError("PAPER_SIGNOFF_TIME_TZ_REQUIRED")
        if live_authorized:
            raise ValueError("PAPER_SIGNOFF_LIVE_AUTHORIZATION_FORBIDDEN")
        if not maturity_report.get("ready"):
            raise ValueError("PAPER_SIGNOFF_MATURITY_NOT_READY")
        if not resilience_report.get("passed"):
            raise ValueError("PAPER_SIGNOFF_RESILIENCE_NOT_READY")
        if not closed_loop_evidence.get("accepted"):
            raise ValueError("PAPER_SIGNOFF_CLOSED_LOOP_NOT_ACCEPTED")
        maturity_type = str(maturity_report.get("evidence_type", "")).upper()
        resilience_type = str(
            resilience_report.get("evidence_type", "")
        ).upper()
        if maturity_type not in {"REAL", "SYNTHETIC"} or resilience_type not in {
            "REAL",
            "SYNTHETIC",
        }:
            raise ValueError("PAPER_SIGNOFF_EVIDENCE_TYPE_INVALID")
        if normalized == "FINAL_REAL" and (
            maturity_type != "REAL"
            or resilience_type != "REAL"
            or int(maturity_report.get("required_sessions", 0)) < 60
            or int(maturity_report.get("observed_sessions", 0)) < 60
        ):
            raise ValueError("PAPER_SIGNOFF_FINAL_REAL_EVIDENCE_REQUIRED")
        if (
            int(verification.get("pytest_passed", 0)) < 1
            or verification.get("ruff_passed") is not True
            or verification.get("compileall_passed") is not True
        ):
            raise ValueError("PAPER_SIGNOFF_VERIFICATION_INCOMPLETE")
        if _REQUIRED_DOCS - set(documents) or any(
            not str(documents[key]).strip() for key in _REQUIRED_DOCS
        ):
            raise ValueError("PAPER_SIGNOFF_DOCUMENTATION_INCOMPLETE")
        if any(node not in call_graph for node in _CALL_GRAPH_NODES):
            raise ValueError("PAPER_SIGNOFF_CALL_GRAPH_INCOMPLETE")
        normalized_limitations = [
            str(item).strip() for item in limitations if str(item).strip()
        ]
        if not normalized_limitations:
            raise ValueError("PAPER_SIGNOFF_LIMITATIONS_REQUIRED")
        status = (
            "FINAL_REAL_READY"
            if normalized == "FINAL_REAL"
            else "ARCHITECTURE_READY"
        )
        payload = {
            "signoff_type": normalized,
            "maturity_report_id": maturity_report["report_id"],
            "maturity_evidence_type": maturity_type,
            "resilience_run_id": resilience_report["drill_run_id"],
            "resilience_evidence_type": resilience_type,
            "closed_loop_evidence_id": closed_loop_evidence["evidence_id"],
            "verification": verification,
            "documents": documents,
            "call_graph": call_graph,
            "limitations": normalized_limitations,
            "live_authorized": False,
            "status": status,
        }
        content_hash = hashlib.sha256(_canonical(payload).encode()).hexdigest()
        signoff_id = "paper-signoff-" + content_hash[:24]
        connection = duckdb.connect(self.db_path)
        try:
            connection.execute(
                """
                INSERT INTO paper_migration_signoffs VALUES
                (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(signoff_id) DO NOTHING
                """,
                [
                    signoff_id,
                    normalized,
                    payload["maturity_report_id"],
                    maturity_type,
                    payload["resilience_run_id"],
                    resilience_type,
                    payload["closed_loop_evidence_id"],
                    _canonical(verification),
                    _canonical(documents),
                    call_graph,
                    _canonical(normalized_limitations),
                    status,
                    content_hash,
                    created_at,
                ],
            )
            connection.commit()
        finally:
            connection.close()
        return {**payload, "signoff_id": signoff_id}


def paper_call_graph() -> str:
    return (
        "ResearchSnapshot -> TradingAgentsResearch -> CandidatePlan -> "
        "FinalTradePlan -> RiskEvent -> OrderIntent -> BrokerOrder -> Fill -> "
        "PositionPlan -> TradeEpisode -> EpisodeReview -> StrategyCandidate -> "
        "PromotionEvidence -> PaperMaturity -> PaperMigrationSignoff"
    )
