"""Daily stock screening and TradingAgents research batches.

The batch is analysis-only: it never imports a broker and never submits orders.
Runtime consumes the immutable completed report during the following session.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import os
import re
import subprocess
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time as wall_time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import duckdb
from dotenv import load_dotenv

from .ai.safety import AIScoreSnapshot
from .daily_candidates import DailyCandidate
from .models import (
    ResearchQuality,
    ResearchSnapshot,
    ResearchSourceManifestEntry,
    ResearchSourceStatus,
)
from .research_screening import build_research_candidates as build_daily_candidates
from .research_snapshot import (
    ResearchSnapshotStore,
    snapshot_content_hash,
)
from .research_snapshot_shadow import (
    build_screening_shadow_snapshot,
    compare_candidate_to_snapshot,
)

logger = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")
_UTC = timezone.utc
_SUCCESS_STATES = {"COMPLETED", "COMPLETED_WITH_ERRORS"}
_WORKER_RESULT_PREFIX = "__TRADINGAGENTS_RESULT__="
_LOW_CONFIDENCE_RESEARCH_RISK = (
    "本地分时数据不足：本次深度研究由 TradingAgents 独立获取市场资料"
)
_FAILED_RUN_UNCLASSIFIED = "DAILY_RESEARCH_FAILED_UNCLASSIFIED"
_FAILED_ITEM_UNCLASSIFIED = "DAILY_RESEARCH_ITEM_FAILED_UNCLASSIFIED"
_FAILED_SNAPSHOT_UNCLASSIFIED = "RESEARCH_SNAPSHOT_FAILED_UNCLASSIFIED"
_INTERRUPTED_RUN_ERROR = "DAILY_RESEARCH_INTERRUPTED"


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("UTC_TIMESTAMP_REQUIRED")
    return value.astimezone(_UTC)


def _json_value(value: Any, depth: int = 0) -> Any:
    """Convert third-party graph results into bounded JSON-compatible values."""
    if depth > 8:
        return "<max-depth>"
    if isinstance(value, str):
        return value[:20_000]
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, datetime):
        return _utc(value).isoformat()
    if isinstance(value, dict):
        return {
            str(key): _json_value(item, depth + 1)
            for key, item in list(value.items())[:200]
        }
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item, depth + 1) for item in list(value)[:200]]
    if hasattr(value, "model_dump"):
        return _json_value(value.model_dump(), depth + 1)
    if hasattr(value, "__dict__"):
        return _json_value(vars(value), depth + 1)
    return str(value)[:20_000]


def _stable_error_code(exc: Exception) -> str:
    """Return a bounded diagnostic code without persisting free-form secrets."""
    message = str(exc).strip()
    if message and re.fullmatch(r"[A-Za-z0-9_.:-]{1,160}", message):
        return message
    if isinstance(exc, TimeoutError):
        return f"TIMEOUT:{type(exc).__name__}"
    if isinstance(exc, ConnectionError):
        return f"CONNECTION_ERROR:{type(exc).__name__}"
    if isinstance(exc, OSError):
        return f"IO_ERROR:{type(exc).__name__}"
    return type(exc).__name__ or "UNKNOWN_ERROR"


def _next_weekday(value: date) -> date:
    candidate = value + timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return candidate


def research_target_date(
    now: datetime,
    *,
    close_hour_et: int = 16,
    close_minute_et: int = 15,
) -> str:
    """Return the trading date produced by a batch started at ``now``.

    A post-close batch is prepared for the next weekday. A pre-market/manual
    batch belongs to the current weekday. Exchange holidays remain governed by
    Runtime's market calendar; the store never silently reuses another date.
    """
    local = _utc(now).astimezone(_ET)
    current = local.date()
    if local.weekday() >= 5:
        while current.weekday() >= 5:
            current += timedelta(days=1)
        return current.isoformat()
    if local.time() >= wall_time(close_hour_et, close_minute_et):
        return _next_weekday(current).isoformat()
    return current.isoformat()


def in_daily_run_window(
    now: datetime,
    *,
    close_hour_et: int = 16,
    close_minute_et: int = 15,
) -> bool:
    local = _utc(now).astimezone(_ET)
    if local.weekday() >= 5:
        return False
    current = local.time()
    premarket = wall_time(6, 0) <= current <= wall_time(9, 15)
    postclose = current >= wall_time(close_hour_et, close_minute_et)
    return premarket or postclose


@dataclass(frozen=True)
class ResearchAnalysis:
    recommendation: str
    score: float
    confidence: float
    thesis: str
    risks: list[str] = field(default_factory=list)
    provider: str = "tradingagents"
    model: str = ""
    raw: Any = field(default_factory=dict)
    source_manifest: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class TradingAgentsInvocation:
    invocation_id: str
    run_id: str
    snapshot_id: str
    snapshot_content_hash: str
    data_version: str
    model_version: str
    symbol: str
    trading_date: str
    data_cutoff: datetime
    requested_at: datetime

    def __post_init__(self) -> None:
        required = (
            self.invocation_id,
            self.run_id,
            self.snapshot_id,
            self.snapshot_content_hash,
            self.data_version,
            self.model_version,
            self.symbol,
            self.trading_date,
        )
        if not all(str(value).strip() for value in required):
            raise ValueError("TRADINGAGENTS_INVOCATION_FIELD_REQUIRED")
        if _utc(self.data_cutoff) > _utc(self.requested_at):
            raise ValueError("TRADINGAGENTS_INVOCATION_CUTOFF_FUTURE")

    def to_dict(self) -> dict[str, str]:
        return {
            "invocation_id": self.invocation_id,
            "run_id": self.run_id,
            "snapshot_id": self.snapshot_id,
            "snapshot_content_hash": self.snapshot_content_hash,
            "data_version": self.data_version,
            "model_version": self.model_version,
            "symbol": self.symbol,
            "trading_date": self.trading_date,
            "data_cutoff": _utc(self.data_cutoff).isoformat(),
            "requested_at": _utc(self.requested_at).isoformat(),
        }


@dataclass(frozen=True)
class DailyResearchItem:
    run_id: str
    trading_date: str
    symbol: str
    rank: int
    screening_score: float
    screening_status: str
    status: str
    recommendation: str = "WATCH"
    ai_score: float | None = None
    confidence: float | None = None
    thesis: str = ""
    risks: tuple[str, ...] = ()
    provider: str = ""
    model: str = ""
    error_code: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(_UTC))
    completed_at: datetime | None = None
    raw: Any = field(default_factory=dict)
    snapshot_id: str = ""
    data_version: str = ""
    model_version: str = ""
    invocation_id: str = ""
    ta_snapshot_id: str = ""


@dataclass(frozen=True)
class DailyResearchRun:
    run_id: str
    trading_date: str
    status: str
    universe_version: str
    data_cutoff: datetime
    timeframe: str
    screen_limit: int
    deep_limit: int
    provider: str
    model: str
    total_symbols: int
    completed_symbols: int
    failed_symbols: int
    started_at: datetime
    completed_at: datetime | None = None
    error_code: str = ""
    config_version: str = ""


class TradingAgentsAdapter:
    """Thin, lazy wrapper around TauricResearch/TradingAgents.

    The optional dependency is deliberately not hidden behind a synthetic
    score. Missing modules or provider failures are persisted as batch errors.
    """

    provider = "tradingagents"

    def __init__(
        self,
        *,
        project_dir: str = "",
        llm_provider: str = "",
        deep_model: str = "",
        quick_model: str = "",
        backend_url: str = "",
        python_executable: str = "",
        cache_dir: str = "",
        results_dir: str = "",
        memory_log_path: str = "",
        timeout_seconds: int | None = None,
        debug: bool = False,
    ) -> None:
        self.project_dir = project_dir or os.getenv("TRADINGAGENTS_PROJECT_DIR", "")
        self.llm_provider = llm_provider or os.getenv("TRADINGAGENTS_LLM_PROVIDER", "")
        self.deep_model = deep_model or os.getenv("TRADINGAGENTS_DEEP_MODEL", "")
        self.quick_model = quick_model or os.getenv("TRADINGAGENTS_QUICK_MODEL", "")
        self.backend_url = backend_url or os.getenv("TRADINGAGENTS_BACKEND_URL", "")
        self.python_executable = python_executable or os.getenv(
            "TRADINGAGENTS_PYTHON", ""
        )
        self.cache_dir = cache_dir or os.getenv("TRADINGAGENTS_CACHE_DIR", "")
        self.results_dir = results_dir or os.getenv(
            "TRADINGAGENTS_RESULTS_DIR", ""
        )
        self.memory_log_path = memory_log_path or os.getenv(
            "TRADINGAGENTS_MEMORY_LOG_PATH", ""
        )
        configured_timeout = os.getenv("TRADINGAGENTS_TIMEOUT_SECONDS", "7200")
        self.timeout_seconds = int(
            timeout_seconds if timeout_seconds is not None else configured_timeout
        )
        self.debug = debug
        self.model = self.deep_model or self.quick_model

    def describe(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "project_dir": self.project_dir,
            "llm_provider": self.llm_provider,
            "deep_model": self.deep_model,
            "quick_model": self.quick_model,
            "backend_url": self.backend_url,
            "python_executable": self.python_executable,
            "cache_dir": self.cache_dir,
            "results_dir": self.results_dir,
            "memory_log_path": self.memory_log_path,
            "timeout_seconds": self.timeout_seconds,
        }

    def _load(self):
        if self.project_dir:
            resolved = str(Path(self.project_dir).expanduser().resolve())
            if resolved not in sys.path:
                sys.path.insert(0, resolved)
        try:
            graph_module = importlib.import_module("tradingagents.graph.trading_graph")
            config_module = importlib.import_module("tradingagents.default_config")
        except Exception as exc:
            raise RuntimeError("TRADINGAGENTS_MODULE_UNAVAILABLE") from exc
        return graph_module.TradingAgentsGraph, config_module.DEFAULT_CONFIG

    def analyze(self, symbol: str, trading_date: str) -> ResearchAnalysis:
        if self.python_executable:
            return self._analyze_subprocess(symbol, trading_date)
        graph_cls, defaults = self._load()
        config = dict(defaults)
        if self.llm_provider:
            config["llm_provider"] = self.llm_provider
        if self.deep_model:
            config["deep_think_llm"] = self.deep_model
        if self.quick_model:
            config["quick_think_llm"] = self.quick_model
        if self.backend_url:
            config["backend_url"] = self.backend_url
        graph = graph_cls(debug=self.debug, config=config)
        state, decision = graph.propagate(symbol, trading_date)
        raw = {"decision": _json_value(decision), "state": _json_value(state)}
        recommendation = _recommendation(decision)
        confidence = _confidence(decision)
        score = _score(recommendation, confidence)
        thesis = _thesis(decision)
        risks = _risks(decision)
        return ResearchAnalysis(
            recommendation=recommendation,
            score=score,
            confidence=confidence,
            thesis=thesis,
            risks=risks,
            provider=self.provider,
            model=self.model,
            raw=raw,
        )

    def analyze_with_context(
        self,
        symbol: str,
        trading_date: str,
        invocation: TradingAgentsInvocation,
    ) -> ResearchAnalysis:
        if symbol.strip().upper() != invocation.symbol:
            raise ValueError("TRADINGAGENTS_INVOCATION_SYMBOL_MISMATCH")
        if trading_date != invocation.trading_date:
            raise ValueError("TRADINGAGENTS_INVOCATION_DATE_MISMATCH")
        if self.python_executable:
            return self._analyze_subprocess(
                symbol,
                trading_date,
                invocation=invocation,
            )
        analysis = self.analyze(symbol, trading_date)
        return ResearchAnalysis(
            **{
                **analysis.__dict__,
                "raw": {
                    **dict(analysis.raw or {}),
                    "invocation": invocation.to_dict(),
                    "worker_contract_version": 1,
                },
            }
        )

    def _analyze_subprocess(
        self,
        symbol: str,
        trading_date: str,
        *,
        invocation: TradingAgentsInvocation | None = None,
    ) -> ResearchAnalysis:
        python_path = Path(self.python_executable).expanduser().resolve()
        if not python_path.is_file():
            raise RuntimeError("TRADINGAGENTS_PYTHON_UNAVAILABLE")
        worker_path = Path(__file__).with_name("tradingagents_worker.py").resolve()
        payload = {
            "symbol": symbol,
            "trading_date": trading_date,
            "llm_provider": self.llm_provider,
            "deep_model": self.deep_model,
            "quick_model": self.quick_model,
            "backend_url": self.backend_url,
            "cache_dir": self.cache_dir,
            "results_dir": self.results_dir,
            "memory_log_path": self.memory_log_path,
            "debug": self.debug,
            "worker_contract_version": 1,
        }
        if invocation is not None:
            payload["invocation"] = invocation.to_dict()
        try:
            completed = subprocess.run(
                [str(python_path), str(worker_path)],
                input=json.dumps(payload, ensure_ascii=False),
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                cwd=self.project_dir or None,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("TRADINGAGENTS_TIMEOUT") from exc
        except OSError as exc:
            raise RuntimeError("TRADINGAGENTS_WORKER_UNAVAILABLE") from exc

        result = self._worker_result(completed.stdout)
        if completed.returncode != 0 or not result.get("ok"):
            error = re.sub(
                r"[^A-Za-z0-9_.-]",
                "_",
                str(result.get("error") or "UNKNOWN"),
            )[:80]
            diagnostic = re.sub(
                r"[^A-Z0-9_]",
                "_",
                str(result.get("diagnostic_code") or ""),
            )[:80].strip("_")
            suffix = f":{diagnostic}" if diagnostic else ""
            raise RuntimeError(
                f"TRADINGAGENTS_WORKER_FAILED:{error}{suffix}"
            )
        if invocation is not None:
            self._validate_worker_contract(result, invocation)
        if "decision" not in result or result.get("decision") is None:
            raise RuntimeError("TRADINGAGENTS_WORKER_OUTPUT_INVALID")
        decision = result.get("decision")
        raw = {
            "decision": _json_value(decision),
            "state": _json_value(result.get("state")),
            "invocation": _json_value(result.get("invocation")),
            "worker_contract_version": result.get(
                "worker_contract_version"
            ),
            "worker_started_at": result.get("worker_started_at"),
            "worker_completed_at": result.get("worker_completed_at"),
            "source_manifest": _json_value(result.get("source_manifest")),
        }
        source_manifest = self._validate_source_manifest(
            result.get("source_manifest")
        ) if invocation is not None else ()
        recommendation = _recommendation(decision)
        confidence = _confidence(decision)
        return ResearchAnalysis(
            recommendation=recommendation,
            score=_score(recommendation, confidence),
            confidence=confidence,
            thesis=_thesis(decision),
            risks=_risks(decision),
            provider=self.provider,
            model=self.model,
            raw=raw,
            source_manifest=source_manifest,
        )

    @staticmethod
    def _validate_worker_contract(
        result: dict[str, Any],
        invocation: TradingAgentsInvocation,
    ) -> None:
        if result.get("worker_contract_version") != 1:
            raise RuntimeError("TRADINGAGENTS_WORKER_CONTRACT_INVALID")
        returned = result.get("invocation")
        if not isinstance(returned, dict):
            raise RuntimeError("TRADINGAGENTS_WORKER_INVOCATION_MISSING")
        if returned != invocation.to_dict():
            raise RuntimeError("TRADINGAGENTS_WORKER_INVOCATION_MISMATCH")
        try:
            started_at = datetime.fromisoformat(
                str(result["worker_started_at"])
            )
            completed_at = datetime.fromisoformat(
                str(result["worker_completed_at"])
            )
        except (KeyError, ValueError) as exc:
            raise RuntimeError(
                "TRADINGAGENTS_WORKER_TIMESTAMP_INVALID"
            ) from exc
        started_at = _utc(started_at)
        completed_at = _utc(completed_at)
        if started_at < _utc(invocation.requested_at):
            raise RuntimeError("TRADINGAGENTS_WORKER_OUTPUT_STALE")
        if completed_at < started_at:
            raise RuntimeError("TRADINGAGENTS_WORKER_TIMESTAMP_INVALID")

    @staticmethod
    def _validate_source_manifest(
        value: Any,
    ) -> tuple[dict[str, Any], ...]:
        if not isinstance(value, list) or not value:
            raise RuntimeError("TRADINGAGENTS_SOURCE_MANIFEST_MISSING")
        required = {
            "source",
            "status",
            "as_of",
            "fetched_at",
            "quality_score",
            "coverage",
            "payload_version",
            "failure_code",
            "metadata",
        }
        result = []
        for raw in value:
            if not isinstance(raw, dict) or not required.issubset(raw):
                raise RuntimeError("TRADINGAGENTS_SOURCE_MANIFEST_INVALID")
            try:
                ResearchSourceManifestEntry(
                    source=str(raw["source"]),
                    status=ResearchSourceStatus(str(raw["status"])),
                    as_of=datetime.fromisoformat(str(raw["as_of"])),
                    fetched_at=datetime.fromisoformat(
                        str(raw["fetched_at"])
                    ),
                    quality_score=float(raw["quality_score"]),
                    coverage=tuple(raw["coverage"]),
                    payload_version=str(raw["payload_version"]),
                    failure_code=str(raw["failure_code"]),
                    metadata=dict(raw["metadata"]),
                )
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "TRADINGAGENTS_SOURCE_MANIFEST_INVALID"
                ) from exc
            result.append(dict(raw))
        return tuple(result)

    @staticmethod
    def _worker_result(stdout: str) -> dict[str, Any]:
        for line in reversed(stdout.splitlines()):
            if line.startswith(_WORKER_RESULT_PREFIX):
                try:
                    value = json.loads(line[len(_WORKER_RESULT_PREFIX) :])
                except json.JSONDecodeError as exc:
                    raise RuntimeError("TRADINGAGENTS_WORKER_INVALID_JSON") from exc
                if isinstance(value, dict):
                    return value
        raise RuntimeError("TRADINGAGENTS_WORKER_NO_RESULT")


def _decision_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        preferred = (
            "final_trade_decision",
            "recommendation",
            "decision",
            "action",
            "signal",
            "investment_plan",
        )
        for key in preferred:
            if key in value:
                return _decision_text(value[key])
        return json.dumps(_json_value(value), ensure_ascii=False)
    return str(value or "")


def _recommendation(value: Any) -> str:
    text = _decision_text(value).upper()
    explicit = re.findall(
        r"(?:FINAL|RECOMMENDATION|DECISION|ACTION)[^A-Z]{0,20}"
        r"(BUY|SELL|HOLD|LONG|SHORT|AVOID|OVERWEIGHT|UNDERWEIGHT)",
        text,
    )
    leading = re.match(r"\s*(BUY|SELL|HOLD|LONG|SHORT|AVOID|OVERWEIGHT|UNDERWEIGHT)\b", text)
    tokens = (
        explicit
        or ([leading.group(1)] if leading else [])
        or re.findall(r"\b(BUY|SELL|HOLD|LONG|SHORT|AVOID|OVERWEIGHT|UNDERWEIGHT)\b", text)
    )
    if not tokens:
        return "HOLD"
    final = tokens[-1]
    if final in {"BUY", "LONG", "OVERWEIGHT"}:
        return "BUY"
    if final in {"SELL", "SHORT", "AVOID", "UNDERWEIGHT"}:
        return "SELL"
    return "HOLD"


def _find_number(value: Any, keys: tuple[str, ...]) -> float | None:
    if isinstance(value, dict):
        for key in keys:
            raw = value.get(key)
            if isinstance(raw, (int, float)):
                return float(raw)
        for child in value.values():
            found = _find_number(child, keys)
            if found is not None:
                return found
    return None


def _confidence(value: Any) -> float:
    found = _find_number(value, ("confidence", "probability", "conviction"))
    if found is None:
        rating = _decision_text(value).strip().upper()
        if rating in {"BUY", "SELL"}:
            return 0.85
        if rating in {"OVERWEIGHT", "UNDERWEIGHT"}:
            return 0.7
        return 0.65
    if found > 1:
        found /= 100.0
    return max(0.0, min(1.0, found))


def _score(recommendation: str, confidence: float) -> float:
    if recommendation == "BUY":
        return round(50.0 + 50.0 * confidence, 1)
    if recommendation == "SELL":
        return round(50.0 - 50.0 * confidence, 1)
    return round(50.0 + 10.0 * (confidence - 0.5), 1)


def _thesis(value: Any) -> str:
    """研究结论原文，不在这里裁剪。

    原来这里 ``text[:4000]`` 是想防 Discord 超限，但位置不对：thesis 会进
    embed 的 field.value，那里的上限是 1024 而不是 4096，4000 根本挡不住；
    而且 thesis 还要落库供决策台展示，在数据层就砍掉等于永久丢失。长度整形
    交给推送层（discord_limits + build_daily_research_message 的预算分配）。
    """
    return _decision_text(value).strip()


def _risks(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return []
    for key in ("risks", "risk_factors", "concerns"):
        raw = value.get(key)
        if isinstance(raw, list):
            return [str(item)[:500] for item in raw[:8]]
        if isinstance(raw, str) and raw.strip():
            return [raw.strip()[:500]]
    return []


class DailyResearchStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._init_db()

    def _connect(self, *, read_only: bool = False):
        for attempt in range(5):
            try:
                return duckdb.connect(self.db_path, read_only=read_only)
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(0.1 * (attempt + 1))

    def _init_db(self) -> None:
        con = self._connect()
        try:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS daily_research_runs (
                    run_id VARCHAR PRIMARY KEY,
                    trading_date VARCHAR,
                    status VARCHAR,
                    universe_version VARCHAR,
                    data_cutoff TIMESTAMPTZ,
                    timeframe VARCHAR,
                    screen_limit INTEGER,
                    deep_limit INTEGER,
                    provider VARCHAR,
                    model VARCHAR,
                    total_symbols INTEGER,
                    completed_symbols INTEGER,
                    failed_symbols INTEGER,
                    started_at TIMESTAMPTZ,
                    completed_at TIMESTAMPTZ,
                    error_code VARCHAR,
                    config_version VARCHAR
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS daily_research_items (
                    run_id VARCHAR,
                    trading_date VARCHAR,
                    symbol VARCHAR,
                    rank INTEGER,
                    screening_score DOUBLE,
                    screening_status VARCHAR,
                    status VARCHAR,
                    recommendation VARCHAR,
                    ai_score DOUBLE,
                    confidence DOUBLE,
                    thesis VARCHAR,
                    risks_json VARCHAR,
                    provider VARCHAR,
                    model VARCHAR,
                    error_code VARCHAR,
                    created_at TIMESTAMPTZ,
                    completed_at TIMESTAMPTZ,
                    raw_json VARCHAR,
                    snapshot_id VARCHAR,
                    data_version VARCHAR,
                    model_version VARCHAR,
                    invocation_id VARCHAR,
                    ta_snapshot_id VARCHAR,
                    PRIMARY KEY(run_id, symbol)
                )
                """
            )
            columns = {
                str(row[1])
                for row in con.execute(
                    "PRAGMA table_info('daily_research_items')"
                ).fetchall()
            }
            for name in (
                "snapshot_id",
                "data_version",
                "model_version",
                "invocation_id",
                "ta_snapshot_id",
            ):
                if name not in columns:
                    con.execute(
                        f"ALTER TABLE daily_research_items "
                        f"ADD COLUMN {name} VARCHAR"
                    )
            con.execute(
                """
                UPDATE daily_research_runs
                SET error_code=?
                WHERE status='FAILED'
                  AND coalesce(trim(error_code), '')=''
                """,
                [_FAILED_RUN_UNCLASSIFIED],
            )
            con.execute(
                """
                UPDATE daily_research_items
                SET error_code=?
                WHERE status='FAILED'
                  AND coalesce(trim(error_code), '')=''
                """,
                [_FAILED_ITEM_UNCLASSIFIED],
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS daily_research_publications (
                    run_id VARCHAR PRIMARY KEY,
                    status VARCHAR,
                    attempts INTEGER,
                    last_error VARCHAR,
                    sent_at TIMESTAMPTZ,
                    updated_at TIMESTAMPTZ
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS daily_research_ta_snapshot_links (
                    run_id VARCHAR,
                    symbol VARCHAR,
                    snapshot_id VARCHAR,
                    status VARCHAR,
                    error_code VARCHAR,
                    created_at TIMESTAMPTZ,
                    PRIMARY KEY(run_id, symbol)
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS daily_research_snapshot_links (
                    run_id VARCHAR,
                    symbol VARCHAR,
                    snapshot_id VARCHAR,
                    status VARCHAR,
                    error_code VARCHAR,
                    created_at TIMESTAMPTZ,
                    PRIMARY KEY(run_id, symbol)
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS daily_research_snapshot_comparisons (
                    run_id VARCHAR,
                    symbol VARCHAR,
                    snapshot_id VARCHAR,
                    status VARCHAR,
                    differences_json VARCHAR,
                    classification VARCHAR,
                    checked_at TIMESTAMPTZ,
                    PRIMARY KEY(run_id, symbol)
                )
                """
            )
            con.commit()
        finally:
            con.close()

    def start_run(self, run: DailyResearchRun) -> None:
        values = self._run_values(run)
        con = self._connect()
        try:
            con.execute(
                """
                INSERT INTO daily_research_runs VALUES
                (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(run_id) DO UPDATE SET
                    status=excluded.status,
                    completed_symbols=excluded.completed_symbols,
                    failed_symbols=excluded.failed_symbols,
                    completed_at=excluded.completed_at,
                    error_code=excluded.error_code
                """,
                values,
            )
            con.commit()
        finally:
            con.close()

    def save_item(self, item: DailyResearchItem) -> None:
        error_code = item.error_code.strip()
        if item.status == "FAILED" and not error_code:
            error_code = _FAILED_ITEM_UNCLASSIFIED
        con = self._connect()
        try:
            con.execute(
                """
                INSERT INTO daily_research_items (
                    run_id, trading_date, symbol, rank, screening_score,
                    screening_status, status, recommendation, ai_score,
                    confidence, thesis, risks_json, provider, model,
                    error_code, created_at, completed_at, raw_json,
                    snapshot_id, data_version, model_version, invocation_id,
                    ta_snapshot_id
                ) VALUES
                (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(run_id, symbol) DO UPDATE SET
                    status=excluded.status,
                    recommendation=excluded.recommendation,
                    ai_score=excluded.ai_score,
                    confidence=excluded.confidence,
                    thesis=excluded.thesis,
                    risks_json=excluded.risks_json,
                    provider=excluded.provider,
                    model=excluded.model,
                    error_code=excluded.error_code,
                    completed_at=excluded.completed_at,
                    raw_json=excluded.raw_json,
                    snapshot_id=excluded.snapshot_id,
                    data_version=excluded.data_version,
                    model_version=excluded.model_version,
                    invocation_id=excluded.invocation_id,
                    ta_snapshot_id=excluded.ta_snapshot_id
                """,
                [
                    item.run_id,
                    item.trading_date,
                    item.symbol,
                    item.rank,
                    item.screening_score,
                    item.screening_status,
                    item.status,
                    item.recommendation,
                    item.ai_score,
                    item.confidence,
                    item.thesis,
                    json.dumps(item.risks, ensure_ascii=False),
                    item.provider,
                    item.model,
                    error_code,
                    item.created_at,
                    item.completed_at,
                    json.dumps(_json_value(item.raw), ensure_ascii=False),
                    item.snapshot_id,
                    item.data_version,
                    item.model_version,
                    item.invocation_id,
                    item.ta_snapshot_id,
                ],
            )
            con.commit()
        finally:
            con.close()

    def finish_run(
        self,
        run_id: str,
        *,
        status: str,
        completed: int,
        failed: int,
        error_code: str = "",
        at: datetime | None = None,
    ) -> None:
        normalized_error = error_code.strip()
        if status == "FAILED" and not normalized_error:
            normalized_error = _FAILED_RUN_UNCLASSIFIED
        con = self._connect()
        try:
            con.execute(
                """
                UPDATE daily_research_runs
                SET status=?, completed_symbols=?, failed_symbols=?,
                    completed_at=?, error_code=?
                WHERE run_id=?
                """,
                [
                    status,
                    completed,
                    failed,
                    _utc(at or datetime.now(_UTC)),
                    normalized_error,
                    run_id,
                ],
            )
            con.commit()
        finally:
            con.close()

    def update_progress(self, run_id: str, *, completed: int, failed: int) -> None:
        con = self._connect()
        try:
            con.execute(
                """
                UPDATE daily_research_runs
                SET completed_symbols=?, failed_symbols=?
                WHERE run_id=?
                """,
                [completed, failed, run_id],
            )
            con.commit()
        finally:
            con.close()

    def save_snapshot_link(
        self,
        *,
        run_id: str,
        symbol: str,
        snapshot_id: str,
        status: str,
        error_code: str = "",
        created_at: datetime | None = None,
    ) -> None:
        normalized_error = error_code.strip()
        if status == "FAILED" and not normalized_error:
            normalized_error = _FAILED_SNAPSHOT_UNCLASSIFIED
        con = self._connect()
        try:
            con.execute(
                """
                INSERT INTO daily_research_snapshot_links VALUES (?,?,?,?,?,?)
                ON CONFLICT(run_id, symbol) DO UPDATE SET
                    snapshot_id=excluded.snapshot_id,
                    status=excluded.status,
                    error_code=excluded.error_code,
                    created_at=excluded.created_at
                """,
                [
                    run_id,
                    symbol,
                    snapshot_id,
                    status,
                    normalized_error,
                    _utc(created_at or datetime.now(_UTC)),
                ],
            )
            con.commit()
        finally:
            con.close()

    def recover_stale_runs(
        self,
        *,
        now: datetime,
        stale_after_seconds: int,
    ) -> list[str]:
        """Fail closed any interrupted run older than the worker timeout."""
        now = _utc(now)
        if stale_after_seconds <= 0:
            raise ValueError("STALE_RUN_TIMEOUT_INVALID")
        cutoff = now - timedelta(seconds=stale_after_seconds)
        con = self._connect()
        try:
            con.execute("BEGIN TRANSACTION")
            rows = con.execute(
                """
                SELECT run_id, total_symbols
                FROM daily_research_runs
                WHERE status='RUNNING' AND started_at<=?
                ORDER BY started_at, run_id
                """,
                [cutoff],
            ).fetchall()
            recovered = []
            for run_id, total_symbols in rows:
                con.execute(
                    """
                    UPDATE daily_research_items
                    SET status='FAILED',
                        error_code=?,
                        completed_at=?
                    WHERE run_id=? AND status IN ('PENDING','RUNNING')
                    """,
                    [_INTERRUPTED_RUN_ERROR, now, run_id],
                )
                completed, failed = con.execute(
                    """
                    SELECT
                        count(*) FILTER (WHERE status='COMPLETED'),
                        count(*) FILTER (WHERE status='FAILED')
                    FROM daily_research_items
                    WHERE run_id=?
                    """,
                    [run_id],
                ).fetchone()
                incomplete = max(
                    0,
                    int(total_symbols or 0) - int(completed or 0),
                )
                failed = max(int(failed or 0), incomplete)
                con.execute(
                    """
                    UPDATE daily_research_runs
                    SET status='FAILED',
                        completed_symbols=?,
                        failed_symbols=?,
                        completed_at=?,
                        error_code=?
                    WHERE run_id=?
                    """,
                    [
                        int(completed or 0),
                        failed,
                        now,
                        _INTERRUPTED_RUN_ERROR,
                        run_id,
                    ],
                )
                recovered.append(str(run_id))
            con.commit()
            return recovered
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def snapshot_link(
        self,
        run_id: str,
        symbol: str,
    ) -> tuple[str, str] | None:
        con = self._connect(read_only=True)
        try:
            row = con.execute(
                "SELECT snapshot_id, status "
                "FROM daily_research_snapshot_links "
                "WHERE run_id=? AND symbol=?",
                [run_id, symbol.strip().upper()],
            ).fetchone()
        finally:
            con.close()
        return (str(row[0]), str(row[1])) if row else None

    def save_ta_snapshot_link(
        self,
        *,
        run_id: str,
        symbol: str,
        snapshot_id: str,
        status: str,
        error_code: str = "",
        created_at: datetime,
    ) -> None:
        normalized_error = error_code.strip()
        if status == "FAILED" and not normalized_error:
            normalized_error = _FAILED_SNAPSHOT_UNCLASSIFIED
        con = self._connect()
        try:
            con.execute(
                """
                INSERT INTO daily_research_ta_snapshot_links
                VALUES (?,?,?,?,?,?)
                ON CONFLICT(run_id, symbol) DO UPDATE SET
                    snapshot_id=excluded.snapshot_id,
                    status=excluded.status,
                    error_code=excluded.error_code,
                    created_at=excluded.created_at
                """,
                [
                    run_id,
                    symbol.strip().upper(),
                    snapshot_id,
                    status,
                    normalized_error,
                    _utc(created_at),
                ],
            )
            con.commit()
        finally:
            con.close()

    def snapshot_links(self, run_id: str) -> list[dict[str, Any]]:
        con = self._connect(read_only=True)
        try:
            cursor = con.execute(
                """
                SELECT * FROM daily_research_snapshot_links
                WHERE run_id=? ORDER BY symbol
                """,
                [run_id],
            )
            columns = [item[0] for item in cursor.description]
            rows = cursor.fetchall()
        finally:
            con.close()
        return [dict(zip(columns, row)) for row in rows]

    def save_snapshot_comparison(
        self,
        *,
        run_id: str,
        symbol: str,
        snapshot_id: str,
        status: str,
        differences: list[dict[str, Any]],
        classification: str,
        checked_at: datetime,
    ) -> None:
        con = self._connect()
        try:
            con.execute(
                """
                INSERT INTO daily_research_snapshot_comparisons
                VALUES (?,?,?,?,?,?,?)
                ON CONFLICT(run_id, symbol) DO UPDATE SET
                    snapshot_id=excluded.snapshot_id,
                    status=excluded.status,
                    differences_json=excluded.differences_json,
                    classification=excluded.classification,
                    checked_at=excluded.checked_at
                """,
                [
                    run_id,
                    symbol,
                    snapshot_id,
                    status,
                    json.dumps(
                        differences,
                        ensure_ascii=False,
                        sort_keys=True,
                        default=str,
                    ),
                    classification,
                    _utc(checked_at),
                ],
            )
            con.commit()
        finally:
            con.close()

    def snapshot_comparisons(
        self,
        run_id: str,
    ) -> list[dict[str, Any]]:
        con = self._connect(read_only=True)
        try:
            cursor = con.execute(
                """
                SELECT * FROM daily_research_snapshot_comparisons
                WHERE run_id=? ORDER BY symbol
                """,
                [run_id],
            )
            columns = [item[0] for item in cursor.description]
            rows = cursor.fetchall()
        finally:
            con.close()
        result = []
        for row in rows:
            value = dict(zip(columns, row))
            value["differences"] = json.loads(
                value.pop("differences_json") or "[]"
            )
            result.append(value)
        return result

    def latest_run(
        self,
        trading_date: str | None = None,
        *,
        successful_only: bool = False,
    ) -> DailyResearchRun | None:
        where: list[str] = []
        params: list[Any] = []
        if trading_date:
            where.append("trading_date=?")
            params.append(trading_date)
        if successful_only:
            where.append("status IN ('COMPLETED','COMPLETED_WITH_ERRORS')")
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        con = self._connect(read_only=True)
        try:
            row = con.execute(
                f"""
                SELECT * FROM daily_research_runs {clause}
                ORDER BY started_at DESC LIMIT 1
                """,
                params,
            ).fetchone()
        finally:
            con.close()
        return self._run_from_row(row) if row else None

    def items(self, run_id: str) -> list[DailyResearchItem]:
        con = self._connect(read_only=True)
        try:
            rows = con.execute(
                """
                SELECT * FROM daily_research_items
                WHERE run_id=? ORDER BY rank, symbol
                """,
                [run_id],
            ).fetchall()
        finally:
            con.close()
        return [self._item_from_row(row) for row in rows]

    def score_snapshots(
        self,
        now: datetime,
        *,
        max_age_hours: float = 36.0,
    ) -> dict[str, AIScoreSnapshot]:
        now = _utc(now)
        trading_date = now.astimezone(_ET).date().isoformat()
        con = self._connect(read_only=True)
        try:
            successful = con.execute(
                """
                SELECT run_id FROM daily_research_runs
                WHERE trading_date=? AND status IN (?,?)
                ORDER BY completed_at DESC
                """,
                [
                    trading_date,
                    "COMPLETED",
                    "COMPLETED_WITH_ERRORS",
                ],
            ).fetchall()
        finally:
            con.close()
        if len(successful) != 1:
            return {}
        run = self.latest_run(trading_date, successful_only=True)
        if run is None or run.completed_at is None:
            return {}
        if now - _utc(run.completed_at) > timedelta(hours=max_age_hours):
            return {}
        snapshots: dict[str, AIScoreSnapshot] = {}
        snapshot_store = ResearchSnapshotStore(self.db_path)
        for item in self.items(run.run_id):
            if (
                item.status != "COMPLETED"
                or item.ai_score is None
                or item.completed_at is None
            ):
                continue
            if not self._trusted_item_contract(
                run,
                item,
                snapshot_store,
            ):
                continue
            snapshots[item.symbol] = AIScoreSnapshot(
                symbol=item.symbol,
                score=item.ai_score,
                created_at=_utc(item.completed_at),
                run_id=run.run_id,
                provider=item.provider or run.provider,
                model=item.model or run.model,
                source="daily_research",
                generated_by="TradingAgentsAdapter",
                is_stub=False,
                contributors=[
                    {
                        "agent_name": "tradingagents_graph",
                        "score": item.ai_score,
                        "created_at": item.completed_at,
                        "provider": item.provider or run.provider,
                        "model": item.model or run.model,
                        "is_stub": False,
                        "is_fallback": False,
                    }
                ],
                contributor_count=1,
                weight_coverage=1.0,
                has_llm=True,
                fallback_count=0,
                recommendation=item.recommendation,
            )
        return snapshots

    @staticmethod
    def _trusted_item_contract(
        run: DailyResearchRun,
        item: DailyResearchItem,
        snapshot_store: ResearchSnapshotStore,
    ) -> bool:
        required = (
            item.snapshot_id,
            item.ta_snapshot_id,
            item.data_version,
            item.model_version,
            item.invocation_id,
        )
        if not all(str(value).strip() for value in required):
            return False
        if item.model_version != run.config_version:
            return False
        if (
            item.run_id != run.run_id
            or item.trading_date != run.trading_date
        ):
            return False
        try:
            input_snapshot = snapshot_store.replay(item.snapshot_id)
            output_snapshot = snapshot_store.replay(item.ta_snapshot_id)
        except (KeyError, ValueError):
            return False
        for snapshot in (input_snapshot, output_snapshot):
            if (
                snapshot.run_id != run.run_id
                or snapshot.symbol != item.symbol
                or snapshot.trading_date != run.trading_date
                or snapshot.data_cutoff != run.data_cutoff
            ):
                return False
        expected_data_version = (
            f"{input_snapshot.payload_version}:"
            f"{snapshot_content_hash(input_snapshot)[:20]}"
        )
        if item.data_version != expected_data_version:
            return False
        invocation = dict(output_snapshot.payload or {}).get("invocation")
        if not isinstance(invocation, dict):
            return False
        if (
            invocation.get("invocation_id") != item.invocation_id
            or invocation.get("run_id") != run.run_id
            or invocation.get("snapshot_id") != item.snapshot_id
            or invocation.get("data_version") != item.data_version
            or invocation.get("model_version") != item.model_version
        ):
            return False
        if (
            output_snapshot.quality != ResearchQuality.GOOD
            or output_snapshot.quality_score < 1.0
            or any(
                entry.status != ResearchSourceStatus.OK
                for entry in output_snapshot.source_manifest
            )
        ):
            return False
        return True

    def begin_publication(self, run_id: str) -> bool:
        now = datetime.now(_UTC)
        con = self._connect()
        try:
            row = con.execute(
                "SELECT status FROM daily_research_publications WHERE run_id=?",
                [run_id],
            ).fetchone()
            if row and row[0] == "SENT":
                return False
            if row:
                con.execute(
                    """
                    UPDATE daily_research_publications
                    SET status='PENDING', attempts=attempts+1,
                        last_error=NULL, updated_at=?
                    WHERE run_id=?
                    """,
                    [now, run_id],
                )
            else:
                con.execute(
                    "INSERT INTO daily_research_publications VALUES (?,?,?,?,?,?)",
                    [run_id, "PENDING", 1, None, None, now],
                )
            con.commit()
            return True
        finally:
            con.close()

    def finish_publication(self, run_id: str, ok: bool, error: str = "") -> None:
        now = datetime.now(_UTC)
        con = self._connect()
        try:
            con.execute(
                """
                UPDATE daily_research_publications
                SET status=?, last_error=?, sent_at=?, updated_at=?
                WHERE run_id=?
                """,
                [
                    "SENT" if ok else "FAILED",
                    error or None,
                    now if ok else None,
                    now,
                    run_id,
                ],
            )
            con.commit()
        finally:
            con.close()

    @staticmethod
    def _run_values(run: DailyResearchRun) -> list[Any]:
        return [
            run.run_id,
            run.trading_date,
            run.status,
            run.universe_version,
            run.data_cutoff,
            run.timeframe,
            run.screen_limit,
            run.deep_limit,
            run.provider,
            run.model,
            run.total_symbols,
            run.completed_symbols,
            run.failed_symbols,
            run.started_at,
            run.completed_at,
            run.error_code,
            run.config_version,
        ]

    @staticmethod
    def _run_from_row(row: tuple[Any, ...]) -> DailyResearchRun:
        return DailyResearchRun(
            run_id=row[0],
            trading_date=row[1],
            status=row[2],
            universe_version=row[3],
            data_cutoff=row[4],
            timeframe=row[5],
            screen_limit=row[6],
            deep_limit=row[7],
            provider=row[8],
            model=row[9],
            total_symbols=row[10],
            completed_symbols=row[11],
            failed_symbols=row[12],
            started_at=row[13],
            completed_at=row[14],
            error_code=row[15] or "",
            config_version=row[16] or "",
        )

    @staticmethod
    def _item_from_row(row: tuple[Any, ...]) -> DailyResearchItem:
        return DailyResearchItem(
            run_id=row[0],
            trading_date=row[1],
            symbol=row[2],
            rank=row[3],
            screening_score=row[4],
            screening_status=row[5],
            status=row[6],
            recommendation=row[7],
            ai_score=row[8],
            confidence=row[9],
            thesis=row[10] or "",
            risks=tuple(json.loads(row[11] or "[]")),
            provider=row[12] or "",
            model=row[13] or "",
            error_code=row[14] or "",
            created_at=row[15],
            completed_at=row[16],
            raw=json.loads(row[17] or "{}"),
            snapshot_id=(row[18] or "") if len(row) > 18 else "",
            data_version=(row[19] or "") if len(row) > 19 else "",
            model_version=(row[20] or "") if len(row) > 20 else "",
            invocation_id=(row[21] or "") if len(row) > 21 else "",
            ta_snapshot_id=(row[22] or "") if len(row) > 22 else "",
        )


class DailyResearchService:
    def __init__(
        self,
        store: DailyResearchStore,
        analyzer: Any,
        *,
        notifier: Any | None = None,
        snapshot_store: ResearchSnapshotStore | None = None,
        stale_run_seconds: int | None = None,
    ) -> None:
        self.store = store
        self.analyzer = analyzer
        self.notifier = notifier
        self.snapshot_store = snapshot_store or ResearchSnapshotStore(
            store.db_path
        )
        configured_stale = os.getenv(
            "TRADINGAGENTS_TIMEOUT_SECONDS",
            "7200",
        )
        self.stale_run_seconds = int(
            stale_run_seconds
            if stale_run_seconds is not None
            else configured_stale
        )
        if self.stale_run_seconds <= 0:
            raise ValueError("STALE_RUN_TIMEOUT_INVALID")

    def run(
        self,
        universe: Iterable[str],
        *,
        trading_date: str,
        timeframe: str = "5m",
        screen_limit: int = 10,
        deep_limit: int = 5,
        strategy_statistics_path: str = "",
        market_regime: str = "",
        force: bool = False,
        now: datetime | None = None,
    ) -> DailyResearchRun:
        now = _utc(now or datetime.now(_UTC))
        self.store.recover_stale_runs(
            now=now,
            stale_after_seconds=self.stale_run_seconds,
        )
        existing = self.store.latest_run(trading_date, successful_only=True)
        if existing is not None and not force:
            return existing
        symbols = tuple(
            dict.fromkeys(
                str(symbol).strip().upper()
                for symbol in universe
                if str(symbol).strip()
            )
        )
        if not symbols:
            raise ValueError("DAILY_RESEARCH_UNIVERSE_EMPTY")
        screening_inputs: dict[str, Any] = {}
        candidates = build_daily_candidates(
            symbols,
            timeframe=timeframe,
            strategy_statistics_path=strategy_statistics_path,
            market_regime=market_regime,
            limit=screen_limit,
            now=now,
            input_capture=screening_inputs,
        )
        deep = self._deep_candidates(candidates, deep_limit)
        universe_version = hashlib.sha256(
            json.dumps(symbols, separators=(",", ":")).encode()
        ).hexdigest()[:20]
        analyzer_config = (
            self.analyzer.describe()
            if hasattr(self.analyzer, "describe")
            else {"provider": type(self.analyzer).__name__}
        )
        analyzer_config = {
            **analyzer_config,
            "strategy_statistics_path": strategy_statistics_path,
            "market_regime": market_regime,
        }
        config_version = hashlib.sha256(
            json.dumps(
                {
                    "analyzer": analyzer_config,
                    "timeframe": timeframe,
                    "screen_limit": screen_limit,
                    "deep_limit": deep_limit,
                },
                sort_keys=True,
                default=str,
            ).encode()
        ).hexdigest()[:20]
        run_id = (
            "research-"
            + hashlib.sha256(
                f"{trading_date}|{now.isoformat()}|{config_version}".encode()
            ).hexdigest()[:24]
        )
        provider = str(getattr(self.analyzer, "provider", type(self.analyzer).__name__))
        model = str(getattr(self.analyzer, "model", ""))
        run = DailyResearchRun(
            run_id=run_id,
            trading_date=trading_date,
            status="RUNNING",
            universe_version=universe_version,
            data_cutoff=now,
            timeframe=timeframe,
            screen_limit=screen_limit,
            deep_limit=deep_limit,
            provider=provider,
            model=model,
            total_symbols=len(deep),
            completed_symbols=0,
            failed_symbols=0,
            started_at=now,
            config_version=config_version,
        )
        self.store.start_run(run)
        self._write_shadow_snapshots(
            run,
            candidates,
            screening_inputs=screening_inputs,
            strategy_statistics_path=strategy_statistics_path,
        )
        deep_symbols = {candidate.symbol for candidate in deep}
        for candidate in candidates:
            self.store.save_item(
                self._screened_item(run, candidate, candidate.symbol in deep_symbols)
            )

        completed = 0
        failed = 0
        first_error = "" if deep else "NO_ELIGIBLE_DEEP_CANDIDATES"
        for candidate in deep:
            created_at = now
            self.store.save_item(
                self._screened_item(
                    run,
                    candidate,
                    selected=True,
                    status="RUNNING",
                )
            )
            try:
                invocation = self._invocation(run, candidate.symbol, now)
                if hasattr(self.analyzer, "analyze_with_context"):
                    analysis = self.analyzer.analyze_with_context(
                        candidate.symbol,
                        trading_date,
                        invocation,
                    )
                else:
                    analysis = self.analyzer.analyze(
                        candidate.symbol,
                        trading_date,
                    )
                ta_snapshot_id = self._write_ta_snapshot(
                    run,
                    candidate.symbol,
                    analysis,
                    invocation,
                )
                item = DailyResearchItem(
                    run_id=run_id,
                    trading_date=trading_date,
                    symbol=candidate.symbol,
                    rank=candidate.rank,
                    screening_score=candidate.score,
                    screening_status=candidate.status,
                    status="COMPLETED",
                    recommendation=analysis.recommendation,
                    ai_score=max(0.0, min(100.0, float(analysis.score))),
                    confidence=max(0.0, min(1.0, float(analysis.confidence))),
                    thesis=analysis.thesis,
                    risks=self._research_risks(candidate, analysis.risks),
                    provider=analysis.provider,
                    model=analysis.model,
                    created_at=created_at,
                    completed_at=now,
                    raw=analysis.raw,
                    snapshot_id=invocation.snapshot_id,
                    data_version=invocation.data_version,
                    model_version=invocation.model_version,
                    invocation_id=invocation.invocation_id,
                    ta_snapshot_id=ta_snapshot_id,
                )
                completed += 1
            except Exception as exc:
                error_code = _stable_error_code(exc)
                first_error = first_error or error_code
                failed += 1
                item = DailyResearchItem(
                    run_id=run_id,
                    trading_date=trading_date,
                    symbol=candidate.symbol,
                    rank=candidate.rank,
                    screening_score=candidate.score,
                    screening_status=candidate.status,
                    status="FAILED",
                    error_code=error_code,
                    provider=provider,
                    model=model,
                    created_at=created_at,
                    completed_at=now,
                )
                logger.warning(
                    "Daily research %s failed: %s", candidate.symbol, error_code
                )
            self.store.save_item(item)
            self.store.update_progress(run_id, completed=completed, failed=failed)

        if completed and failed:
            status = "COMPLETED_WITH_ERRORS"
        elif completed:
            status = "COMPLETED"
        else:
            status = "FAILED"
        self.store.finish_run(
            run_id,
            status=status,
            completed=completed,
            failed=failed,
            at=now,
            error_code=first_error if status == "FAILED" else "",
        )
        final = self.store.latest_run(trading_date)
        if final is None:
            raise RuntimeError("DAILY_RESEARCH_RUN_NOT_PERSISTED")
        if final.status in _SUCCESS_STATES:
            self._publish(final)
        return final

    def _write_ta_snapshot(
        self,
        run: DailyResearchRun,
        symbol: str,
        analysis: ResearchAnalysis,
        invocation: TradingAgentsInvocation,
    ) -> str:
        if not analysis.source_manifest:
            if hasattr(self.analyzer, "analyze_with_context"):
                raise RuntimeError("TRADINGAGENTS_SOURCE_MANIFEST_MISSING")
            return ""
        manifest = tuple(
            ResearchSourceManifestEntry(
                source=str(item["source"]),
                status=ResearchSourceStatus(str(item["status"])),
                as_of=datetime.fromisoformat(str(item["as_of"])),
                fetched_at=datetime.fromisoformat(str(item["fetched_at"])),
                quality_score=float(item["quality_score"]),
                coverage=tuple(item["coverage"]),
                payload_version=str(item["payload_version"]),
                failure_code=str(item["failure_code"]),
                metadata=dict(item["metadata"]),
            )
            for item in analysis.source_manifest
        )
        failed = sum(
            entry.status in {
                ResearchSourceStatus.FAILED,
                ResearchSourceStatus.MISSING,
            }
            for entry in manifest
        )
        quality = (
            ResearchQuality.GOOD
            if failed == 0
            else (
                ResearchQuality.FAILED
                if failed == len(manifest)
                else ResearchQuality.PARTIAL
            )
        )
        quality_score = sum(
            entry.quality_score for entry in manifest
        ) / len(manifest)
        completed_raw = dict(analysis.raw or {}).get(
            "worker_completed_at"
        )
        captured_at = (
            datetime.fromisoformat(str(completed_raw))
            if completed_raw
            else invocation.requested_at
        )
        captured_at = max(_utc(captured_at), _utc(run.data_cutoff))
        payload = {
            "invocation": invocation.to_dict(),
            "decision": dict(analysis.raw or {}).get("decision"),
            "state": dict(analysis.raw or {}).get("state"),
            "source_manifest": [dict(item) for item in analysis.source_manifest],
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        snapshot = ResearchSnapshot(
            snapshot_id=(
                "ta-snapshot-"
                + hashlib.sha256(encoded.encode()).hexdigest()[:24]
            ),
            symbol=symbol,
            trading_date=run.trading_date,
            as_of=run.data_cutoff,
            data_cutoff=run.data_cutoff,
            captured_at=captured_at,
            source_manifest=manifest,
            quality=quality,
            quality_score=quality_score,
            payload_version="tradingagents-output:v1",
            payload=payload,
            run_id=run.run_id,
            created_at=captured_at,
        )
        result = self.snapshot_store.save_or_get(snapshot)
        self.store.save_ta_snapshot_link(
            run_id=run.run_id,
            symbol=symbol,
            snapshot_id=result.snapshot_id,
            status="WRITTEN" if result.created else "EXISTS",
            created_at=captured_at,
        )
        self.snapshot_store.replay(result.snapshot_id)
        return result.snapshot_id

    def _invocation(
        self,
        run: DailyResearchRun,
        symbol: str,
        requested_at: datetime,
    ) -> TradingAgentsInvocation:
        link = self.store.snapshot_link(run.run_id, symbol)
        if link is None or not link[0] or link[1] not in {"WRITTEN", "EXISTS"}:
            raise RuntimeError("TRADINGAGENTS_SNAPSHOT_LINK_UNAVAILABLE")
        snapshot = self.snapshot_store.replay(link[0])
        if snapshot.run_id and snapshot.run_id != run.run_id:
            raise RuntimeError("TRADINGAGENTS_SNAPSHOT_RUN_MISMATCH")
        if snapshot.symbol != symbol.strip().upper():
            raise RuntimeError("TRADINGAGENTS_SNAPSHOT_SYMBOL_MISMATCH")
        if snapshot.trading_date != run.trading_date:
            raise RuntimeError("TRADINGAGENTS_SNAPSHOT_DATE_MISMATCH")
        content_hash = snapshot_content_hash(snapshot)
        data_version = (
            f"{snapshot.payload_version}:{content_hash[:20]}"
        )
        raw = "|".join(
            (
                run.run_id,
                snapshot.snapshot_id,
                symbol.strip().upper(),
                data_version,
                run.config_version,
            )
        )
        return TradingAgentsInvocation(
            invocation_id=(
                "ta-invocation-"
                + hashlib.sha256(raw.encode()).hexdigest()[:24]
            ),
            run_id=run.run_id,
            snapshot_id=snapshot.snapshot_id,
            snapshot_content_hash=content_hash,
            data_version=data_version,
            model_version=run.config_version,
            symbol=symbol.strip().upper(),
            trading_date=run.trading_date,
            data_cutoff=run.data_cutoff,
            requested_at=requested_at,
        )

    def _write_shadow_snapshots(
        self,
        run: DailyResearchRun,
        candidates: list[DailyCandidate],
        *,
        screening_inputs: dict[str, Any],
        strategy_statistics_path: str,
    ) -> None:
        bars_by_symbol = dict(screening_inputs.get("bars") or {})
        statistics = tuple(
            screening_inputs.get("strategy_statistics") or ()
        )
        for candidate in candidates:
            try:
                snapshot = build_screening_shadow_snapshot(
                    run_id=run.run_id,
                    trading_date=run.trading_date,
                    timeframe=run.timeframe,
                    candidate=candidate,
                    bars=bars_by_symbol.get(candidate.symbol),
                    strategy_statistics=statistics,
                    strategy_statistics_path=strategy_statistics_path,
                    captured_at=run.data_cutoff,
                )
                if hasattr(self.snapshot_store, "save_or_get"):
                    result = self.snapshot_store.save_or_get(snapshot)
                    snapshot_id = result.snapshot_id
                    created = result.created
                    self.snapshot_store.bind_to_run(
                        run_id=run.run_id,
                        symbol=candidate.symbol,
                        trading_date=run.trading_date,
                        snapshot_id=snapshot_id,
                        bound_at=run.data_cutoff,
                    )
                else:
                    created = self.snapshot_store.save(snapshot)
                    snapshot_id = snapshot.snapshot_id
                replayed = (
                    self.snapshot_store.replay(snapshot_id)
                    if hasattr(self.snapshot_store, "replay")
                    else snapshot
                )
                differences = compare_candidate_to_snapshot(
                    candidate,
                    replayed,
                )
                comparison_status = (
                    "MATCH" if not differences else "MISMATCH"
                )
                classification = (
                    "MATCH"
                    if not differences
                    else (
                        "CLASSIFIED"
                        if all(
                            item.get("classification")
                            not in {"", "UNCLASSIFIED", None}
                            for item in differences
                        )
                        else "UNCLASSIFIED"
                    )
                )
                self.store.save_snapshot_link(
                    run_id=run.run_id,
                    symbol=candidate.symbol,
                    snapshot_id=snapshot_id,
                    status="WRITTEN" if created else "EXISTS",
                    created_at=run.data_cutoff,
                )
                self.store.save_snapshot_comparison(
                    run_id=run.run_id,
                    symbol=candidate.symbol,
                    snapshot_id=snapshot_id,
                    status=comparison_status,
                    differences=differences,
                    classification=classification,
                    checked_at=run.data_cutoff,
                )
            except Exception as exc:
                error_code = _stable_error_code(exc)
                self.store.save_snapshot_link(
                    run_id=run.run_id,
                    symbol=candidate.symbol,
                    snapshot_id="",
                    status="FAILED",
                    error_code=error_code,
                    created_at=run.data_cutoff,
                )
                self.store.save_snapshot_comparison(
                    run_id=run.run_id,
                    symbol=candidate.symbol,
                    snapshot_id="",
                    status="NOT_AVAILABLE",
                    differences=[],
                    classification="SNAPSHOT_WRITE_FAILED",
                    checked_at=run.data_cutoff,
                )
                logger.warning(
                    "ResearchSnapshot shadow write %s failed: %s",
                    candidate.symbol,
                    error_code,
                )

    @staticmethod
    def _deep_candidates(
        candidates: list[DailyCandidate], deep_limit: int
    ) -> list[DailyCandidate]:
        eligible = [
            candidate
            for candidate in candidates
            if candidate.status not in {"AVOID_NOW", "MARKET_ANCHOR"}
            and candidate.data_confidence != "低"
        ]
        fallback = [
            candidate
            for candidate in candidates
            if candidate.status not in {"AVOID_NOW", "MARKET_ANCHOR"}
            and candidate.data_confidence == "低"
        ]
        return (eligible + fallback)[: max(1, int(deep_limit))]

    @staticmethod
    def _research_risks(
        candidate: DailyCandidate,
        extra_risks: Iterable[str] = (),
    ) -> tuple[str, ...]:
        risks = list(candidate.risk_flags)
        if candidate.data_confidence == "低":
            risks.append(_LOW_CONFIDENCE_RESEARCH_RISK)
        risks.extend(str(risk) for risk in extra_risks if str(risk).strip())
        return tuple(dict.fromkeys(risks))

    @staticmethod
    def _screened_item(
        run: DailyResearchRun,
        candidate: DailyCandidate,
        selected: bool,
        *,
        status: str = "",
    ) -> DailyResearchItem:
        risks = tuple(candidate.risk_flags)
        if selected:
            risks = DailyResearchService._research_risks(candidate)
        return DailyResearchItem(
            run_id=run.run_id,
            trading_date=run.trading_date,
            symbol=candidate.symbol,
            rank=candidate.rank,
            screening_score=candidate.score,
            screening_status=candidate.status,
            status=status or ("PENDING" if selected else "SCREENED"),
            thesis="；".join(candidate.reasons),
            risks=risks,
            provider=run.provider,
            model=run.model,
            created_at=run.started_at,
        )

    def _publish(self, run: DailyResearchRun) -> None:
        if self.notifier is None or not self.store.begin_publication(run.run_id):
            return
        try:
            from .daily_discord import build_daily_research_message

            note = build_daily_research_message(run, self.store.items(run.run_id))
            ok = bool(self.notifier.send(note))
            self.store.finish_publication(
                run.run_id, ok, "" if ok else "DELIVERY_FAILED"
            )
        except Exception as exc:
            self.store.finish_publication(run.run_id, False, type(exc).__name__)


class DailyResearchWorker:
    """Single-flight scheduler for one post-close or pre-market batch."""

    def __init__(
        self,
        service: DailyResearchService,
        universe: Iterable[str],
        *,
        timeframe: str,
        screen_limit: int,
        deep_limit: int,
        strategy_statistics_path: str = "",
        market_regime: str = "",
        close_hour_et: int = 16,
        close_minute_et: int = 15,
        timeout_seconds: int = 7200,
    ) -> None:
        self.service = service
        self.universe = tuple(universe)
        self.timeframe = timeframe
        self.screen_limit = screen_limit
        self.deep_limit = deep_limit
        self.strategy_statistics_path = strategy_statistics_path
        self.market_regime = market_regime
        self.close_hour_et = close_hour_et
        self.close_minute_et = close_minute_et
        self.timeout_seconds = timeout_seconds
        self._pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="daily-research"
        )
        self._future: Future | None = None
        self._started = 0.0

    def start_if_due(self, now: datetime, *, force: bool = False) -> bool:
        now = _utc(now)
        if self._future is not None and not self._future.done():
            return False
        recover = getattr(
            self.service.store,
            "recover_stale_runs",
            None,
        )
        if callable(recover):
            recover(
                now=now,
                stale_after_seconds=int(
                    getattr(
                        self.service,
                        "stale_run_seconds",
                        self.timeout_seconds,
                    )
                ),
            )
        if not force and not in_daily_run_window(
            now,
            close_hour_et=self.close_hour_et,
            close_minute_et=self.close_minute_et,
        ):
            return False
        trading_date = research_target_date(
            now,
            close_hour_et=self.close_hour_et,
            close_minute_et=self.close_minute_et,
        )
        if not force and self.service.store.latest_run(trading_date) is not None:
            return False
        self._started = time.monotonic()
        self._future = self._pool.submit(
            self.service.run,
            self.universe,
            trading_date=trading_date,
            timeframe=self.timeframe,
            screen_limit=self.screen_limit,
            deep_limit=self.deep_limit,
            strategy_statistics_path=self.strategy_statistics_path,
            market_regime=self.market_regime,
            force=force,
            now=now,
        )
        return True

    def poll(self) -> DailyResearchRun | None:
        if self._future is None:
            return None
        if self._future.done():
            future, self._future = self._future, None
            return future.result()
        if time.monotonic() - self._started > self.timeout_seconds:
            future, self._future = self._future, None
            future.cancel()
            old_pool = self._pool
            self._pool = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="daily-research"
            )
            old_pool.shutdown(wait=False, cancel_futures=True)
            raise TimeoutError("DAILY_RESEARCH_TIMEOUT")
        return None

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


def build_default_service(db_path: str) -> DailyResearchService:
    return DailyResearchService(
        DailyResearchStore(db_path),
        TradingAgentsAdapter(),
        notifier=None,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one immutable daily TradingAgents research batch"
    )
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols")
    parser.add_argument("--trading-date", default="")
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--screen-limit", type=int, default=10)
    parser.add_argument("--deep-limit", type=int, default=5)
    parser.add_argument("--db-path", default="ai_states.duckdb")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--strategy-statistics-path", default="")
    parser.add_argument("--market-regime", default="")
    return parser


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)
    args = _parser().parse_args()
    now = datetime.now(_UTC)
    symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
    service = build_default_service(args.db_path)
    result = service.run(
        symbols,
        trading_date=args.trading_date or research_target_date(now),
        timeframe=args.timeframe,
        screen_limit=args.screen_limit,
        deep_limit=args.deep_limit,
        strategy_statistics_path=args.strategy_statistics_path,
        market_regime=args.market_regime,
        force=args.force,
        now=now,
    )
    print(json.dumps(asdict(result), ensure_ascii=False, default=str, indent=2))


if __name__ == "__main__":
    main()
