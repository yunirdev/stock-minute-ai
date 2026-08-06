"""Standalone TradingAgents subprocess worker.

This module intentionally depends only on the standard library and the
third-party ``tradingagents`` package. It is executed with the dedicated
TradingAgents Python interpreter, keeping LangChain/LangGraph dependencies out
of the production Runtime environment.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_RESULT_PREFIX = "__TRADINGAGENTS_RESULT__="
_UTC = timezone.utc


def _json_value(value: Any, depth: int = 0) -> Any:
    if depth > 8:
        return "<max-depth>"
    if isinstance(value, str):
        return value[:20_000]
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=_UTC)
        return value.astimezone(_UTC).isoformat()
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


def _emit(payload: dict[str, Any]) -> None:
    serialized = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    print(f"{_RESULT_PREFIX}{serialized}", flush=True)


def _exception_diagnostic(exc: Exception) -> str:
    """Classify worker failures without returning free-form sensitive text."""
    chain = []
    current: BaseException | None = exc
    while current is not None and len(chain) < 6:
        chain.append(current)
        current = current.__cause__ or current.__context__
    names = {type(item).__name__ for item in chain}
    text = " ".join(str(item) for item in chain).lower()
    if "application control" in text or "dll load failed" in text:
        return "DEPENDENCY_LOAD_BLOCKED"
    if "unable to open database file" in text:
        return "CACHE_DATABASE_UNAVAILABLE"
    if (
        "model" in text
        and "not found" in text
        and ("404" in text or "not_found_error" in text)
    ):
        return "MODEL_NOT_FOUND"
    if any("timeout" in name.lower() for name in names):
        return "LLM_API_TIMEOUT"
    if any(
        token in name.lower()
        for name in names
        for token in ("connection", "connecterror")
    ):
        return "LLM_API_CONNECTION_UNAVAILABLE"
    if "http 5" in text or "status code: 5" in text:
        return "LLM_API_HTTP_5XX"
    return "WORKER_FAILURE"


def _configured_graph(payload: dict[str, Any]):
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    config = dict(DEFAULT_CONFIG)
    overrides = {
        "llm_provider": payload.get("llm_provider"),
        "deep_think_llm": payload.get("deep_model"),
        "quick_think_llm": payload.get("quick_model"),
        "backend_url": payload.get("backend_url"),
        "data_cache_dir": payload.get("cache_dir"),
        "results_dir": payload.get("results_dir"),
        "memory_log_path": payload.get("memory_log_path"),
    }
    for key, value in overrides.items():
        if value not in (None, ""):
            config[key] = value
    _apply_complexity_overrides(config, payload.get("complexity_overrides"))
    _prepare_runtime_paths(config)
    return (
        TradingAgentsGraph(
            debug=bool(payload.get("debug", False)),
            config=config,
        ),
        config,
    )


def _apply_complexity_overrides(
    config: dict[str, Any], overrides: dict[str, Any] | None
) -> None:
    """Mirrors daily_research._apply_complexity() on this side of the subprocess
    boundary — the parent process can't reach into our config dict directly."""
    if not overrides:
        return
    for key, value in overrides.items():
        if key == "use_quick_model_only":
            if value and config.get("quick_think_llm"):
                config["deep_think_llm"] = config["quick_think_llm"]
            continue
        config[key] = value


def _prepare_runtime_paths(config: dict[str, Any]) -> None:
    """Prepare every writable path used by TradingAgents and yfinance."""
    cache_dir = Path(str(config["data_cache_dir"])).expanduser().resolve()
    results_dir = Path(str(config["results_dir"])).expanduser().resolve()
    memory_log_path = Path(
        str(config["memory_log_path"])
    ).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    memory_log_path.parent.mkdir(parents=True, exist_ok=True)

    # yfinance otherwise opens its SQLite cookie/timezone cache under the
    # interactive user's profile, which is not writable by the unattended
    # Runtime worker. This public API relocates both caches to the run-owned
    # data directory before the first ticker request.
    import yfinance as yf

    yfinance_cache = cache_dir / "yfinance"
    yfinance_cache.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(yfinance_cache))

    config["data_cache_dir"] = str(cache_dir)
    config["results_dir"] = str(results_dir)
    config["memory_log_path"] = str(memory_log_path)
    os.environ["TRADINGAGENTS_CACHE_DIR"] = str(cache_dir)
    os.environ["TRADINGAGENTS_RESULTS_DIR"] = str(results_dir)
    os.environ["TRADINGAGENTS_MEMORY_LOG_PATH"] = str(memory_log_path)


def _source_manifest(
    config: dict[str, Any],
    state: Any,
    invocation: dict[str, Any] | None,
    completed_at: datetime,
) -> list[dict[str, Any]]:
    state = state if isinstance(state, dict) else {}
    vendors = dict(config.get("data_vendors") or {})
    as_of = (
        str(invocation.get("data_cutoff"))
        if invocation
        else completed_at.isoformat()
    )
    specs = (
        (
            "tradingagents.market",
            "market_report",
            (
                str(vendors.get("core_stock_apis") or "unconfigured"),
                str(vendors.get("technical_indicators") or "unconfigured"),
            ),
            ("ohlcv", "technical_indicators"),
        ),
        (
            "tradingagents.fundamentals",
            "fundamentals_report",
            (str(vendors.get("fundamental_data") or "unconfigured"),),
            ("fundamentals", "financial_statements"),
        ),
        (
            "tradingagents.news",
            "news_report",
            (
                str(vendors.get("news_data") or "unconfigured"),
                str(vendors.get("macro_data") or "unconfigured"),
                str(vendors.get("prediction_markets") or "unconfigured"),
            ),
            ("company_news", "macro", "prediction_markets"),
        ),
        (
            "tradingagents.sentiment",
            "sentiment_report",
            (
                str(vendors.get("news_data") or "unconfigured"),
                "stocktwits",
                "reddit",
            ),
            ("news_sentiment", "stocktwits", "reddit"),
        ),
    )
    result = []
    for source, report_key, source_vendors, coverage in specs:
        present = bool(str(state.get(report_key) or "").strip())
        result.append(
            {
                "source": source,
                "status": "OK" if present else "FAILED",
                "as_of": as_of,
                "fetched_at": completed_at.isoformat(),
                "quality_score": 1.0 if present else 0.0,
                "coverage": list(coverage),
                "payload_version": "tradingagents-source:v1",
                "failure_code": "" if present else "ANALYST_REPORT_MISSING",
                "metadata": {
                    "configured_vendors": list(source_vendors),
                    "report_key": report_key,
                    "as_of_basis": (
                        "invocation_data_cutoff"
                        if invocation
                        else "worker_completion"
                    ),
                },
            }
        )
    return result


def _validated_invocation(
    payload: dict[str, Any],
    symbol: str,
    trading_date: str,
) -> dict[str, Any] | None:
    value = payload.get("invocation")
    if value is None:
        return None
    if payload.get("worker_contract_version") != 1:
        raise ValueError("WORKER_CONTRACT_VERSION_INVALID")
    if not isinstance(value, dict):
        raise ValueError("INVOCATION_INVALID")
    required = (
        "invocation_id",
        "run_id",
        "snapshot_id",
        "snapshot_content_hash",
        "data_version",
        "model_version",
        "symbol",
        "trading_date",
        "data_cutoff",
        "requested_at",
    )
    if not all(str(value.get(key, "")).strip() for key in required):
        raise ValueError("INVOCATION_FIELD_REQUIRED")
    if str(value["symbol"]).strip().upper() != symbol:
        raise ValueError("INVOCATION_SYMBOL_MISMATCH")
    if str(value["trading_date"]) != trading_date:
        raise ValueError("INVOCATION_DATE_MISMATCH")
    cutoff = datetime.fromisoformat(str(value["data_cutoff"]))
    requested_at = datetime.fromisoformat(str(value["requested_at"]))
    if (
        cutoff.tzinfo is None
        or cutoff.utcoffset() is None
        or requested_at.tzinfo is None
        or requested_at.utcoffset() is None
    ):
        raise ValueError("INVOCATION_TIMESTAMP_TIMEZONE_REQUIRED")
    if cutoff.astimezone(_UTC) > requested_at.astimezone(_UTC):
        raise ValueError("INVOCATION_CUTOFF_FUTURE")
    return value


def main() -> int:
    try:
        worker_started_at = datetime.now(_UTC)
        payload = json.loads(sys.stdin.read())
        symbol = str(payload["symbol"]).strip().upper()
        trading_date = str(payload["trading_date"]).strip()
        if not symbol or not trading_date:
            raise ValueError("SYMBOL_AND_TRADING_DATE_REQUIRED")
        invocation = _validated_invocation(
            payload,
            symbol,
            trading_date,
        )
        graph, graph_config = _configured_graph(payload)
        state, decision = graph.propagate(symbol, trading_date)
        worker_completed_at = datetime.now(_UTC)
        _emit(
            {
                "ok": True,
                "worker_contract_version": 1,
                "invocation": _json_value(invocation),
                "worker_started_at": worker_started_at.isoformat(),
                "worker_completed_at": worker_completed_at.isoformat(),
                "source_manifest": _source_manifest(
                    graph_config,
                    state,
                    invocation,
                    worker_completed_at,
                ),
                "decision": _json_value(decision),
                "state": _json_value(state),
            }
        )
        return 0
    except Exception as exc:
        _emit(
            {
                "ok": False,
                "error": type(exc).__name__,
                "diagnostic_code": _exception_diagnostic(exc),
                "message": str(exc)[:500],
            }
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
