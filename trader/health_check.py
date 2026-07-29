"""Independent architecture health checks for the engine, database, and UI."""
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

from .bug_reporting import BugReporter


@dataclass(frozen=True)
class HealthFinding:
    code: str
    status: str
    component: str
    message: str
    details: dict[str, Any]


_MOJIBAKE_MARKERS = ("Ã", "Â", "â€", "ðŸ", "æœ", "çš")


def _looks_like_mojibake(text: str) -> bool:
    return sum(text.count(marker) for marker in _MOJIBAKE_MARKERS) >= 3

def _parse_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def _heartbeat_timestamp(db_path: str, sidecar_path: Path) -> datetime | None:
    timestamps: list[datetime] = []
    if sidecar_path.exists():
        try:
            payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
            parsed = _parse_timestamp(payload.get("ts"))
            if parsed:
                timestamps.append(parsed)
        except (OSError, ValueError, TypeError):
            pass
    if Path(db_path).exists():
        try:
            con = duckdb.connect(db_path, read_only=True)
            try:
                row = con.execute("SELECT MAX(ts) FROM heartbeat").fetchone()
            finally:
                con.close()
            parsed = _parse_timestamp(row[0] if row else None)
            if parsed:
                timestamps.append(parsed)
        except duckdb.Error:
            pass
    return max(timestamps) if timestamps else None


def check_architecture(
    db_path: str = "trade.duckdb",
    *,
    heartbeat_path: str | Path = "logs/heartbeat.json",
    max_stale_seconds: int = 120,
    ui_url: str = "",
    now: datetime | None = None,
) -> list[HealthFinding]:
    """Return machine-readable findings without mutating trading state."""
    current = now or datetime.now(timezone.utc)
    findings: list[HealthFinding] = []
    heartbeat_ts = _heartbeat_timestamp(db_path, Path(heartbeat_path))
    if heartbeat_ts is None:
        findings.append(HealthFinding(
            "heartbeat_missing", "critical", "runtime",
            "Runtime heartbeat is missing.", {},
        ))
    else:
        age = max(0.0, (current - heartbeat_ts).total_seconds())
        if age > max_stale_seconds:
            findings.append(HealthFinding(
                "heartbeat_stale", "critical", "runtime",
                "Runtime heartbeat is stale.",
                {"age_seconds": round(age, 1), "last_heartbeat": heartbeat_ts.isoformat()},
            ))

    required = {"heartbeat", "bug_issues", "bug_events", "order_intents"}
    if not Path(db_path).exists():
        findings.append(HealthFinding(
            "database_missing", "critical", "database",
            "Trading database does not exist.", {"db_path": str(db_path)},
        ))
    else:
        try:
            con = duckdb.connect(db_path, read_only=True)
            try:
                tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
            finally:
                con.close()
            missing = sorted(required - tables)
            if missing:
                findings.append(HealthFinding(
                    "database_schema_incomplete", "critical", "database",
                    "Required health or order tables are missing.",
                    {"missing_tables": missing},
                ))
        except duckdb.Error as exc:
            findings.append(HealthFinding(
                "database_unreadable", "critical", "database",
                "Trading database cannot be read.",
                {"error_type": type(exc).__name__},
            ))

    if ui_url:
        url = ui_url.rstrip("/") + "/healthz"
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if response.status != 200 or payload.get("status") != "ok":
                raise ValueError("unexpected health response")
        except (OSError, ValueError, urllib.error.URLError) as exc:
            findings.append(HealthFinding(
                "ui_unreachable", "critical", "ui",
                "UI health endpoint is unavailable.",
                {"url": url, "error_type": type(exc).__name__},
            ))
    source_dir = Path(__file__).resolve().parent
    affected_sources = []
    for name in ("main.py", "runtime.py", "monitor_nice.py"):
        source = source_dir / name
        try:
            if _looks_like_mojibake(source.read_text(encoding="utf-8")):
                affected_sources.append(name)
        except OSError:
            continue
    if affected_sources:
        findings.append(HealthFinding(
            "source_encoding_corrupt", "critical", "source",
            "User-visible source strings contain mojibake.",
            {"files": affected_sources},
        ))
    return findings


def report_findings(findings: list[HealthFinding], db_path: str) -> None:
    reporter = BugReporter(db_path, "architecture_health")
    for finding in findings:
        reporter.capture_message(
            finding.message,
            error_type=finding.code,
            operation="architecture.health_check",
            severity=finding.status.upper(),
            context=finding.details,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Check stock-minute-ai health")
    parser.add_argument("--db", default="trade.duckdb")
    parser.add_argument("--heartbeat", default="logs/heartbeat.json")
    parser.add_argument("--max-stale-seconds", type=int, default=120)
    parser.add_argument("--ui-url", default="")
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args()
    findings = check_architecture(
        args.db,
        heartbeat_path=args.heartbeat,
        max_stale_seconds=args.max_stale_seconds,
        ui_url=args.ui_url,
    )
    if findings and not args.no_report and Path(args.db).exists():
        report_findings(findings, args.db)
    print(json.dumps([asdict(item) for item in findings], ensure_ascii=False, indent=2))
    return 1 if any(item.status == "critical" for item in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())