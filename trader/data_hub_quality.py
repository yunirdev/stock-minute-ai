"""Auditable Data Hub double-read comparisons and daily quality gates."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable, Mapping

import duckdb

from .data_hub import DataDomain, DataEnvelope, DataStatus

_MISSING = object()


class DifferenceSeverity(StrEnum):
    CRITICAL = "CRITICAL"
    RESEARCH = "RESEARCH"


@dataclass(frozen=True)
class FieldCheck:
    path: str
    severity: DifferenceSeverity
    absolute_tolerance: float = 0.0
    tolerance_bps: float | None = None
    classification: str = ""

    def __post_init__(self) -> None:
        if not self.path.strip():
            raise ValueError("DATA_QUALITY_FIELD_PATH_REQUIRED")
        if (
            not math.isfinite(self.absolute_tolerance)
            or self.absolute_tolerance < 0
        ):
            raise ValueError("DATA_QUALITY_ABSOLUTE_TOLERANCE_INVALID")
        if self.tolerance_bps is not None and (
            not math.isfinite(self.tolerance_bps)
            or self.tolerance_bps < 0
        ):
            raise ValueError("DATA_QUALITY_BPS_TOLERANCE_INVALID")


@dataclass(frozen=True)
class ComparisonPolicy:
    fields: tuple[FieldCheck, ...]
    max_as_of_skew_seconds: float
    status_severity: DifferenceSeverity

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.max_as_of_skew_seconds)
            or self.max_as_of_skew_seconds < 0
        ):
            raise ValueError("DATA_QUALITY_AS_OF_SKEW_INVALID")


@dataclass(frozen=True)
class ApprovedDifferenceRule:
    rule_id: str
    domain: DataDomain
    field: str
    reason: str
    max_absolute_difference: float | None = None
    max_difference_bps: float | None = None
    expires_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.rule_id.strip():
            raise ValueError("DATA_QUALITY_RULE_ID_REQUIRED")
        if not self.field.strip():
            raise ValueError("DATA_QUALITY_RULE_FIELD_REQUIRED")
        if not self.reason.strip():
            raise ValueError("DATA_QUALITY_RULE_REASON_REQUIRED")
        for name in (
            "max_absolute_difference",
            "max_difference_bps",
        ):
            value = getattr(self, name)
            if value is not None and (
                not math.isfinite(value) or value < 0
            ):
                raise ValueError(
                    f"DATA_QUALITY_RULE_{name.upper()}_INVALID"
                )
        if self.expires_at is not None and (
            self.expires_at.tzinfo is None
            or self.expires_at.utcoffset() is None
        ):
            raise ValueError("DATA_QUALITY_RULE_EXPIRY_TIMEZONE_REQUIRED")

    def matches(
        self,
        *,
        domain: DataDomain,
        difference: Mapping[str, Any],
        observed_at: datetime,
    ) -> bool:
        if self.domain != domain or self.field != difference.get("field"):
            return False
        if (
            self.expires_at is not None
            and observed_at > self.expires_at.astimezone(timezone.utc)
        ):
            return False
        if self.max_absolute_difference is not None:
            value = difference.get("absolute_difference")
            if value is None or value > self.max_absolute_difference:
                return False
        if self.max_difference_bps is not None:
            value = difference.get("difference_bps")
            if value is None or value > self.max_difference_bps:
                return False
        return True


@dataclass(frozen=True)
class SourceReadMetrics:
    source_id: str
    latency_ms: float
    request_count: int = 1
    failure_count: int = 0
    quota_applicable: bool = False
    quota_used: float | None = None
    quota_limit: float | None = None

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("DATA_QUALITY_METRIC_SOURCE_REQUIRED")
        if not math.isfinite(self.latency_ms) or self.latency_ms < 0:
            raise ValueError("DATA_QUALITY_LATENCY_INVALID")
        if self.request_count < 1:
            raise ValueError("DATA_QUALITY_REQUEST_COUNT_INVALID")
        if not 0 <= self.failure_count <= self.request_count:
            raise ValueError("DATA_QUALITY_FAILURE_COUNT_INVALID")
        if self.quota_applicable:
            if self.quota_used is None or self.quota_limit is None:
                raise ValueError("DATA_QUALITY_QUOTA_METRIC_REQUIRED")
            if (
                not math.isfinite(self.quota_used)
                or self.quota_used < 0
                or not math.isfinite(self.quota_limit)
                or self.quota_limit <= 0
            ):
                raise ValueError("DATA_QUALITY_QUOTA_METRIC_INVALID")

    @property
    def quota_utilization(self) -> float | None:
        if not self.quota_applicable:
            return None
        return float(self.quota_used) / float(self.quota_limit)


@dataclass(frozen=True)
class DoubleReadObservation:
    observation_id: str
    trading_date: str
    observed_at: datetime
    domain: DataDomain
    key: str
    primary_source: str
    shadow_source: str
    primary_status: DataStatus
    shadow_status: DataStatus
    primary_metrics: SourceReadMetrics
    shadow_metrics: SourceReadMetrics
    comparable: bool
    differences: tuple[Mapping[str, Any], ...]

    @property
    def unclassified_critical_differences(self) -> int:
        return sum(
            difference.get("severity") == DifferenceSeverity.CRITICAL.value
            and difference.get("classification") == "UNCLASSIFIED"
            for difference in self.differences
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "trading_date": self.trading_date,
            "observed_at": self.observed_at.astimezone(
                timezone.utc
            ).isoformat(),
            "domain": self.domain.value,
            "key": self.key,
            "primary_source": self.primary_source,
            "shadow_source": self.shadow_source,
            "primary_status": self.primary_status.value,
            "shadow_status": self.shadow_status.value,
            "primary_metrics": asdict(self.primary_metrics),
            "shadow_metrics": asdict(self.shadow_metrics),
            "comparable": self.comparable,
            "differences": [dict(item) for item in self.differences],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]):
        return cls(
            observation_id=str(payload["observation_id"]),
            trading_date=str(payload["trading_date"]),
            observed_at=datetime.fromisoformat(str(payload["observed_at"])),
            domain=DataDomain(str(payload["domain"])),
            key=str(payload["key"]),
            primary_source=str(payload["primary_source"]),
            shadow_source=str(payload["shadow_source"]),
            primary_status=DataStatus(str(payload["primary_status"])),
            shadow_status=DataStatus(str(payload["shadow_status"])),
            primary_metrics=SourceReadMetrics(
                **dict(payload["primary_metrics"])
            ),
            shadow_metrics=SourceReadMetrics(
                **dict(payload["shadow_metrics"])
            ),
            comparable=bool(payload["comparable"]),
            differences=tuple(payload.get("differences") or ()),
        )


DEFAULT_POLICIES: Mapping[DataDomain, ComparisonPolicy] = {
    DataDomain.MARKET: ComparisonPolicy(
        fields=(
            FieldCheck("symbol", DifferenceSeverity.CRITICAL),
            FieldCheck("timeframe", DifferenceSeverity.CRITICAL),
            FieldCheck(
                "last_price",
                DifferenceSeverity.CRITICAL,
                tolerance_bps=5.0,
            ),
            FieldCheck("bars.#", DifferenceSeverity.CRITICAL),
            FieldCheck(
                "execution_eligible",
                DifferenceSeverity.CRITICAL,
            ),
        ),
        max_as_of_skew_seconds=60.0,
        status_severity=DifferenceSeverity.CRITICAL,
    ),
    DataDomain.BROKER: ComparisonPolicy(
        fields=(
            FieldCheck(
                "equity",
                DifferenceSeverity.CRITICAL,
                absolute_tolerance=0.01,
            ),
            FieldCheck("positions", DifferenceSeverity.CRITICAL),
            FieldCheck("open_orders", DifferenceSeverity.CRITICAL),
            FieldCheck("recent_fills", DifferenceSeverity.CRITICAL),
            FieldCheck(
                "execution_eligible",
                DifferenceSeverity.CRITICAL,
            ),
        ),
        max_as_of_skew_seconds=5.0,
        status_severity=DifferenceSeverity.CRITICAL,
    ),
    DataDomain.CORPORATE: ComparisonPolicy(
        fields=(
            FieldCheck(
                "financial_facts.#",
                DifferenceSeverity.RESEARCH,
                classification="RESEARCH_COVERAGE_VARIANCE",
            ),
            FieldCheck(
                "disclosures.#",
                DifferenceSeverity.RESEARCH,
                classification="RESEARCH_COVERAGE_VARIANCE",
            ),
            FieldCheck(
                "insider_filings.#",
                DifferenceSeverity.RESEARCH,
                classification="RESEARCH_COVERAGE_VARIANCE",
            ),
        ),
        max_as_of_skew_seconds=86_400.0,
        status_severity=DifferenceSeverity.RESEARCH,
    ),
    DataDomain.NEWS: ComparisonPolicy(
        fields=(
            FieldCheck(
                "items.#",
                DifferenceSeverity.RESEARCH,
                classification="RESEARCH_COVERAGE_VARIANCE",
            ),
            FieldCheck(
                "conflicts.#",
                DifferenceSeverity.RESEARCH,
                classification="SOURCE_CONFLICT_VARIANCE",
            ),
        ),
        max_as_of_skew_seconds=3_600.0,
        status_severity=DifferenceSeverity.RESEARCH,
    ),
    DataDomain.MACRO: ComparisonPolicy(
        fields=(
            FieldCheck(
                "observations.#",
                DifferenceSeverity.RESEARCH,
                classification="RESEARCH_COVERAGE_VARIANCE",
            ),
            FieldCheck(
                "missing_series",
                DifferenceSeverity.RESEARCH,
                classification="SOURCE_MISSING_VARIANCE",
            ),
        ),
        max_as_of_skew_seconds=86_400.0,
        status_severity=DifferenceSeverity.RESEARCH,
    ),
    DataDomain.SENTIMENT: ComparisonPolicy(
        fields=(
            FieldCheck(
                "signals.#",
                DifferenceSeverity.RESEARCH,
                classification="RESEARCH_COVERAGE_VARIANCE",
            ),
            FieldCheck(
                "low_quality_sources",
                DifferenceSeverity.RESEARCH,
                classification="SOURCE_QUALITY_VARIANCE",
            ),
        ),
        max_as_of_skew_seconds=43_200.0,
        status_severity=DifferenceSeverity.RESEARCH,
    ),
}


def _resolve(payload: Mapping[str, Any], path: str) -> Any:
    count = path.endswith(".#")
    parts = path[:-2].split(".") if count else path.split(".")
    value: Any = payload
    for part in parts:
        if not isinstance(value, Mapping) or part not in value:
            return _MISSING
        value = value[part]
    if count:
        try:
            return len(value)
        except TypeError:
            return _MISSING
    return value


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _display_value(value: Any) -> Any:
    if value is _MISSING:
        return {"missing": True}
    if isinstance(value, Mapping):
        return {
            "type": "mapping",
            "count": len(value),
            "fingerprint": _fingerprint(value),
        }
    if isinstance(value, (list, tuple)):
        return {
            "type": "sequence",
            "count": len(value),
            "fingerprint": _fingerprint(value),
        }
    return value


def _field_difference(
    check: FieldCheck,
    primary_value: Any,
    shadow_value: Any,
) -> dict[str, Any] | None:
    if primary_value is _MISSING or shadow_value is _MISSING:
        if primary_value is shadow_value:
            return None
        return {
            "field": check.path,
            "primary": _display_value(primary_value),
            "shadow": _display_value(shadow_value),
        }
    if (
        isinstance(primary_value, (int, float))
        and not isinstance(primary_value, bool)
        and isinstance(shadow_value, (int, float))
        and not isinstance(shadow_value, bool)
    ):
        left, right = float(primary_value), float(shadow_value)
        if not math.isfinite(left) or not math.isfinite(right):
            equal = left == right
            absolute_difference = None
            difference_bps = None
        else:
            absolute_difference = abs(left - right)
            difference_bps = (
                absolute_difference / abs(left) * 10_000
                if left
                else (0.0 if right == 0 else float("inf"))
            )
            equal = absolute_difference <= check.absolute_tolerance
            if check.tolerance_bps is not None:
                equal = equal or difference_bps <= check.tolerance_bps
        if equal:
            return None
        return {
            "field": check.path,
            "primary": primary_value,
            "shadow": shadow_value,
            "absolute_difference": absolute_difference,
            "difference_bps": difference_bps,
        }
    if _fingerprint(primary_value) == _fingerprint(shadow_value):
        return None
    return {
        "field": check.path,
        "primary": _display_value(primary_value),
        "shadow": _display_value(shadow_value),
    }


def _classify_difference(
    difference: dict[str, Any],
    *,
    severity: DifferenceSeverity,
    default_classification: str,
    domain: DataDomain,
    observed_at: datetime,
    approved_rules: Iterable[ApprovedDifferenceRule],
) -> dict[str, Any]:
    result = {
        **difference,
        "severity": severity.value,
        "classification": (
            default_classification
            or (
                "UNCLASSIFIED"
                if severity == DifferenceSeverity.CRITICAL
                else "RESEARCH_VARIANCE"
            )
        ),
        "approved_rule_id": "",
    }
    for rule in approved_rules:
        if rule.matches(
            domain=domain,
            difference=result,
            observed_at=observed_at,
        ):
            result["classification"] = "APPROVED_RULE"
            result["approved_rule_id"] = rule.rule_id
            break
    return result


def observe_double_read(
    primary: DataEnvelope,
    shadow: DataEnvelope,
    *,
    observed_at: datetime,
    primary_metrics: SourceReadMetrics,
    shadow_metrics: SourceReadMetrics,
    policy: ComparisonPolicy | None = None,
    approved_rules: Iterable[ApprovedDifferenceRule] = (),
    trading_date: str | None = None,
) -> DoubleReadObservation:
    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        raise ValueError("DATA_QUALITY_OBSERVED_AT_TIMEZONE_REQUIRED")
    observed_at = observed_at.astimezone(timezone.utc)
    if primary.domain != shadow.domain:
        raise ValueError("DATA_QUALITY_DOMAIN_MISMATCH")
    if primary.key != shadow.key:
        raise ValueError("DATA_QUALITY_KEY_MISMATCH")
    selected_policy = policy or DEFAULT_POLICIES.get(
        primary.domain,
        ComparisonPolicy((), 0.0, DifferenceSeverity.RESEARCH),
    )
    rules = tuple(approved_rules)
    differences: list[dict[str, Any]] = []
    comparable = (
        primary.status != DataStatus.FAILED
        and shadow.status != DataStatus.FAILED
    )

    if primary.status != shadow.status or not comparable:
        differences.append(
            _classify_difference(
                {
                    "field": "$envelope.status",
                    "primary": primary.status.value,
                    "shadow": shadow.status.value,
                },
                severity=selected_policy.status_severity,
                default_classification="",
                domain=primary.domain,
                observed_at=observed_at,
                approved_rules=rules,
            )
        )
    if comparable:
        as_of_skew = abs(
            (primary.as_of - shadow.as_of).total_seconds()
        )
        if as_of_skew > selected_policy.max_as_of_skew_seconds:
            differences.append(
                _classify_difference(
                    {
                        "field": "$envelope.as_of",
                        "primary": primary.as_of.isoformat(),
                        "shadow": shadow.as_of.isoformat(),
                        "absolute_difference": as_of_skew,
                    },
                    severity=selected_policy.status_severity,
                    default_classification=(
                        "AS_OF_VARIANCE"
                        if selected_policy.status_severity
                        == DifferenceSeverity.RESEARCH
                        else ""
                    ),
                    domain=primary.domain,
                    observed_at=observed_at,
                    approved_rules=rules,
                )
            )
        for check in selected_policy.fields:
            difference = _field_difference(
                check,
                _resolve(primary.payload, check.path),
                _resolve(shadow.payload, check.path),
            )
            if difference is None:
                continue
            differences.append(
                _classify_difference(
                    difference,
                    severity=check.severity,
                    default_classification=check.classification,
                    domain=primary.domain,
                    observed_at=observed_at,
                    approved_rules=rules,
                )
            )

    raw = {
        "trading_date": trading_date or observed_at.date().isoformat(),
        "observed_at": observed_at.isoformat(),
        "domain": primary.domain.value,
        "key": primary.key,
        "primary_source": primary_metrics.source_id,
        "shadow_source": shadow_metrics.source_id,
        "primary_status": primary.status.value,
        "shadow_status": shadow.status.value,
        "primary_metrics": asdict(primary_metrics),
        "shadow_metrics": asdict(shadow_metrics),
        "comparable": comparable,
        "differences": differences,
    }
    fingerprint = json.dumps(
        raw,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return DoubleReadObservation(
        observation_id=(
            "data-quality-"
            + hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:20]
        ),
        trading_date=raw["trading_date"],
        observed_at=observed_at,
        domain=primary.domain,
        key=primary.key,
        primary_source=primary_metrics.source_id,
        shadow_source=shadow_metrics.source_id,
        primary_status=primary.status,
        shadow_status=shadow.status,
        primary_metrics=primary_metrics,
        shadow_metrics=shadow_metrics,
        comparable=comparable,
        differences=tuple(differences),
    )


@dataclass(frozen=True)
class DataHubQualityThresholds:
    required_days: int = 20
    min_comparisons_per_day: int = 1
    max_failure_rate: float = 0.01
    max_primary_p95_latency_ms: float = 1_000.0
    max_shadow_p95_latency_ms: float = 3_000.0
    max_quota_utilization: float = 0.8

    def __post_init__(self) -> None:
        if self.required_days < 1:
            raise ValueError("DATA_QUALITY_REQUIRED_DAYS_INVALID")
        if self.min_comparisons_per_day < 1:
            raise ValueError(
                "DATA_QUALITY_MIN_COMPARISONS_PER_DAY_INVALID"
            )
        if not 0 <= self.max_failure_rate <= 1:
            raise ValueError("DATA_QUALITY_FAILURE_RATE_INVALID")
        for name in (
            "max_primary_p95_latency_ms",
            "max_shadow_p95_latency_ms",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"DATA_QUALITY_{name.upper()}_INVALID"
                )
        if (
            not math.isfinite(self.max_quota_utilization)
            or self.max_quota_utilization < 0
        ):
            raise ValueError("DATA_QUALITY_QUOTA_UTILIZATION_INVALID")


def _p95(values: Iterable[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def _metric_summary(
    observations: Iterable[DoubleReadObservation],
) -> dict[str, Any]:
    rows = list(observations)
    primary_metrics = [row.primary_metrics for row in rows]
    shadow_metrics = [row.shadow_metrics for row in rows]
    all_metrics = primary_metrics + shadow_metrics
    request_count = sum(item.request_count for item in all_metrics)
    failure_count = sum(item.failure_count for item in all_metrics)
    quota_values = [
        item.quota_utilization
        for item in all_metrics
        if item.quota_utilization is not None
    ]
    return {
        "request_count": request_count,
        "failure_count": failure_count,
        "failure_rate": (
            failure_count / request_count if request_count else 1.0
        ),
        "primary_p95_latency_ms": _p95(
            item.latency_ms for item in primary_metrics
        ),
        "shadow_p95_latency_ms": _p95(
            item.latency_ms for item in shadow_metrics
        ),
        "quota_applicable_reads": len(quota_values),
        "max_quota_utilization": max(quota_values, default=0.0),
    }


def generate_data_hub_quality_report(
    observations: Iterable[DoubleReadObservation],
    *,
    thresholds: DataHubQualityThresholds | None = None,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    limits = thresholds or DataHubQualityThresholds()
    rows = sorted(
        observations,
        key=lambda item: (item.trading_date, item.observed_at),
    )
    dates = sorted({item.trading_date for item in rows})
    selected_dates = dates[-limits.required_days :]
    selected = [
        item for item in rows if item.trading_date in selected_dates
    ]
    by_date = {
        trading_date: [
            item
            for item in selected
            if item.trading_date == trading_date
        ]
        for trading_date in selected_dates
    }

    differences = [
        difference
        for item in selected
        for difference in item.differences
    ]
    critical_differences = [
        item
        for item in differences
        if item.get("severity") == DifferenceSeverity.CRITICAL.value
    ]
    unclassified_critical = [
        item
        for item in critical_differences
        if item.get("classification") == "UNCLASSIFIED"
    ]
    approved_critical = [
        item
        for item in critical_differences
        if item.get("classification") == "APPROVED_RULE"
    ]
    metrics = _metric_summary(selected)
    daily = []
    for trading_date, day_rows in by_date.items():
        day_metrics = _metric_summary(day_rows)
        daily.append(
            {
                "trading_date": trading_date,
                "comparisons": len(day_rows),
                "unclassified_critical_differences": sum(
                    item.unclassified_critical_differences
                    for item in day_rows
                ),
                **day_metrics,
            }
        )

    observed_days = len(selected_dates)
    daily_coverage_passed = (
        observed_days >= limits.required_days
        and all(
            len(day_rows) >= limits.min_comparisons_per_day
            for day_rows in by_date.values()
        )
    )
    gates = {
        "observation_window": daily_coverage_passed,
        "critical_differences": not unclassified_critical,
        "failure_rate": (
            metrics["failure_rate"] <= limits.max_failure_rate
        ),
        "primary_latency": (
            metrics["primary_p95_latency_ms"]
            <= limits.max_primary_p95_latency_ms
        ),
        "shadow_latency": (
            metrics["shadow_p95_latency_ms"]
            <= limits.max_shadow_p95_latency_ms
        ),
        "quota_utilization": (
            metrics["max_quota_utilization"]
            <= limits.max_quota_utilization
        ),
    }
    report = {
        "generated_at": (
            generated_at or datetime.now(timezone.utc)
        ).astimezone(timezone.utc).isoformat(),
        "required_days": limits.required_days,
        "observed_trading_days": observed_days,
        "available_trading_days": len(dates),
        "window_start": selected_dates[0] if selected_dates else None,
        "window_end": selected_dates[-1] if selected_dates else None,
        "comparisons": len(selected),
        "total_differences": len(differences),
        "critical_differences": len(critical_differences),
        "approved_critical_differences": len(approved_critical),
        "unclassified_critical_differences": len(unclassified_critical),
        "approved_rule_ids": sorted(
            {
                str(item.get("approved_rule_id"))
                for item in approved_critical
                if item.get("approved_rule_id")
            }
        ),
        **metrics,
        "thresholds": asdict(limits),
        "gates": gates,
        "passed": all(gates.values()),
        "daily": daily,
        "execution_input_switched": False,
    }
    fingerprint = json.dumps(
        report,
        sort_keys=True,
        separators=(",", ":"),
    )
    report["report_id"] = (
        "data-hub-quality-"
        + hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:20]
    )
    return report


class DataHubQualityStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        conn = duckdb.connect(str(self.db_path))
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS data_hub_double_reads (
                    observation_id TEXT PRIMARY KEY,
                    trading_date TEXT,
                    observed_at TIMESTAMPTZ,
                    domain TEXT,
                    key TEXT,
                    payload TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS data_hub_quality_reports (
                    report_id TEXT PRIMARY KEY,
                    generated_at TIMESTAMPTZ,
                    window_start TEXT,
                    window_end TEXT,
                    passed BOOLEAN,
                    payload TEXT
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def save_observation(
        self,
        observation: DoubleReadObservation,
    ) -> bool:
        conn = duckdb.connect(str(self.db_path))
        try:
            existing = conn.execute(
                "SELECT 1 FROM data_hub_double_reads "
                "WHERE observation_id=?",
                [observation.observation_id],
            ).fetchone()
            if existing:
                return False
            conn.execute(
                "INSERT INTO data_hub_double_reads VALUES (?,?,?,?,?,?)",
                [
                    observation.observation_id,
                    observation.trading_date,
                    observation.observed_at,
                    observation.domain.value,
                    observation.key,
                    json.dumps(
                        observation.to_dict(),
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                ],
            )
            conn.commit()
            return True
        finally:
            conn.close()

    def load_observations(self) -> list[DoubleReadObservation]:
        conn = duckdb.connect(str(self.db_path), read_only=True)
        try:
            rows = conn.execute(
                "SELECT payload FROM data_hub_double_reads "
                "ORDER BY trading_date, observed_at"
            ).fetchall()
        finally:
            conn.close()
        return [
            DoubleReadObservation.from_dict(json.loads(row[0]))
            for row in rows
        ]

    def save_report(self, report: Mapping[str, Any]) -> bool:
        conn = duckdb.connect(str(self.db_path))
        try:
            existing = conn.execute(
                "SELECT 1 FROM data_hub_quality_reports "
                "WHERE report_id=?",
                [report["report_id"]],
            ).fetchone()
            if existing:
                return False
            conn.execute(
                "INSERT INTO data_hub_quality_reports VALUES (?,?,?,?,?,?)",
                [
                    report["report_id"],
                    datetime.fromisoformat(str(report["generated_at"])),
                    report["window_start"],
                    report["window_end"],
                    bool(report["passed"]),
                    json.dumps(
                        dict(report),
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                ],
            )
            conn.commit()
            return True
        finally:
            conn.close()
