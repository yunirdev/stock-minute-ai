from datetime import datetime, timedelta, timezone

from trader.data_hub import DataDomain, DataEnvelope, DataStatus
from trader.data_hub_quality import (
    ApprovedDifferenceRule,
    DataHubQualityStore,
    DataHubQualityThresholds,
    SourceReadMetrics,
    generate_data_hub_quality_report,
    observe_double_read,
)

NOW = datetime(2026, 7, 26, 18, tzinfo=timezone.utc)


def _market(*, price=100.0, as_of=NOW, status=DataStatus.OK):
    return DataEnvelope(
        request_id="request",
        domain=DataDomain.MARKET,
        key="AAPL",
        source_id="source",
        status=status,
        payload={
            "symbol": "AAPL",
            "timeframe": "5m",
            "last_price": price,
            "bars": [{"close": price}] * 2,
            "execution_eligible": True,
        },
        as_of=as_of,
        fetched_at=NOW,
        expires_at=NOW,
        quality_score=1.0,
    )


def _metrics(
    source,
    *,
    latency=100.0,
    failures=0,
    quota_used=40.0,
):
    return SourceReadMetrics(
        source,
        latency_ms=latency,
        failure_count=failures,
        quota_applicable=True,
        quota_used=quota_used,
        quota_limit=100.0,
    )


def _observation(
    observed_at,
    *,
    primary=None,
    shadow=None,
    primary_metrics=None,
    shadow_metrics=None,
    rules=(),
):
    return observe_double_read(
        primary or _market(),
        shadow or _market(),
        observed_at=observed_at,
        primary_metrics=primary_metrics or _metrics("runtime_primary"),
        shadow_metrics=shadow_metrics or _metrics(
            "data_hub_shadow",
            latency=200.0,
        ),
        approved_rules=rules,
        trading_date=observed_at.date().isoformat(),
    )


def _twenty_trading_dates():
    dates = []
    current = datetime(2026, 6, 29, 18, tzinfo=timezone.utc)
    while len(dates) < 20:
        if current.weekday() < 5:
            dates.append(current)
        current += timedelta(days=1)
    return dates


def test_critical_market_difference_is_unclassified_until_rule_approves():
    unapproved = _observation(
        NOW,
        shadow=_market(price=99.0),
    )

    assert unapproved.unclassified_critical_differences == 1
    difference = unapproved.differences[0]
    assert difference["field"] == "last_price"
    assert difference["classification"] == "UNCLASSIFIED"
    assert difference["difference_bps"] == 100.0

    rule = ApprovedDifferenceRule(
        rule_id="approved-delayed-price",
        domain=DataDomain.MARKET,
        field="last_price",
        reason="Shadow feed is one completed bar behind during validation.",
        max_difference_bps=101.0,
        expires_at=NOW + timedelta(days=30),
    )
    approved = _observation(
        NOW,
        shadow=_market(price=99.0),
        rules=(rule,),
    )

    assert approved.unclassified_critical_differences == 0
    assert approved.differences[0]["classification"] == "APPROVED_RULE"
    assert approved.differences[0]["approved_rule_id"] == rule.rule_id


def test_research_domain_differences_are_classified_not_critical():
    primary = DataEnvelope(
        request_id="news",
        domain=DataDomain.NEWS,
        key="AAPL",
        source_id="old",
        status=DataStatus.OK,
        payload={"items": [1, 2], "conflicts": []},
        as_of=NOW,
        fetched_at=NOW,
        expires_at=NOW,
        quality_score=1.0,
    )
    shadow = DataEnvelope(
        **{
            **primary.__dict__,
            "source_id": "new",
            "payload": {"items": [1], "conflicts": []},
        }
    )

    observation = observe_double_read(
        primary,
        shadow,
        observed_at=NOW,
        primary_metrics=_metrics("old"),
        shadow_metrics=_metrics("new"),
    )

    assert observation.unclassified_critical_differences == 0
    assert observation.differences[0]["severity"] == "RESEARCH"
    assert observation.differences[0]["classification"] == (
        "RESEARCH_COVERAGE_VARIANCE"
    )


def test_twenty_day_quality_report_passes_and_store_is_idempotent(tmp_path):
    observations = [
        _observation(observed_at)
        for observed_at in _twenty_trading_dates()
    ]
    report = generate_data_hub_quality_report(
        observations,
        generated_at=NOW,
    )

    assert report["passed"]
    assert report["observed_trading_days"] == 20
    assert report["comparisons"] == 20
    assert report["unclassified_critical_differences"] == 0
    assert report["failure_rate"] == 0.0
    assert report["primary_p95_latency_ms"] == 100.0
    assert report["shadow_p95_latency_ms"] == 200.0
    assert report["max_quota_utilization"] == 0.4
    assert all(report["gates"].values())
    assert report["execution_input_switched"] is False

    store = DataHubQualityStore(tmp_path / "quality.duckdb")
    for observation in observations:
        assert store.save_observation(observation)
        assert not store.save_observation(observation)
    loaded = store.load_observations()
    assert [item.observation_id for item in loaded] == [
        item.observation_id for item in observations
    ]
    assert store.save_report(report)
    assert not store.save_report(report)


def test_report_fails_for_unclassified_critical_difference():
    observations = [
        _observation(observed_at)
        for observed_at in _twenty_trading_dates()
    ]
    observations[-1] = _observation(
        observations[-1].observed_at,
        shadow=_market(price=98.0),
    )

    report = generate_data_hub_quality_report(observations)

    assert not report["passed"]
    assert report["unclassified_critical_differences"] == 1
    assert not report["gates"]["critical_differences"]


def test_report_gates_failure_latency_quota_and_incomplete_window():
    dates = _twenty_trading_dates()[:19]
    observations = [
        _observation(observed_at)
        for observed_at in dates
    ]
    observations[-1] = _observation(
        observations[-1].observed_at,
        primary_metrics=_metrics(
            "runtime_primary",
            latency=1_500.0,
            failures=1,
            quota_used=95.0,
        ),
        shadow_metrics=_metrics(
            "data_hub_shadow",
            latency=3_500.0,
        ),
    )
    thresholds = DataHubQualityThresholds(max_failure_rate=0.01)

    report = generate_data_hub_quality_report(
        observations,
        thresholds=thresholds,
    )

    assert not report["passed"]
    assert report["gates"] == {
        "observation_window": False,
        "critical_differences": True,
        "failure_rate": False,
        "primary_latency": False,
        "shadow_latency": False,
        "quota_utilization": False,
    }
