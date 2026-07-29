from datetime import date, datetime, timedelta, timezone

from trader.models import (
    Position,
    PositionPlan,
    PositionPlanStatus,
    Side,
)
from trader.position_plans import PositionPlanStore
from trader.position_quality import PositionQualityStore


def _weekdays(end: date, count: int) -> list[date]:
    values = []
    current = end
    while len(values) < count:
        if current.weekday() < 5:
            values.append(current)
        current -= timedelta(days=1)
    return list(reversed(values))


def _record(
    store,
    trading_date,
    *,
    broker=10,
    local=10,
    plan=10,
    errors=None,
    duplicates=0,
    kind="SYNTHETIC",
):
    observed_at = datetime.combine(
        trading_date,
        datetime.min.time(),
        tzinfo=timezone.utc,
    ) + timedelta(hours=20)
    return store.record_snapshot(
        trading_date=trading_date,
        observed_at=observed_at,
        broker_positions={"AAPL": broker},
        local_positions={"AAPL": local},
        plan_positions={"AAPL": plan},
        version_errors=errors or [],
        duplicate_adjustments=duplicates,
        observation_kind=kind,
    )


def test_thirty_day_synthetic_position_gate_passes_and_is_idempotent(
    tmp_path,
):
    store = PositionQualityStore(tmp_path / "trade.duckdb")
    days = _weekdays(date(2026, 7, 27), 30)
    for trading_date in days:
        first = _record(store, trading_date)
        assert _record(store, trading_date) == first

    report = store.build_report(
        as_of=days[-1],
        required_days=30,
        observation_kind="SYNTHETIC",
    )

    assert report["ready"]
    assert report["observed_days"] == 30
    assert report["passed_days"] == 30
    assert report["position_mismatches"] == 0
    assert report["silent_rewrites"] == 0
    assert report["duplicate_adjustments"] == 0


def test_gate_rejects_mismatch_silent_rewrite_and_duplicate_adjustment(
    tmp_path,
):
    store = PositionQualityStore(tmp_path / "trade.duckdb")
    days = _weekdays(date(2026, 7, 27), 30)
    for index, trading_date in enumerate(days):
        _record(
            store,
            trading_date,
            local=9 if index == 5 else 10,
            errors=(
                ["plan-1:SILENT_INVALIDATION:2"]
                if index == 10
                else []
            ),
            duplicates=1 if index == 15 else 0,
        )

    report = store.build_report(
        as_of=days[-1],
        required_days=30,
        observation_kind="SYNTHETIC",
    )

    assert not report["ready"]
    assert report["position_mismatches"] == 1
    assert report["silent_rewrites"] == 1
    assert report["duplicate_adjustments"] == 1
    assert len(report["failed_dates"]) == 3


def test_real_and_synthetic_observations_never_mix(tmp_path):
    store = PositionQualityStore(tmp_path / "trade.duckdb")
    trading_date = date(2026, 7, 27)
    _record(store, trading_date, kind="SYNTHETIC")
    _record(store, trading_date, kind="REAL")

    real = store.build_report(
        as_of=trading_date,
        required_days=2,
        observation_kind="REAL",
    )
    synthetic = store.build_report(
        as_of=trading_date,
        required_days=1,
        observation_kind="SYNTHETIC",
    )

    assert not real["ready"]
    assert real["observed_days"] == 1
    assert synthetic["ready"]


def test_real_capture_detects_unlinked_position_plan_rewrite(tmp_path):
    path = tmp_path / "trade.duckdb"
    plan_store = PositionPlanStore(path)
    created_at = datetime(2026, 7, 27, 15, 0, tzinfo=timezone.utc)
    first = PositionPlan(
        position_plan_id="position-quality-plan",
        version_id="position-quality-version-1",
        version=1,
        parent_version_id="",
        symbol="AAPL",
        side=Side.BUY,
        status=PositionPlanStatus.ACTIVE,
        source_trade_plan_id="trade-plan-1",
        initial_fill_id="fill-1",
        initial_entry_price=100,
        initial_quantity=10,
        open_quantity=10,
        average_entry_price=100,
        stop_loss=95,
        take_profit=115,
        invalidation_rules=("PRICE_STOP",),
        change_reason="INITIAL_FILL",
        created_at=created_at,
    )
    plan_store.create(first)
    quality = PositionQualityStore(path)
    positions = [Position("AAPL", qty=10, avg_entry_px=100)]
    quality.capture(
        broker_positions=positions,
        local_positions=positions,
        observed_at=created_at,
    )
    assert quality.build_report(
        as_of=created_at.date(),
        required_days=1,
    )["ready"]

    silent = PositionPlan(
        **{
            **first.__dict__,
            "version_id": "position-quality-version-2",
            "version": 2,
            "parent_version_id": first.version_id,
            "stop_loss": 96,
            "change_reason": "MANUAL_REWRITE",
            "created_at": created_at + timedelta(minutes=1),
        }
    )
    plan_store.append(silent, expected_version=1)
    quality.capture(
        broker_positions=positions,
        local_positions=positions,
        observed_at=created_at + timedelta(minutes=2),
    )
    report = quality.build_report(
        as_of=created_at.date(),
        required_days=1,
    )
    assert not report["ready"]
    assert report["silent_rewrites"] == 1
