from dataclasses import replace
from datetime import datetime, timedelta, timezone

import duckdb
import pytest

from trader.models import (
    ResearchQuality,
    ResearchSnapshot,
    ResearchSourceManifestEntry,
    ResearchSourceStatus,
)
from trader.research_snapshot import (
    CURRENT_SNAPSHOT_SCHEMA_VERSION,
    ResearchSnapshotStore,
    snapshot_from_dict,
    snapshot_to_dict,
)


CAPTURED = datetime(2026, 7, 25, 21, tzinfo=timezone.utc)


def _source(
    *,
    status=ResearchSourceStatus.OK,
    quality_score=1.0,
    failure_code="",
) -> ResearchSourceManifestEntry:
    return ResearchSourceManifestEntry(
        source="alpaca.market.bars",
        status=status,
        as_of=CAPTURED - timedelta(minutes=5),
        fetched_at=CAPTURED - timedelta(minutes=1),
        quality_score=quality_score,
        coverage=("ohlcv", "volume", "ohlcv"),
        payload_version="bars:v1",
        failure_code=failure_code,
        metadata={"timeframe": "5m"},
    )


def _snapshot(snapshot_id="snapshot-1") -> ResearchSnapshot:
    return ResearchSnapshot(
        snapshot_id=snapshot_id,
        run_id="research-run-1",
        symbol="aapl",
        trading_date="2026-07-25",
        as_of=CAPTURED - timedelta(minutes=5),
        data_cutoff=CAPTURED - timedelta(minutes=5),
        captured_at=CAPTURED,
        source_manifest=(_source(),),
        quality=ResearchQuality.GOOD,
        quality_score=1.0,
        payload_version="research-input:v1",
        payload={"bars": [{"close": 213.5}]},
        schema_version=CURRENT_SNAPSHOT_SCHEMA_VERSION,
        created_at=CAPTURED,
    )


def test_snapshot_serialization_and_store_round_trip(tmp_path):
    original = _snapshot()

    serialized = snapshot_to_dict(original)
    restored = snapshot_from_dict(serialized)
    store = ResearchSnapshotStore(tmp_path / "research.duckdb")

    assert restored == original
    assert store.save(original)
    assert not store.save(original)
    assert store.get(original.snapshot_id) == original
    assert store.list_for_symbol("aapl", trading_date="2026-07-25") == [
        original
    ]


def test_source_manifest_records_explicit_failure():
    source = _source(
        status=ResearchSourceStatus.FAILED,
        quality_score=0.0,
        failure_code="SOURCE_TIMEOUT",
    )

    assert source.failure_code == "SOURCE_TIMEOUT"
    assert source.quality_score == 0.0


@pytest.mark.parametrize("quality_score", [-0.01, 1.01, float("nan")])
def test_source_quality_rejects_invalid_range(quality_score):
    with pytest.raises(ValueError, match="SOURCE_QUALITY_SCORE_OUT_OF_RANGE"):
        _source(quality_score=quality_score)


def test_failed_source_requires_zero_quality_and_failure_code():
    with pytest.raises(
        ValueError,
        match="FAILED_SOURCE_QUALITY_MUST_BE_ZERO",
    ):
        _source(
            status=ResearchSourceStatus.FAILED,
            quality_score=0.5,
            failure_code="TIMEOUT",
        )
    with pytest.raises(ValueError, match="FAILED_SOURCE_CODE_REQUIRED"):
        _source(
            status=ResearchSourceStatus.MISSING,
            quality_score=0.0,
        )


def test_snapshot_rejects_naive_or_future_ordered_timestamps():
    with pytest.raises(
        ValueError,
        match="SNAPSHOT_AS_OF_TIMEZONE_REQUIRED",
    ):
        ResearchSnapshot(
            **{
                **_snapshot().__dict__,
                "as_of": datetime(2026, 7, 25, 20, 55),
            }
        )

    value = snapshot_to_dict(_snapshot())
    value["data_cutoff"] = (
        CAPTURED + timedelta(minutes=1)
    ).isoformat()
    with pytest.raises(
        ValueError,
        match="SNAPSHOT_DATA_CUTOFF_AFTER_CAPTURE",
    ):
        snapshot_from_dict(value)


def test_snapshot_rejects_invalid_quality_and_missing_current_manifest():
    with pytest.raises(
        ValueError,
        match="FAILED_SNAPSHOT_QUALITY_MUST_BE_ZERO",
    ):
        ResearchSnapshot(
            **{
                **_snapshot().__dict__,
                "quality": ResearchQuality.FAILED,
                "quality_score": 0.5,
            }
        )
    with pytest.raises(
        ValueError,
        match="SNAPSHOT_SOURCE_MANIFEST_REQUIRED",
    ):
        ResearchSnapshot(
            **{
                **_snapshot().__dict__,
                "source_manifest": (),
            }
        )


def test_legacy_snapshot_table_migrates_without_losing_row(tmp_path):
    db_path = tmp_path / "legacy.duckdb"
    conn = duckdb.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE research_snapshots (
            snapshot_id TEXT PRIMARY KEY,
            symbol TEXT,
            as_of TIMESTAMPTZ,
            payload TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO research_snapshots VALUES (?,?,?,?)",
        [
            "legacy-1",
            "MSFT",
            CAPTURED,
            '{"legacy":true}',
        ],
    )
    conn.close()

    store = ResearchSnapshotStore(db_path)
    legacy = store.get("legacy-1")

    assert legacy is not None
    assert legacy.snapshot_id == "legacy-1"
    assert legacy.symbol == "MSFT"
    assert legacy.trading_date == "2026-07-25"
    assert legacy.quality == ResearchQuality.UNKNOWN
    assert legacy.quality_score == 0.0
    assert legacy.payload_version == "legacy"
    assert legacy.schema_version == 1
    assert legacy.source_manifest == ()
    assert legacy.payload == {"legacy": True}

    current = _snapshot("snapshot-current")
    assert store.save(current)
    assert store.get(current.snapshot_id) == current


def test_snapshot_deserialization_rejects_invalid_time_and_quality():
    value = snapshot_to_dict(_snapshot())
    value["as_of"] = "2026-07-25T20:55:00"
    with pytest.raises(
        ValueError,
        match="SNAPSHOT_AS_OF_TIMEZONE_REQUIRED",
    ):
        snapshot_from_dict(value)

    value = snapshot_to_dict(_snapshot())
    value["quality"] = "FRESHISH"
    with pytest.raises(ValueError, match="SNAPSHOT_QUALITY_INVALID"):
        snapshot_from_dict(value)


def test_snapshot_store_deduplicates_and_rejects_identity_conflict(tmp_path):
    store = ResearchSnapshotStore(tmp_path / "research.duckdb")
    original = _snapshot()

    first = store.save_or_get(original)
    identical = store.save_or_get(original)
    alias = store.save_or_get(
        replace(original, snapshot_id="snapshot-alias")
    )

    assert first.created and not first.deduplicated
    assert identical.snapshot_id == original.snapshot_id
    assert not identical.created and identical.deduplicated
    assert alias.snapshot_id == original.snapshot_id
    assert not alias.created and alias.deduplicated
    assert store.get("snapshot-alias") is None

    with pytest.raises(
        ValueError,
        match="SNAPSHOT_IMMUTABLE_CONFLICT",
    ):
        store.save_or_get(
            replace(
                original,
                payload={"bars": [{"close": 999.0}]},
            )
        )


def test_run_binding_replay_cross_day_and_keep_all_retention(tmp_path):
    db_path = tmp_path / "research.duckdb"
    store = ResearchSnapshotStore(db_path)
    first = _snapshot("snapshot-day-1")
    day_two_time = CAPTURED + timedelta(days=1)
    second_source = replace(
        _source(),
        as_of=day_two_time - timedelta(minutes=5),
        fetched_at=day_two_time - timedelta(minutes=1),
    )
    second = replace(
        first,
        snapshot_id="snapshot-day-2",
        run_id="research-run-2",
        trading_date="2026-07-26",
        as_of=day_two_time - timedelta(minutes=5),
        data_cutoff=day_two_time - timedelta(minutes=5),
        captured_at=day_two_time,
        source_manifest=(second_source,),
        payload={"bars": [{"close": 214.0}]},
        created_at=day_two_time,
    )
    assert store.save(first)
    assert store.save(second)
    assert store.bind_to_run(
        run_id=first.run_id,
        symbol=first.symbol,
        trading_date=first.trading_date,
        snapshot_id=first.snapshot_id,
        bound_at=first.created_at,
    )
    assert not store.bind_to_run(
        run_id=first.run_id,
        symbol=first.symbol,
        trading_date=first.trading_date,
        snapshot_id=first.snapshot_id,
        bound_at=first.created_at,
    )
    assert store.bind_to_run(
        run_id=second.run_id,
        symbol=second.symbol,
        trading_date=second.trading_date,
        snapshot_id=second.snapshot_id,
        bound_at=second.created_at,
    )

    replayed = store.replay_for_run(first.run_id, first.symbol)
    assert replayed == first
    replayed.payload["bars"][0]["close"] = -1
    assert (
        store.replay_for_run(first.run_id, first.symbol).payload["bars"][0][
            "close"
        ]
        == 213.5
    )
    assert store.list_for_symbol("AAPL") == [first, second]
    assert store.retention_policy() == {
        "mode": "KEEP_ALL",
        "automatic_delete": False,
        "minimum_days": None,
    }

    alternate = replace(
        first,
        snapshot_id="snapshot-alternate",
        payload={"bars": [{"close": 212.0}]},
    )
    assert store.save(alternate)
    with pytest.raises(
        ValueError,
        match="RUN_SNAPSHOT_BINDING_IMMUTABLE",
    ):
        store.bind_to_run(
            run_id=first.run_id,
            symbol=first.symbol,
            trading_date=first.trading_date,
            snapshot_id=alternate.snapshot_id,
        )
    assert len(store.list_for_symbol("AAPL")) == 3


def test_replay_detects_persisted_content_tampering(tmp_path):
    db_path = tmp_path / "research.duckdb"
    store = ResearchSnapshotStore(db_path)
    snapshot = _snapshot()
    store.save(snapshot)
    conn = duckdb.connect(str(db_path))
    conn.execute(
        "UPDATE research_snapshots SET payload=? WHERE snapshot_id=?",
        ['{"bars":[{"close":0}]}', snapshot.snapshot_id],
    )
    conn.close()

    with pytest.raises(
        ValueError,
        match="SNAPSHOT_CONTENT_HASH_MISMATCH",
    ):
        store.replay(snapshot.snapshot_id)


def test_source_manifest_rejects_future_as_of_relative_to_fetch():
    with pytest.raises(ValueError, match="SOURCE_AS_OF_AFTER_FETCH"):
        ResearchSourceManifestEntry(
            source="future-source",
            status=ResearchSourceStatus.OK,
            as_of=CAPTURED,
            fetched_at=CAPTURED - timedelta(seconds=1),
            quality_score=1.0,
            coverage=("facts",),
            payload_version="v1",
        )
