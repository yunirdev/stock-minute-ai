from datetime import datetime, timedelta, timezone

from trader.focus_pool import FocusPoolInput, FocusPoolPolicy, FocusPoolStore
from trader.universe_registry import UniverseAsset, UniverseRegistryStore

NOW = datetime(2026, 7, 27, 20, 0, tzinfo=timezone.utc)


def _universe(db_path):
    assets = [
        UniverseAsset(
            symbol=symbol,
            asset_type=asset_type,
            exchange="NASDAQ",
            status=status,
            tradable=tradable,
            source="alpaca-assets",
            as_of=NOW,
        )
        for symbol, asset_type, status, tradable in (
            ("AAPL", "STOCK", "ACTIVE", True),
            ("MSFT", "STOCK", "ACTIVE", True),
            ("SPY", "ETF", "ACTIVE", True),
            ("VTSAX", "FUND", "ACTIVE", False),
            ("OLD", "STOCK", "DELISTED", False),
        )
    ]
    return UniverseRegistryStore(db_path).create_version(
        universe_name="US listed",
        source_version="source-v1",
        assets=assets,
        as_of=NOW,
        created_at=NOW,
    )


def _input(
    symbol,
    *,
    holdout=True,
    holdout_score=0.8,
    liquidity=30_000_000,
    quality=0.95,
):
    return FocusPoolInput(
        symbol=symbol,
        holdout_reliable=holdout,
        holdout_score=holdout_score,
        average_dollar_volume=liquidity,
        data_quality=quality,
    )


def test_focus_pool_is_deterministic_ranked_and_fully_audited(tmp_path):
    db_path = tmp_path / "research.duckdb"
    universe = _universe(db_path)
    store = FocusPoolStore(db_path)
    policy = FocusPoolPolicy(max_size=2)
    inputs = [
        _input("AAPL", holdout_score=0.9),
        _input("MSFT", holdout=False),
        _input("SPY", holdout_score=0.7),
    ]

    first = store.attempt_build(
        pool_name="daily-focus",
        universe_version=universe["version_id"],
        inputs=inputs,
        policy=policy,
        as_of=NOW,
        created_at=NOW,
    )
    duplicate = store.attempt_build(
        pool_name="daily-focus",
        universe_version=universe["version_id"],
        inputs=reversed(inputs),
        policy=policy,
        as_of=NOW,
        created_at=NOW,
    )

    assert duplicate == first
    assert first["member_count"] == 2
    assert len(first["decisions"]) == universe["asset_count"]
    included = [row for row in first["decisions"] if row["included"]]
    assert [(row["symbol"], row["rank"]) for row in included] == [
        ("AAPL", 1),
        ("SPY", 2),
    ]
    reasons = {row["symbol"]: row["reasons"] for row in first["decisions"]}
    assert reasons["MSFT"] == ["HOLDOUT_UNRELIABLE"]
    assert "ASSET_NOT_TRADABLE" in reasons["VTSAX"]
    assert "ASSET_DELISTED" in reasons["OLD"]


def test_quality_liquidity_and_rank_cutoff_are_explicit(tmp_path):
    db_path = tmp_path / "research.duckdb"
    universe = _universe(db_path)
    store = FocusPoolStore(db_path)
    pool = store.attempt_build(
        pool_name="daily-focus",
        universe_version=universe["version_id"],
        inputs=[
            _input("AAPL", holdout_score=0.9),
            _input("MSFT", quality=0.5),
            _input("SPY", liquidity=100),
        ],
        policy=FocusPoolPolicy(max_size=1),
        as_of=NOW,
        created_at=NOW,
    )
    reasons = {row["symbol"]: row["reasons"] for row in pool["decisions"]}
    assert reasons["MSFT"] == ["DATA_QUALITY_BELOW_MINIMUM"]
    assert reasons["SPY"] == ["LIQUIDITY_BELOW_MINIMUM"]
    assert pool["member_count"] == 1


def test_failed_rebuild_preserves_previous_valid_pool(tmp_path):
    db_path = tmp_path / "research.duckdb"
    universe = _universe(db_path)
    store = FocusPoolStore(db_path)
    valid = store.attempt_build(
        pool_name="daily-focus",
        universe_version=universe["version_id"],
        inputs=[_input("AAPL"), _input("MSFT"), _input("SPY")],
        policy=FocusPoolPolicy(max_size=2),
        as_of=NOW,
        created_at=NOW,
    )

    preserved = store.attempt_build(
        pool_name="daily-focus",
        universe_version=universe["version_id"],
        inputs=[_input("AAPL")],
        policy=FocusPoolPolicy(max_size=2),
        as_of=NOW + timedelta(days=1),
        created_at=NOW + timedelta(days=1),
        source_complete=False,
    )

    assert preserved["pool_id"] == valid["pool_id"]
    assert preserved["preserved_after_failure"]
    assert preserved["failure_code"] == "FOCUS_POOL_SOURCE_INCOMPLETE"
    assert store.latest("daily-focus")["pool_id"] == valid["pool_id"]


def test_empty_screen_result_does_not_replace_valid_pool(tmp_path):
    db_path = tmp_path / "research.duckdb"
    universe = _universe(db_path)
    store = FocusPoolStore(db_path)
    valid = store.attempt_build(
        pool_name="daily-focus",
        universe_version=universe["version_id"],
        inputs=[_input("AAPL"), _input("MSFT"), _input("SPY")],
        policy=FocusPoolPolicy(max_size=2),
        as_of=NOW,
        created_at=NOW,
    )
    preserved = store.attempt_build(
        pool_name="daily-focus",
        universe_version=universe["version_id"],
        inputs=[
            _input("AAPL", holdout=False),
            _input("MSFT", holdout=False),
            _input("SPY", holdout=False),
        ],
        policy=FocusPoolPolicy(max_size=2),
        as_of=NOW + timedelta(days=1),
        created_at=NOW + timedelta(days=1),
    )

    assert preserved["pool_id"] == valid["pool_id"]
    assert preserved["failure_code"] == "FOCUS_POOL_BELOW_MINIMUM_SIZE"
