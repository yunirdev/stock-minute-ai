from datetime import datetime, timedelta, timezone

import pytest

from trader.universe_registry import UniverseAsset, UniverseRegistryStore

AS_OF = datetime(2026, 7, 27, 20, 0, tzinfo=timezone.utc)


def _asset(
    symbol,
    asset_type="STOCK",
    *,
    status="ACTIVE",
    tradable=True,
):
    return UniverseAsset(
        symbol=symbol,
        asset_type=asset_type,
        exchange="nasdaq",
        status=status,
        tradable=tradable,
        source="alpaca-assets",
        as_of=AS_OF,
    )


def test_universe_version_normalizes_deduplicates_and_filters_assets(tmp_path):
    store = UniverseRegistryStore(tmp_path / "research.duckdb")
    version = store.create_version(
        universe_name="US listed",
        source_version="alpaca-2026-07-27",
        assets=[
            _asset("aapl"),
            _asset("AAPL"),
            _asset("SPY", "ETF"),
            _asset("VTSAX", "FUND", tradable=False),
            _asset("OLD", status="DELISTED"),
        ],
        as_of=AS_OF,
        created_at=AS_OF,
    )

    assert version["asset_count"] == 4
    assert version["eligible_count"] == 2
    assert [asset.symbol for asset in store.eligible_assets(version["version_id"])] == [
        "AAPL",
        "SPY",
    ]
    assert {asset.asset_type for asset in version["assets"]} == {
        "STOCK",
        "ETF",
        "FUND",
    }


def test_same_content_is_idempotent_and_changed_content_appends_version(tmp_path):
    store = UniverseRegistryStore(tmp_path / "research.duckdb")
    values = {
        "universe_name": "US listed",
        "source_version": "source-v1",
        "assets": [_asset("AAPL"), _asset("SPY", "ETF")],
        "as_of": AS_OF,
        "created_at": AS_OF,
    }
    first = store.create_version(**values)
    duplicate = store.create_version(**values)
    values["source_version"] = "source-v2"
    values["created_at"] = AS_OF + timedelta(minutes=1)
    changed = store.create_version(**values)

    assert duplicate == first
    assert changed["version_id"] != first["version_id"]
    assert store.get_version(first["version_id"]) == first
    assert store.latest("US listed") == changed


def test_conflicting_duplicate_symbol_fails_closed(tmp_path):
    store = UniverseRegistryStore(tmp_path / "research.duckdb")
    with pytest.raises(ValueError, match="DUPLICATE_SYMBOL_CONFLICT"):
        store.create_version(
            universe_name="US listed",
            source_version="source-v1",
            assets=[_asset("AAPL"), _asset("aapl", "ETF")],
            as_of=AS_OF,
            created_at=AS_OF,
        )


@pytest.mark.parametrize(
    "asset,error",
    [
        (_asset("AAPL", "CRYPTO"), "TYPE_INVALID"),
        (_asset("AAPL", status="UNKNOWN"), "STATUS_INVALID"),
        (
            UniverseAsset(
                symbol="AAPL",
                asset_type="STOCK",
                exchange="NASDAQ",
                status="ACTIVE",
                tradable=True,
                source="alpaca-assets",
                as_of=AS_OF.replace(tzinfo=None),
            ),
            "AS_OF_TZ_REQUIRED",
        ),
    ],
)
def test_invalid_asset_metadata_fails_closed(tmp_path, asset, error):
    store = UniverseRegistryStore(tmp_path / "research.duckdb")
    with pytest.raises(ValueError, match=error):
        store.create_version(
            universe_name="US listed",
            source_version="source-v1",
            assets=[asset],
            as_of=AS_OF,
            created_at=AS_OF,
        )
