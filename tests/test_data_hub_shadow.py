from datetime import datetime, timedelta, timezone
from pathlib import Path

from trader.data_hub import DataStatus
from trader.data_hub_quality import DataHubQualityStore
from trader.data_hub_shadow import ShadowDataHubRunner
from trader.models import Bar

NOW = datetime(2026, 7, 24, 20, tzinfo=timezone.utc)


class FakeMarketFeed:
    def __init__(self, *, empty: bool = False) -> None:
        self.empty = empty
        self.calls: list[tuple[str, int]] = []

    def fetch_bars(self, symbol: str, n_bars: int = 120):
        self.calls.append((symbol, n_bars))
        if self.empty:
            return []
        return [
            Bar(
                symbol=symbol,
                timestamp=NOW - timedelta(minutes=5),
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=1_000.0,
            ),
            Bar(
                symbol=symbol,
                timestamp=NOW,
                open=100.5,
                high=102.0,
                low=100.0,
                close=101.0,
                volume=1_200.0,
            ),
        ]


def test_shadow_cycle_double_reads_and_only_persists_quality(tmp_path):
    feed = FakeMarketFeed()
    db_path = tmp_path / "quality.duckdb"
    store = DataHubQualityStore(db_path)
    runner = ShadowDataHubRunner(
        feed=feed,
        store=store,
        timeframe="5m",
        n_bars=40,
        clock=lambda: NOW + timedelta(minutes=1),
    )

    result = runner.run(["aapl", "MSFT"])

    assert result.successful
    assert result.trading_date == "2026-07-24"
    assert result.saved_observations == 2
    assert len(store.load_observations()) == 2
    assert feed.calls == [
        ("AAPL", 40),
        ("AAPL", 40),
        ("MSFT", 40),
        ("MSFT", 40),
    ]
    assert all(item.comparable for item in result.observations)
    assert all(not item.differences for item in result.observations)
    assert all(
        item.primary_source == "runtime_alpaca_feed"
        and item.shadow_source == "alpaca_market"
        for item in result.observations
    )
    assert result.quality_report["observed_trading_days"] == 1
    assert not result.quality_report["passed"]
    assert result.quality_report["execution_input_switched"] is False
    assert not _table_exists(db_path, "orders")
    assert not _table_exists(db_path, "order_intents")


def test_shadow_cycle_records_source_failures_without_execution(tmp_path):
    store = DataHubQualityStore(tmp_path / "quality.duckdb")
    result = ShadowDataHubRunner(
        feed=FakeMarketFeed(empty=True),
        store=store,
        clock=lambda: NOW,
    ).run(["AAPL"], trading_date="2026-07-24")

    observation = result.observations[0]
    assert not result.successful
    assert not observation.comparable
    assert observation.primary_status == DataStatus.FAILED
    assert observation.shadow_status == DataStatus.FAILED
    assert observation.primary_metrics.failure_count == 1
    assert observation.shadow_metrics.failure_count == 1
    assert result.quality_report["failure_rate"] == 1.0
    assert result.quality_report["execution_input_switched"] is False


def test_shadow_module_has_no_trading_execution_dependency():
    source = (
        Path(__file__).parents[1]
        / "trader"
        / "data_hub_shadow.py"
    ).read_text(encoding="utf-8")

    forbidden = (
        "broker.alpaca",
        "from .runtime",
        "from .order",
        "place_order",
        "submit_order",
    )
    assert all(value not in source for value in forbidden)


def _table_exists(path, name):
    import duckdb

    conn = duckdb.connect(str(path), read_only=True)
    try:
        return (
            conn.execute(
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_name=?",
                [name],
            ).fetchone()
            is not None
        )
    finally:
        conn.close()
