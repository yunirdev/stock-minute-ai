from datetime import datetime
from zoneinfo import ZoneInfo

import duckdb

from trader.production_operations import ProductionOperationsCoordinator


def _database(path):
    connection = duckdb.connect(str(path))
    try:
        connection.execute("CREATE TABLE evidence(value INTEGER)")
        connection.execute("INSERT INTO evidence VALUES (1)")
        connection.commit()
    finally:
        connection.close()


def test_daily_backup_is_verified_and_idempotent(tmp_path):
    trade_db = tmp_path / "trade.duckdb"
    ai_db = tmp_path / "ai_states.duckdb"
    _database(trade_db)
    _database(ai_db)
    coordinator = ProductionOperationsCoordinator(
        trade_db,
        ai_db_path=ai_db,
        backup_root=tmp_path / "backups",
    )
    now = datetime(2026, 7, 27, 21, 0, tzinfo=ZoneInfo("America/New_York"))

    first = coordinator.tick(now=now)
    second = coordinator.tick(now=now)

    assert first["status"] == "SUCCESS"
    assert first["source_count"] == 2
    assert first["attempt_count"] == 1
    assert second["manifest_id"] == first["manifest_id"]
    assert (tmp_path / "backups" / "2026-07-27" / "attempt-01").is_dir()


def test_daily_backup_waits_until_after_cutoff(tmp_path):
    trade_db = tmp_path / "trade.duckdb"
    _database(trade_db)
    coordinator = ProductionOperationsCoordinator(
        trade_db,
        backup_root=tmp_path / "backups",
    )

    result = coordinator.tick(
        now=datetime(
            2026,
            7,
            27,
            19,
            59,
            tzinfo=ZoneInfo("America/New_York"),
        )
    )

    assert result["status"] == "NOT_DUE"
    assert not (tmp_path / "backups").exists()
