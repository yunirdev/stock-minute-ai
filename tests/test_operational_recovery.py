from datetime import datetime, timezone

import duckdb
import pytest

from trader.operational_recovery import RecoveryManager, run_fault_drill

NOW = datetime(2026, 7, 27, 20, 0, tzinfo=timezone.utc)


class APIConnectionError(ConnectionError):
    pass


def test_database_backup_restore_uses_new_copy_and_checksums(tmp_path):
    source = tmp_path / "trade.duckdb"
    connection = duckdb.connect(str(source))
    connection.execute("CREATE TABLE audit(id INTEGER, value TEXT)")
    connection.execute("INSERT INTO audit VALUES (1, 'kept')")
    connection.close()

    manager = RecoveryManager()
    manifest = manager.create_backup(
        [source],
        destination=tmp_path / "backup",
        created_at=NOW,
    )
    restored = manager.restore_to_new_directory(
        manifest["manifest_path"],
        destination=tmp_path / "restored",
    )

    assert source.exists()
    assert len(restored) == 1
    restored_db = duckdb.connect(str(restored[0]), read_only=True)
    assert restored_db.execute("SELECT * FROM audit").fetchall() == [(1, "kept")]
    restored_db.close()
    with pytest.raises(FileExistsError):
        manager.restore_to_new_directory(
            manifest["manifest_path"],
            destination=tmp_path / "restored",
        )


@pytest.mark.parametrize(
    "kind,error",
    [
        ("API", APIConnectionError("API_CONNECTION_FAILED")),
        ("DATABASE", duckdb.Error("DATABASE_UNAVAILABLE")),
        ("RUNTIME", RuntimeError("RUNTIME_HEARTBEAT_STALE")),
    ],
)
def test_api_database_and_runtime_fault_drills_are_classified(kind, error):
    def operation():
        raise error

    result = run_fault_drill(
        kind,
        operation,
        expected_exception=type(error),
    )
    assert result["passed"]
    assert result["recovered"]
    assert result["classified_as"] == kind
