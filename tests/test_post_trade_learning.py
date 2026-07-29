from datetime import datetime, timezone

import duckdb

from trader.post_trade_learning import PostTradeLearningCoordinator

NOW = datetime(2026, 7, 27, 20, 0, tzinfo=timezone.utc)


def _schema(db_path):
    connection = duckdb.connect(str(db_path))
    connection.execute(
        """
        CREATE TABLE position_plan_versions(
            position_plan_id TEXT, version INTEGER,
            source_trade_plan_id TEXT
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE decision_plan_links(
            decision_id TEXT, plan_id TEXT
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE strategy_decisions(
            decision_id TEXT, strategy TEXT, strategy_version TEXT,
            data_version TEXT
        )
        """
    )
    connection.execute("INSERT INTO position_plan_versions VALUES ('episode-1',1,'plan-1')")
    connection.execute("INSERT INTO decision_plan_links VALUES ('decision-1','plan-1')")
    connection.execute(
        "INSERT INTO strategy_decisions VALUES ('decision-1','ema','v1','data-v1')"
    )
    connection.close()


def test_closed_episode_creates_one_review_and_candidate(tmp_path):
    db_path = tmp_path / "trade.duckdb"
    _schema(db_path)
    coordinator = PostTradeLearningCoordinator(db_path)
    episode = {
        "episode_id": "episode-1",
        "snapshot_id": "snapshot-1",
        "status": "CLOSED",
        "symbol": "AAPL",
        "first_fill_at": NOW,
        "last_fill_at": NOW,
        "attribution": {"fill_count": 2},
        "adverse_slippage": 1.5,
        "adjustment_count": 0,
        "realized_pnl": -10.0,
        "open_quantity": 0.0,
    }
    first = coordinator.process(episode, created_at=NOW)
    second = coordinator.process(episode, created_at=NOW)
    assert first == second
    assert first["review"]["outcome"] == "SUCCESS"
    assert first["review"]["result"]["realized_pnl"] == -10.0
    assert first["candidate"]["strategy_name"] == "ema"
    assert (
        first["candidate"]["parameters"]["proposal"]
        == "NO_PARAMETER_CHANGE_CONTROL"
    )


def test_open_episode_does_not_create_review(tmp_path):
    db_path = tmp_path / "trade.duckdb"
    _schema(db_path)
    assert PostTradeLearningCoordinator(db_path).process(
        {
            "episode_id": "episode-1",
            "snapshot_id": "snapshot-1",
            "status": "OPEN",
        },
        created_at=NOW,
    ) is None
