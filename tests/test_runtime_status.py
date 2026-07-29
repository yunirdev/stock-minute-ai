from datetime import datetime, timezone

from trader.ai.safety import AIScoreSnapshot
from trader.models import Bar, Position, Side, TradePlan
from trader.runtime_status import (
    build_runtime_status,
    read_runtime_status,
    write_runtime_status,
)


def test_runtime_status_tracks_candidate_distance_and_round_trips(tmp_path):
    now = datetime(2026, 7, 27, 15, 0, tzinfo=timezone.utc)
    bar = Bar("AAPL", now, 99, 102, 98, 101, 1000)
    plan = TradePlan(
        "p1",
        "AAPL",
        Side.BUY,
        "OPEN",
        100,
        95,
        110,
        qty=5,
    )
    snapshot = AIScoreSnapshot(
        "AAPL",
        82,
        now,
        run_id="research-1",
        source="daily_research",
    )
    payload = build_runtime_status(
        now=now,
        tick_count=3,
        session="open",
        equity=10_000,
        reconciliation_blocked=False,
        kill_switch=False,
        bars={"AAPL": bar},
        positions={"AAPL": Position("AAPL", 0, 0)},
        plans={"AAPL": plan},
        research_snapshots={"AAPL": snapshot},
    )
    assert payload["candidates"][0]["state"] == "READY"
    assert payload["candidates"][0]["distance_to_entry_pct"] == 1.0

    path = tmp_path / "runtime_status.json"
    write_runtime_status(payload, path)
    assert read_runtime_status(path) == payload
