"""Read-only order trace queries for operations and smoke verification."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import duckdb


def _rows_as_dicts(
    conn: duckdb.DuckDBPyConnection,
    query: str,
    params: list[Any] | None = None,
) -> list[dict[str, Any]]:
    cursor = conn.execute(query, params or [])
    columns = [item[0] for item in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def _table_names(conn: duckdb.DuckDBPyConnection) -> set[str]:
    return {
        row[0]
        for row in conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema='main'"
        ).fetchall()
    }


def order_traces(
    db_path: str | Path,
    *,
    plan_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Return plan → risk → intent/idempotency → fill traces without mutation."""
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(path)

    conn = duckdb.connect(str(path), read_only=True)
    try:
        tables = _table_names(conn)
        required = {"trade_plans", "order_intents", "plan_risk_events"}
        missing = sorted(required - tables)
        if missing:
            raise ValueError(
                "AUDIT_SCHEMA_MISSING:" + ",".join(missing)
            )

        plans = _rows_as_dicts(
            conn,
            "SELECT * FROM trade_plans ORDER BY created_at, plan_id",
        )
        intents = _rows_as_dicts(
            conn,
            "SELECT * FROM order_intents ORDER BY updated_at, intent_id",
        )
        risks = _rows_as_dicts(
            conn,
            "SELECT * FROM plan_risk_events ORDER BY ts, plan_id",
        )
        fills = (
            _rows_as_dicts(
                conn,
                "SELECT * FROM fills ORDER BY fill_time, intent_id",
            )
            if "fills" in tables
            else []
        )
    finally:
        conn.close()

    if plan_ids is not None:
        plans = [row for row in plans if row["plan_id"] in plan_ids]
        intents = [row for row in intents if row["plan_id"] in plan_ids]
        risks = [row for row in risks if row["plan_id"] in plan_ids]

    plan_by_id = {row["plan_id"]: row for row in plans}
    risks_by_plan: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in risks:
        risks_by_plan[row["plan_id"]].append(row)
    fills_by_intent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in fills:
        fills_by_intent[row["intent_id"]].append(row)
    intents_by_plan: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in intents:
        intents_by_plan[row["plan_id"]].append(row)

    all_plan_ids = set(plan_by_id) | set(intents_by_plan) | set(risks_by_plan)
    traces: list[dict[str, Any]] = []
    for current_plan_id in sorted(all_plan_ids):
        plan_intents = intents_by_plan.get(current_plan_id) or [None]
        for intent in plan_intents:
            traces.append(
                {
                    "plan_id": current_plan_id,
                    "decision_id": (
                        (intent or {}).get("decision_id")
                        or (risks_by_plan.get(current_plan_id) or [{}])[0].get(
                            "decision_id",
                            "",
                        )
                    ),
                    "plan": plan_by_id.get(current_plan_id),
                    "risk_events": risks_by_plan.get(current_plan_id, []),
                    "order": intent,
                    "fills": (
                        fills_by_intent.get(intent["intent_id"], [])
                        if intent
                        else []
                    ),
                }
            )
    return traces


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only plan/risk/order/fill audit trace query."
    )
    parser.add_argument("--db", required=True, help="DuckDB path to inspect")
    parser.add_argument(
        "--plan-id",
        action="append",
        dest="plan_ids",
        help="Filter to one plan ID; may be repeated",
    )
    parser.add_argument("--compact", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    traces = order_traces(
        args.db,
        plan_ids=set(args.plan_ids) if args.plan_ids else None,
    )
    print(
        json.dumps(
            traces,
            ensure_ascii=True,
            default=str,
            indent=None if args.compact else 2,
        )
    )


if __name__ == "__main__":
    main()
