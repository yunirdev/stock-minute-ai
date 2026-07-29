"""Small CLI for inspecting and triaging locally collected backend bugs."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from .bug_reporting import BugReporter


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description="Inspect the backend bug database")
    parser.add_argument("--db", default="trade.duckdb")
    commands = parser.add_subparsers(dest="command", required=True)
    list_parser = commands.add_parser("list")
    list_parser.add_argument("--limit", type=int, default=50)
    events_parser = commands.add_parser("events")
    events_parser.add_argument("--limit", type=int, default=50)
    for command in ("resolve", "ignore", "reopen"):
        item = commands.add_parser(command)
        item.add_argument("fingerprint")
    args = parser.parse_args()
    reporter = BugReporter(args.db, "cli")

    if args.command == "list":
        payload = [asdict(issue) for issue in reporter.unresolved(args.limit)]
    elif args.command == "events":
        payload = reporter.recent_events(args.limit)
    else:
        changed = getattr(reporter, args.command)(args.fingerprint)
        statuses = {
            "resolve": "RESOLVED",
            "ignore": "IGNORED",
            "reopen": "OPEN",
        }
        payload = {
            "fingerprint": args.fingerprint,
            "updated": changed,
            "status": statuses[args.command],
        }
    print(json.dumps(payload, default=str, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())