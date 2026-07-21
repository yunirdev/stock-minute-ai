# Architecture Health Detection

The health layer detects failures that ordinary application logging cannot
reliably observe.

## Detection paths

- The heartbeat table is the canonical DuckDB heartbeat source. The JSON
  heartbeat sidecar remains the lock-free cross-process source.
- Runtime, startup, reconciliation, broker, allocator, and selection failures
  are persisted by BugReporter.
- Run python -m trader.health_check outside Runtime to detect a missing or stale
  heartbeat, unreadable or incomplete database, and an unreachable UI.
- NiceGUI exposes /healthz.
- The browser reports JavaScript errors, unhandled promise rejections, an empty
  root, horizontal overflow, and multiple visible elements outside the viewport
  to /api/ui-health/report. Reports are sanitized and deduplicated in the bug
  tables.

## Operations

Run locally:

    python -m trader.health_check --ui-url http://127.0.0.1:8080

A zero exit code means no critical finding. Exit code 1 means at least one
critical finding. By default findings are written to the bug database. Use
--no-report for a read-only probe.

For continuous monitoring, schedule the command outside the trading process
(for example, Windows Task Scheduler every minute). This is essential: an
in-process watchdog cannot report after its own process dies.

Inspect and resolve findings with:

    python -m trader.bug_cli --db trade.duckdb list
    python -m trader.bug_cli --db trade.duckdb resolve <fingerprint>

## Limits

The browser probe catches common structural breakage but is not a pixel-perfect
visual regression suite. It does not compare approved screenshots, typography,
or subjective design quality. External alert delivery is still optional and is
not enabled by this change.