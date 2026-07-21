# Backend Bug Reporting

The project has a local backend error platform backed by the existing
`trade.duckdb`. It adds no service, network call, or runtime dependency.

## Collection flow

Runtime tick failures, startup reconciliation failures, and broker submission
exceptions are captured by `BugReporter`. Other Python services can attach
`BugLoggingHandler` to their logger. Collection is best effort: a failure in the
reporter never recursively crashes the trading process.

Every occurrence is sanitized before storage. Known credential fields are
removed from structured context and key-like values in messages and tracebacks
are replaced with `<redacted>`.

## Database tables

- `bug_issues`: one row per stable SHA-256 fingerprint, with first/last seen,
  severity, status, and occurrence count.
- `bug_events`: sanitized occurrence history with traceback, operation, run,
  symbol, `plan_id`, and `intent_id` links.

Identical component, exception type, and message values share a fingerprint.
Repeated occurrences increment the issue count. A recurrence automatically
reopens a resolved or ignored issue.

## Inspect and triage

```powershell
python -m trader.bug_cli --db trade.duckdb list
python -m trader.bug_cli --db trade.duckdb events --limit 20
python -m trader.bug_cli --db trade.duckdb resolve <fingerprint>
python -m trader.bug_cli --db trade.duckdb ignore <fingerprint>
python -m trader.bug_cli --db trade.duckdb reopen <fingerprint>
```

Dashboard or API code can use `bug_issues_df()` and `bug_events_df()` from
`trader.monitor_data`. A practical repair loop is: inspect the newest open
fingerprint, reproduce it from the stored operation/context, add a regression
test, deploy the fix, then resolve it. If it occurs again it returns to `OPEN`.

## Limits

This is an on-host backend platform, not a hosted alerting product. It does not
send email/Slack notifications, assign owners, or upload data externally.
DuckDB retention and database backups remain operational responsibilities.