# Data Hub Shadow Runbook

`trader.data_hub_shadow` is a read-only integration probe. It independently
reads the configured Alpaca market-data endpoint through the current
`AlpacaDataFeed` path and the new `DataHub` path, compares canonical envelopes,
and writes only quality observations and reports to `ai_states.duckdb`.

It does not import Runtime, broker execution, or order code. Its reports always
set `execution_input_switched=false`; a passing report does not automatically
change the production execution input.

## Manual cycle

```powershell
.venv\Scripts\python.exe -m trader.data_hub_shadow `
  --symbols AAPL,MSFT `
  --timeframe 5m `
  --bars 120 `
  --db ai_states.duckdb
```

The command exits successfully only when both reads are comparable and contain
no unclassified critical difference. The 20-trading-day quality report remains
failed until the full observation window is present; that is expected during
the accumulation period.

Use `--trading-date YYYY-MM-DD` only for an explicitly known session. Otherwise
the runner derives the observation date from the newest successful market bar,
which prevents weekend launches from creating a false trading day.

## Accelerated delivery evidence

Historical correctness can be verified immediately without pretending that
historical rows are live latency or quota observations:

```powershell
.venv\Scripts\python.exe -m trader.data_hub_replay `
  --symbols AAPL,MSFT `
  --days 20 `
  --tolerance-bps 1 `
  --db ai_states.duckdb
```

The replay compares the local research cache against a fresh Alpaca history
window for every symbol and trading date. It stores rows in
`data_hub_historical_replays`, separate from live
`data_hub_double_reads`.

`accelerated_shadow_delivery_ready=true` means the read-only integration can be
delivered and subsequent migration tasks can continue. It does not mean the
execution input may switch:

- `evidence_scope=HISTORICAL_DATA_CORRECTNESS_ONLY`
- `can_replace_live_observation_window=false`
- `execution_cutover_ready=false` until the live gate passes
- `execution_input_switched=false`

This separation removes the 20-calendar-day delay from code delivery while
preserving real operational evidence for any future execution-path cutover.

## Acceptance boundary

Before any future execution-input change:

1. Accumulate at least 20 distinct trading days.
2. Keep unclassified critical differences at zero.
3. Meet the failure-rate, latency, and applicable quota gates.
4. Review and approve the report manually as an architecture change.
5. Add a separate migration task and regression suite.

No step in this runbook authorizes live trading or bypasses Runtime risk,
idempotency, reconciliation, or Paper-only constraints.
