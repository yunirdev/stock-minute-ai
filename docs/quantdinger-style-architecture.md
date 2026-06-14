# QuantDinger-Style Architecture Migration

This project can move toward a QuantDinger-style architecture without throwing away the
current trading engine. The existing code already has the important building blocks:
market data adapters, strategy signals, risk checks, pending-order handling, paper
execution, portfolio state, audit logs, multi-session process management, and a Streamlit
operator console.

The main refactor is to turn those pieces into explicit layers with stable contracts.

## Current Shape

```text
Streamlit monitor
  -> starts trader.main subprocess
  -> reads DuckDB directly
  -> contains some research/backtest logic

trader.main
  -> Scheduler polling loop
  -> DataFeed -> compute_signals -> RiskEngine -> OrderManager -> Broker -> Portfolio
  -> AuditLog / DuckDB
```

This works for a local paper-trading app, but several concerns are still coupled:

- The UI owns process control and reads storage directly.
- Live trading and exploration/backtesting duplicate strategy logic.
- The scheduler is polling-oriented instead of event-oriented.
- Storage is split between market DuckDB, trade DuckDB, session DuckDB files, and JSON
  sidecars.
- Broker, risk, and strategy contracts exist, but the strategy contract is still a
  dataframe function rather than a first-class runtime interface.

## Target Shape

```text
clients/
  Streamlit or web UI
  CLI
  future API consumers

api/
  app API for sessions, backtests, orders, logs, settings

services/
  session service
  backtest service
  live runtime service
  worker queue

runtime/
  event bus
  BarEvent -> SignalEvent -> RiskEvent -> OrderIntentEvent -> FillEvent -> PortfolioEvent

strategies/
  IndicatorStrategy: dataframe in, signal columns out
  ScriptStrategy: on_init(ctx), on_bar(ctx, bar), ctx.buy/sell/close_position
  registry and parameter metadata

data/
  market data adapters: yfinance, Alpaca, future IBKR/Finnhub/etc.
  local cache/repository

execution/
  broker adapters: local paper first, future live adapters
  order state and pending-order handling

risk/
  pre-trade checks
  circuit breakers
  portfolio constraints

storage/
  repositories for market data, orders, fills, equity, sessions, audit
  DuckDB now; PostgreSQL migration later if multi-user/server deployment is needed

ai/
  optional LLM analysis, strategy drafting, confidence calibration, reflection workers
```

## Migration Plan

## Current Progress

- Strategy execution has a shared registry used by the live scheduler and the exploration
  panel.
- The scheduler publishes runtime events for bars, signals, execution requests, portfolio
  snapshots, risk decisions, order intents, and fills.
- `ExecutionPipeline` now owns the signal execution chain:
  `ExecutionRequestEvent -> RiskEvent -> OrderIntentEvent -> FillEvent`.
- `SignalRouter` owns the shared signal-to-execution-request path, including pending
  limit-order gap handling.
- `ReplayRuntime` can replay historical bars through `SignalRouter` and
  `ExecutionPipeline`, so backtest/replay can share paper/live execution semantics.
- The exploration backtest tab has an event replay entrypoint that runs selected historical
  windows through `ReplayRuntime` alongside the legacy comparison view.
- `python -m trader.replay` can run the same event replay pipeline from local cached bars
  or an explicit CSV/Parquet file.
- The in-process event bus queues nested publishes, so handlers observe events in stable
  publish order.
- Paper portfolio equity includes marked-to-market position value.

### Phase 1: Core Contracts, Low Risk

- Add a strategy registry so the scheduler stops calling raw strategy functions directly.
- Keep current dataframe strategies as `IndicatorStrategy` adapters.
- Add `ScriptStrategy` interfaces for future event-driven strategies.
- Add runtime event dataclasses even before the full event bus replaces polling.
- Fix pending-order model gaps and add focused tests.
- Make exploration/backtest call the same strategy core used by live trading.

Expected result: behavior stays the same, but strategy/runtime boundaries become visible.

### Phase 2: Event-Driven Runtime

- Introduce an in-process event bus.
- Convert scheduler internals from a single monolithic tick into event handlers.
- Emit and persist events for bars, signals, risk decisions, orders, fills, and equity.
- Keep the CLI entrypoint compatible while the internals change.

Expected result: live trading, paper trading, and replay/backtest can share one execution
path.

### Phase 3: API Boundary

- Add a small backend API for sessions, engine status, strategy catalog, backtests, and
  audit queries.
- Move Streamlit process control and direct DuckDB reads behind that API.
- Keep Streamlit as the first client; a Vue/React frontend can be added later without
  changing trading internals.

Expected result: UI becomes replaceable and the backend becomes deployable.

### Phase 4: Storage and Workers

- Introduce repository classes around DuckDB tables.
- Add optional worker queue support for backtests and longer AI tasks.
- Keep DuckDB for local-first use; add PostgreSQL only when multi-user or remote
  deployment becomes a real requirement.

Expected result: local mode stays simple, but server mode has a path.

### Phase 5: AI Layer

- Add AI analysis as a side service, not inside the execution path.
- Store AI outputs as advisory artifacts with timestamps, model metadata, and confidence.
- Never let an LLM place orders directly; it should produce research or candidate
  signals that still pass deterministic strategy/risk/execution layers.

Expected result: AI helps research and review without weakening trading safety.

## Suggested First Directory Layout

```text
trader/
  runtime/
    events.py
    bus.py
  strategies/
    base.py
    registry.py
    builtin.py
  services/
    live_runtime.py
    backtest_runtime.py
  storage/
    repositories.py
  broker/
  risk_engine.py
  order_manager.py
  portfolio.py
```

## Notes

- This should copy the architectural idea, not QuantDinger's licensed frontend or product
  code.
- DuckDB is still a good default for this project. PostgreSQL and Redis should be added
  when API/server deployment actually needs them.
- The highest-value first step is unifying strategy execution across live and backtest,
  because that removes a class of "backtest says one thing, live engine does another"
  bugs.
