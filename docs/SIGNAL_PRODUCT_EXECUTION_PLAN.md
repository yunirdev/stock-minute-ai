# Signal Product Execution Plan

## Objective

Turn the existing Paper-only runtime into a continuous research-signal product:

    market scan -> strategy signal -> AI evidence -> PaperDecision
      -> ATR/risk plan -> SignalReport -> Discord
      -> Paper fill -> position update -> exit -> performance attribution

Clients remain self-directed. The application never connects to or trades customer
brokerage accounts.

## Confirmed daily architecture

    post-close/premarket scheduler
      -> deterministic cached-data screen
      -> reliable holdout-statistics enrichment
      -> top 10 shortlist
      -> TradingAgents deep research for top 5
      -> immutable daily_research_runs/items in ai_states.duckdb
      -> Runtime consumes current trading-date report
      -> current strategy vote + PaperDecision + ATR/risk
      -> signal_events + logs/runtime_status.json
      -> NiceGUI live overview

The production Runtime no longer starts AgentManager every 15 minutes. The legacy
manager remains available from the research cockpit only. TradingAgents is loaded
lazily from the installed package or TRADINGAGENTS_PROJECT_DIR. A missing module,
model, or provider is a persisted FAILED item/run and never becomes a neutral or
synthetic score.

The automatic schedule makes at most one attempt for a target trading date. A
manual forced retry remains available from NiceGUI and the daily_research CLI.
Daily evidence and intraday market-data freshness have separate clocks.

## Implementation status

- Complete: AI advisory provenance, same-run aggregation, fallback marking,
  contributor/coverage validation, and worker timeout recovery.
- Complete: ETF daily-data preparation and explicit blocking when verification fails.
- Complete: deterministic daily screen, holdout enrichment, TradingAgents adapter,
  immutable run/item/publication schema, one-attempt scheduler, and CLI.
- Complete: Runtime daily-report gate, separate research TTL, signal lifecycle store,
  fill/exit linking, and lock-free runtime status sidecar.
- Complete: NiceGUI batch/run/candidate monitoring and manual forced batch action.
- Implemented but not activated: Discord templates and idempotent publication store.
  New automatic external sends require an explicit user authorization in the
  execution environment.
- External setup pending: TradingAgents is not installed in the current virtual
  environment. All local logic and fake-adapter integration tests remain runnable.

## Current delivery scope

### Sprint 1 - AI runtime reliability

- Read mixed legacy/current advisory rows without losing all symbols.
- Select contributors from one current run only.
- Require configurable contributor count, weight coverage, and real LLM evidence.
- Mark fallback output so it cannot masquerade as model evidence.
- Recover the advisory worker after a timed-out cycle.

Acceptance:

- Current valid rows remain readable when legacy rows are incomplete.
- Default production decisions require at least three contributors, 50% configured
  weight coverage, and one non-fallback LLM contributor.
- Quant-only operation requires the existing explicit configuration.
- A timed-out cycle cannot permanently block future cycles.

### Sprint 2 - Canonical signal lifecycle

- Add SignalReport as the only customer-facing signal source.
- Use deterministic strategy/ATR/risk values for direction, entry, stop, target,
  model weight, and expiry.
- Persist immutable state transitions and active-signal state in DuckDB.
- Deduplicate repeated READY reports across runtime ticks and restarts.
- Support READY, ENTERED, HOLD, EXIT, INVALIDATED, and CLOSED.

Acceptance:

- Every published report has an ID, version, market-data timestamp, expiry,
  entry, stop, target, model weight, reasons, risks, and evidence references.
- Repeated ticks do not publish duplicate READY messages.
- Every terminal report links to the originating decision and plan.

### Sprint 3 - Runtime and Discord

- Publish READY after AI and deterministic risk approval.
- Publish ENTERED/CLOSED from confirmed Alpaca Paper fills.
- Publish EXIT when the position monitor creates a close plan.
- Invalidate expired unfilled signals.
- Persist Discord attempts and delivery status.
- Do not count console fallback as successful Discord delivery.

Acceptance:

- The production Runtime, not the NiceGUI process, owns customer signal events.
- Discord messages are generated only from SignalReport.
- Failed Discord delivery remains visible and retryable.

### Sprint 4 - Effect measurement

- Calculate completed-signal return, PnL, win rate, payoff, and holding time.
- Attribute outcomes by strategy and AI contributor.
- Add the signal summary to the daily review.
- Expose active/recent signals and summary data to NiceGUI read models.

Acceptance:

- A completed Paper position produces one closed signal result.
- Daily review reports open, entered, closed, invalidated, win-rate, and return data.
- Strategy and contributor summaries can be read without parsing Discord messages.

## Deferred

- Customer brokerage execution.
- Billing, subscriptions, and multi-tenant accounts.
- Personalized position sizing.
- Full commercial compliance workflow.
- Large-scale cloud deployment and enterprise disaster recovery.

## Verification

- Targeted tests for AI migration, coverage, timeout recovery, lifecycle,
  deduplication, fill transitions, delivery failure, and performance.
- Full pytest, Ruff, and compileall.
- Paper-only and LMT-only production invariants remain unchanged.
