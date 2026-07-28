# AGENTS.md - stock-minute-ai engineering baseline

Read this file before any work. It is the authoritative production baseline.
Use [docs/CODEX_COLLABORATION.md](docs/CODEX_COLLABORATION.md) for task scoping,
handoff, and the user/Codex collaboration contract. Update this file only when a
material production boundary, safety rule, entrypoint, or baseline changes.

## Product

- AI-assisted Alpaca Paper trading. No per-trade user approval. No automatic live trading.
- Current delivery goal: one autonomous, replayable, measurable loop from data
  acquisition through analysis, plan, Paper execution, risk, review, and
  versioned strategy-candidate iteration; all NiceGUI actions must have complete
  success/empty/error/busy outputs.
- NiceGUI is for monitoring and research. Runtime is the only production trading loop.
- Agents create analysis only. Runtime is the only component that may submit orders.
- TradingAgents runs as one immutable daily research batch; it is never a
  per-tick execution component.
- Without --auto-trade, plans are DRY_RUN. With it, qualifying plans submit Paper LMT orders.

## Product reference

- Long-term product operating model: [docs/PRODUCT_OPERATING_MODEL.md](docs/PRODUCT_OPERATING_MODEL.md).
- Target architecture, data ownership, and implementation slices: [docs/TARGET_ARCHITECTURE.md](docs/TARGET_ARCHITECTURE.md).
- Executable migration roadmap and release gates: [docs/PROJECT_MIGRATION_PLAN.md](docs/PROJECT_MIGRATION_PLAN.md).
- Migration task board and single-command continuation workflow: [docs/MIGRATION_TASK_BOARD.md](docs/MIGRATION_TASK_BOARD.md).
- Accepted architecture decisions: [docs/decisions](docs/decisions).

## Single production path

    Daily research -> deterministic screen + holdout evidence
      -> TradingAgents on the deep shortlist -> ai_states.duckdb
    Runtime -> market data + current strategy votes + frozen daily research
      -> PaperDecision: holdout statistics + market regime + current strategy signal + AI evidence
      -> ATR TradePlan -> allocation -> deterministic risk
      -> DRY_RUN or durable/idempotent Alpaca Paper LMT
      -> order polling -> portfolio, immutable PositionPlan, and audit persistence

Entrypoints:

- python -m trader.monitor_nice
- python -m trader.main [--auto-trade]
- python -m trader.daily_research --symbols AAPL,MSFT,NVDA
- python -m trader.strategy_statistics --symbols AAPL,MSFT --timeframe 5m

## Non-negotiable safety

1. Automatic submission requires broker_type=alpaca_paper and auto_trade_paper=True.
2. Automatic execution creates only LMT orders.
3. A plan needs a current BUY/SELL strategy signal and reliable matching holdout statistics.
4. AI evidence must come from the current trading date's immutable research run,
   be trusted, and meet the configured score. Quant-only operation must be explicit.
5. Risk, kill switch, reconciliation, idempotency, and durable order records cannot be bypassed.
6. Secrets never enter logs, databases, Git, or test snapshots.
7. Agent modules never import broker or order execution.

## Active code

- main.py and runtime.py: CLI, background agent refresh, trading lifecycle
- paper_decision.py and strategy_statistics.py: strategy selection and holdout statistics
- models.py: shared data models and AgentContext
- selection.py, plan.py, allocator.py, risk_engine.py: decision pipeline
- broker/alpaca.py: only execution adapter
- order_store.py, portfolio.py, audit.py: durable state and audit
- daily_research.py, research_screening.py, and daily_runtime_support.py: daily analysis system
- position_plans.py: immutable filled-position baseline and version-chain store
- invalidation_events.py: authoritative invalidation fact validation and store
- position_adjustments.py: deterministic event-to-plan/order adjustments
- position_quality.py: daily broker/local/plan consistency evidence and gate
- execution_pipeline.py: CandidatePlan/FinalTradePlan/OrderIntent state machine
- trade_episodes.py: immutable fill/slippage/PnL episode attribution
- episode_reviews.py: frozen layered reviews and stable error taxonomy
- strategy_candidates.py and strategy_promotion.py: immutable candidate,
  champion/challenger evidence, release, rejection, and rollback versions
- universe_registry.py, focus_pool.py, research_budget.py, and
  universe_research_quality.py: versioned market scope, focus selection,
  capacity-controlled research work, and daily quality gates
- ai/manager.py and ai/agents: legacy/manual research tools, not the production tick loop
- signal_reports.py and runtime_status.py: customer-signal audit and live status sidecar
- data_hub_shadow.py: read-only real-source double-read quality runner
- data_hub_replay.py: historical correctness evidence, separate from live quality
- monitor_nice.py and monitor_data.py: UI and read models
- operations_observability.py and discord_delivery.py: 31-action UI contracts,
  explainability, authorization, delivery, and audit evidence
- operational_recovery.py and closed_loop_delivery.py: verified database
  recovery and frozen complete-loop delivery evidence
- paper_maturity.py, paper_resilience.py, and paper_migration_signoff.py:
  scheduled REAL/SYNTHETIC maturity, fixed fault drills, and two-level sign-off
- strategies, strategy_core.py, factors, backtest: research
- watchdog.py and kill_switch.py: runtime safety

Do not restore Scheduler, in-memory PaperBroker, separate yfinance feed, OrchestratorAgent, Streamlit preferences, per-trade approval, Protocol shells, PendingOrder, BrokerAdapter, or PaperDecision enabled/shadow dual paths.

## Data rules

Commit source, tests, README, this file, configuration templates, and active technical documentation.

Never commit caches, bytecode, test/lint caches, .nicegui, .tmp, egg-info, generated conf snapshots, strategy_statistics.json, downloaded external projects, secrets, databases, or logs.

Do not delete trade.duckdb, ai_states.duckdb, logs, or conf/ui_settings.json. They are user records or preferences.

## Working protocol

Before work: read this file, check git status, and trace only the production
callers within the agreed task slice. For migration work, read
`docs/MIGRATION_TASK_BOARD.md`, select the first unblocked task in listed order,
and execute only that task.

After work: remove new cache files; run proportionate verification; inspect
diff/status for secrets and generated data; update README if behavior changes.
For migration work, mark a task DONE and record evidence only after every listed
acceptance check passes; never mark it complete from inspection alone.
Record material architecture decisions in docs rather than extending the
historical list below. Do not perform a repository-wide audit, refactor, or
cleanup unless the task explicitly asks for it.

## Collaboration settings

- Work in the current task by default; do not spawn sub-agents unless the user,
  this file, or an applicable skill explicitly requests delegation or parallel work.
- In default collaboration mode, make safe, reasonable assumptions and continue;
  ask the user only when a missing choice would materially change the result or risk.
- Keep user-owned worktree changes intact and report sandbox or permission failures
  separately from repository command failures.

## Verification

    .venv\Scripts\python.exe -m pytest tests -q
    .venv\Scripts\python.exe -m ruff check trader tests
    .venv\Scripts\python.exe -m compileall -q trader tests

## Current baseline

- Date: 2026-07-27
- Baseline: Isolated TradingAgents v0.3.1 with local Ollama 32K models.
- Verification: 396 tests passed; full Ruff and compileall passed; MSFT local-model
  end-to-end analysis completed through the subprocess adapter.
- Goal achieved: production Runtime consumes one immutable daily shortlist instead
  of rerunning the full LLM agent set every 15 minutes.

## Recent changes

- Completed pre-trial release cleanup: NiceGUI Paper auto-trade authority is
  session-scoped and always defaults off, stale Runtime sidecars no longer
  render as healthy, interrupted daily research is recovered fail-closed, and
  the simplified five-entry platform navigation passed real browser validation.
- Connected the remaining evidence architecture to production callers: Runtime
  now freezes natural REAL maturity evidence, produces verified daily trade/AI
  DuckDB backups, and turns closed trade episodes into frozen reviews and
  conservative strategy candidates. All 31 NiceGUI actions use durable
  BUSY/SUCCESS/EMPTY/ERROR auditing, and every Discord send uses the central
  authorization, redaction, deduplication, and delivery-audit gateway.
- Added the complete I-stage architecture: immutable scheduled Paper sessions
  prevent skipped failure days; 60-session REAL/SYNTHETIC maturity gates track
  report completeness, duplicate orders, plan rewrites, state differences, and
  unresolved failures. Six fixed resilience drills fail on any unexpected
  submit. ARCHITECTURE_READY accepts isolated evidence, while FINAL_REAL_READY
  requires 60 REAL sessions plus REAL resilience evidence and never authorizes
  live trading.
- Added H-stage observability, notification, recovery, and frozen delivery
  contracts. All 31 NiceGUI actions support SUCCESS/EMPTY/ERROR/BUSY audit
  states; Discord external sends fail closed without authorization; DuckDB
  backups are hash/read-only verified; complete Paper evidence binds snapshot
  through strategy candidate. The activity page renders the latest order's
  source/research/plan/risk/intent/fill chain, with read-only action-manifest
  and order-explanation APIs. NiceGUI web mode now defaults to localhost.
  User-browser evidence confirmed the complete EMPTY explanation state; H05
  and the current Alpaca Paper closed-loop H06 target are signed off.
- Added append-only stock/ETF/fund universe versions, deterministic focus pools,
  durable quota/batch/retry/timeout research work, and REAL/SYNTHETIC-separated
  daily coverage/cost/timeliness evidence. Failed focus rebuilds preserve the
  prior pool, completed work is restart-safe, and a 20-session synthetic
  positive/negative gate cannot substitute for natural REAL observations.
- Added frozen holdout, non-overlapping historical replay, and Paper
  champion/challenger comparisons with explicit fee, slippage, return, Sharpe,
  drawdown, and minimum-sample gates. Append-only release events promote or
  reject with stable evidence/reason references, preserve the prior champion
  for restart-safe rollback, reject stale baselines, and never edit Runtime
  strategy configuration.
- Added an append-only strategy candidate/version store. Every candidate must
  reference a frozen EpisodeReview and records its production baseline,
  dataset, code, parameter hash, and non-overlapping training/holdout boundary.
  Duplicate generation is idempotent, parameter changes append linked versions,
  restart recovery is tested, and the store cannot promote or alter production.
- Added immutable layered episode reviews with stable SUCCESS, RISK_REJECTED,
  NO_FILL, DATA_FAILURE, and BROKER_FAILURE taxonomy. Facts, decisions,
  execution, and results remain separately replayable; content hashes make
  repeats idempotent, and a losing successful trade is not automatically
  labeled a strategy invalidation.
- PositionPlan fills now synchronize an immutable trade-episode attribution
  snapshot after normal polling and restart reconciliation. Partial entries,
  reductions, closes, cross-day duration, realized PnL, adverse limit slippage,
  invalidation events, and adjustments share one episode ID; content hashes
  make restart synchronization idempotent and broker fills remain immutable.
- Removed the old `_execute_plan` compatibility entry. The internal submit
  function now requires an OrderIntent produced by the state machine, and all
  Paper smoke/boundary tests use the pipeline. Runtime fallback identity is
  stable, pre-migration SENDING/UNKNOWN keys remain protected, rejected risk
  becomes an audited terminal candidate, and stale TradePlan execution-state
  labels were removed.
- All Runtime production callers now enter the execution state machine first:
  new positions, monitored exits, invalidation adjustments, and restart recovery.
  The prepared OrderIntent retains final/risk/evidence references and reaches one
  internal broker submission function. Static tests prove there is no direct
  Runtime bypass and no second `place_order` caller.
- Added the durable CandidatePlan -> FinalTradePlan -> OrderIntent state machine.
  Stable IDs and references bind decision, strategy/data/evidence versions, risk
  check/config version, validity, direction, and approved quantity. Expired,
  evidence-free, direction-invalid, risk-rejected, and illegal transitions fail
  closed. OrderIntent storage gained backward-compatible pipeline references;
  Runtime caller migration remains F02.
- Runtime now records REAL daily broker/local/PositionPlan quality evidence and
  a 30-session gate for quantity mismatches, silent version rewrites, and
  duplicate adjustments; SYNTHETIC gate evidence is physically separated. A
  broker-authoritative portfolio baseline preserves legacy fills while replaying
  only later fills. The real database retained seven retired `PAPER-*` QQQ fills,
  then successfully reconciled empty Alpaca Paper/local/plan state at 1/30 REAL
  days. Alpaca position/cash failures can no longer masquerade as empty facts.
- Startup reconciliation now restores planned adjustments, open adjustment
  orders, PositionPlan cumulative-fill cursors, partial exits, and completed
  exits. A missed first plan can be rebuilt from its audited TradePlan; repeated
  restarts are idempotent. Plan/local/broker quantity mismatches fail closed,
  and UNKNOWN orders remain unresolved without resubmission or guessed state.
- Validated invalidation events now produce one transactional PositionAdjustment
  and PositionPlan version. EXIT and REDUCE adjustments use Runtime's existing
  idempotent Alpaca Paper LMT path; TIGHTEN_STOP changes only the versioned stop.
  Long and short stops can only tighten, duplicate events cannot repeat a
  version or submission, and the compatible E04 table was added to trade.duckdb.
- Added source-backed InvalidationEvent contracts for price stops, broker facts,
  corporate actions, trading restrictions, and deterministic strategy failures.
  Events bind to the current PositionPlan version and require canonical facts,
  evidence, valid time order/freshness, deterministic IDs, and type-specific
  authoritative sources. Duplicate source facts are idempotent, conflicting
  rewrites and free-form model text fail closed, and the compatible E03 table
  was added to trade.duckdb without changing plans or orders.
- Runtime now projects confirmed cumulative fills into immutable PositionPlan
  version chains. Initial, partial, duplicate, reduction, and closing fills are
  durable and idempotent; each plan advance and its fill cursor commit in one
  transaction. The Paper smoke path covers the integration, and the compatible
  E02 fill-event table was added to trade.duckdb without rewriting old records.
- Daily research now injects one clock through deterministic screening and
  snapshot capture, preventing millisecond `SOURCE_AS_OF_AFTER_FETCH` failures.
  Runs left RUNNING beyond the worker timeout atomically become FAILED together
  with PENDING/RUNNING items, and every failed run/item/link gets a stable,
  non-empty, secret-safe diagnostic code. The actual legacy interrupted batch
  was recovered without deleting history.
- Both configured local 32K models now pass real Ollama Chat Completions and
  TradingAgents LangChain-client calls. The external runtime pins pandas 2.3.3
  because Windows Application Control rejected pandas 3.0.5's native period
  module.
- The subprocess contract now carries cache, result, and memory paths; the
  worker creates them and redirects yfinance's SQLite caches into the writable
  application `.tmp` tree before the first request. Sixteen focused adapter,
  worker, and immutable-contract tests pass. The resource-intensive full graph
  verification was stopped at the user's request after model inference began.
- A persisted TradingAgents batch-quality report now quantifies observation
  coverage, duplicate runs, success rates, latency, invocation-contract and
  output-snapshot coverage. The real database currently fails this gate with
  four APIConnectionError runs on one date; D04 is recorded as externally
  blocked rather than falsely passed.
- Runtime AI evidence now requires one unambiguous successful run for the
  current date, freshness, replayable input/output snapshots, matching
  data/model/invocation contracts, and all TradingAgents sources at OK quality.
  Legacy, stale, duplicated, mismatched, or degraded evidence fails closed.
- TradingAgents configured market/fundamental/news/sentiment vendor chains now
  become validated source manifests in a separate immutable, replayable output
  snapshot linked to each new research item. Missing or malformed provenance
  fails closed; legacy rows remain readable without fabricated provenance.
- TradingAgents subprocess calls now carry and echo a versioned immutable
  invocation contract linking run, snapshot, snapshot content hash, data
  version, model/config version, symbol, date, and cutoff. New research items
  persist those references; mismatched, stale, malformed, timed-out, or crashed
  worker results fail closed. Legacy item rows remain readable with empty links.
- The current delivery target and acceptance contract now cover the complete
  Paper loop, quantitative/replay evidence, controlled strategy-candidate
  iteration, and all 31 rendered NiceGUI button actions. D01 remains the next
  implementation task; A–C alone are not considered complete-loop delivery.
- Accelerated Data Hub evidence now replays 20 historical sessions from the
  local research cache against fresh Alpaca daily bars in dedicated tables.
  The first AAPL/MSFT run produced 40 matching OHLCV comparisons at 1 bps
  tolerance. It can unblock shadow-code delivery and D01, but explicitly cannot
  replace live observations or authorize an execution-input switch.
- A read-only ShadowDataHubRunner now independently reads AAPL/MSFT through
  the legacy Alpaca feed and Data Hub, persists only quality observations,
  derives weekend runs from the newest market session, and never switches
  Runtime execution inputs; the first real 2026-07-24 observation had zero
  differences and zero read failures.
- Data Hub primary/shadow reads now produce payload-minimized, auditable
  comparisons with critical versus research classifications, bounded/expiring
  approval rules, idempotent DuckDB observations, and a 20-trading-day gate for
  unclassified differences, failure rate, P95 latency, and quota utilization;
  execution inputs remain unchanged.
- FRED macro observations and StockTwits/Reddit/Polymarket research signals
  now expose per-record freshness, required-series/sample coverage, explicit
  low-quality and failure states, and permanent non-broker/non-execution
  boundaries; production readers remain unchanged.
- Finnhub, Nasdaq, WallStreetCN, Yahoo, and RSS news/event inputs now share
  one research-only Data Hub envelope with inclusive time windows, cross-source
  deduplication, deterministic source priority, conflict evidence, and explicit
  partial-source degradation; production readers remain unchanged.
- SEC EDGAR CompanyFacts and submissions now normalize financial revision
  histories, corporate disclosures, and Form 3/4/5 insider filings into
  source-only Data Hub envelopes; missing sections and quota exhaustion are
  explicit, and no production reader has switched.
- Data Hub adapters now normalize Alpaca market data and authoritative account/
  position/order/fill facts; local-cache and Yahoo fallbacks are explicitly
  degraded and `execution_eligible=false`, with structured double-read differences.
- A production-neutral Data Hub contract now provides domain source registration,
  ordered fallback, adapter timeouts, required-field/time/quality validation, TTL
  cache, and explicitly degraded stale-cache fallback; no real reader switched yet.
- Shadow snapshots are replay-compared field by field, and an auditable N-trading-day
  quality report now gates on snapshot/source/critical-field/comparison coverage,
  content-hash replay, and zero unclassified differences without affecting orders.
- ResearchSnapshot content hashes, same-content deduplication, immutable run-symbol
  bindings, verified replay, cross-day isolation, and a KEEP_ALL retention policy
  now freeze each shadow research input without deleting historical records.
- Daily screening now shadow-captures the exact consumed bar frames, strategy
  statistics, and candidate fields into ResearchSnapshots; missing inputs and
  snapshot-write failures are explicit while the existing reader/conclusion is unchanged.
- Versioned immutable ResearchSnapshot and per-source manifest contracts now record
  source status, as-of/fetch/cutoff times, quality, failure, coverage, and payload
  versions in a backward-compatible DuckDB store; no production reader changed.
- A network-free Paper smoke harness now repeats BUY/SELL/reject/partial-fill/
  UNKNOWN/restart scenarios, while read-only audit queries trace each plan through
  deterministic risk, idempotent intent state, and incremental fills.
- Automatic submission is independently guarded at Runtime and Alpaca adapter
  boundaries: only Alpaca Paper LMT orders are accepted, Kill Switch is rechecked
  immediately before persistence/submission, and unknown broker types fail closed.
- Startup reconciliation restores Portfolio from durable fill deltas, applies known
  recent fills before quantity comparison, recovers open orders by broker or client
  order ID, and audits/blocks unexplained broker facts and API failures.
- Order submission is restart-safe for SENDING/UNKNOWN/open intents, and repeated
  cumulative broker fills now persist the correct cumulative and remaining quantity
  while Portfolio applies only the new delta.
- TradePlan pre-trade and pre-submit checks now enforce configured stop-loss
  risk as `abs(entry - stop) * qty`; invalid/non-finite inputs fail closed.
- BUY allocation and pre-submit risk now reserve existing long positions plus
  durable unfilled BUY notional, including UNKNOWN and partial orders across restarts.
- PaperDecision BUY/SELL direction now passes explicitly into ATRPlanner; candidate
  score is confidence evidence only and can no longer reverse a SELL decision.
- Installed official TradingAgents v0.3.1 in an external Python 3.13 environment;
  the application invokes it through an ASCII-safe JSON subprocess worker.
- Added Ollama qwen2.5/qwen3.6 32K model templates and verified the configured
  quick/deep model switch on a full MSFT analysis.
- TradingAgents Buy/Overweight/Hold/Underweight/Sell ratings map deterministically
  into BUY/HOLD/SELL research evidence; worker failures and timeouts remain explicit.
- Added deterministic daily screening enriched by reliable holdout statistics,
  an optional TradingAgents adapter, immutable run/item storage, and once-daily scheduling.
- Runtime consumes only the current trading date's frozen research snapshots;
  daily research and intraday market data now have separate validity windows.
- Added canonical READY/ENTERED/EXIT/CLOSED signal persistence and a lock-free
  real-time status sidecar rendered in the NiceGUI overview.
- Automatic Discord publication of the new report types remains disabled until
  the user explicitly authorizes the new external-send action.
- Codex collaboration defaults are explicit: single-task execution, opt-in
  delegation, reasonable safe assumptions, and separate sandbox-failure reporting.
- Documented the target autonomous operating model, unified research data hub,
  and versioned position-plan/invalidation contract; implementation remains
  explicitly separate from this target documentation.
- Decision-pool UI rebuilds prepare missing 1d history for forced ETFs before scoring; library callers remain local-only by default.
- PaperDecision is the mandatory strategy gate; enabled/shadow and per-trade approval paths are gone.
- Runtime runs a rate-limited AdvisoryWorker; AI production no longer depends on the UI.
- strategy_statistics.py evaluates the final 30% holdout from the local bar cache.
- Strategy statistics reject future timestamps, non-finite metrics, and invalid ranges.
- Research calculations run outside the NiceGUI event loop and restore button state on failure.
- Both batch entrypoints are ASCII/CRLF; setup preserves .env, and the monitor launcher was verified through HTTP 200.
- Removed unused functions, the single-implementation BrokerAdapter, stale tests, wrappers, and caches.
- PyYAML is explicit, Anthropic is optional, and test tools are development dependencies.
