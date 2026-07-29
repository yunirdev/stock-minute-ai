# stock-minute-ai

AI-assisted minute-bar Alpaca Paper trading with multi-agent analysis, verified strategy statistics, deterministic risk controls, DuckDB audit, and a NiceGUI monitor.

Automatic trading is Paper-only. It never requests per-trade approval and never submits automatic live orders.

## Start

Python 3.13+ and uv are required.

    setup.bat
    # Fill Alpaca Paper keys in .env
    启动监控台.bat

Before automated trading, download enough local bar history in the UI and generate statistics:

    uv run python -m trader.strategy_statistics --symbols AAPL,MSFT --timeframe 5m
    uv run python -m trader.daily_research --symbols AAPL,MSFT,NVDA --strategy-statistics-path conf/strategy_statistics.json
    uv run python -m trader.main --symbols AAPL,MSFT
    uv run python -m trader.main --symbols AAPL,MSFT --auto-trade --min-ai-score 70

The NiceGUI selection-pool rebuild actions fetch missing daily history for the ETFs
injected into the decision pool before scoring. Failed downloads remain blocked.

Without --auto-trade, plans are recorded as DRY_RUN. With it, qualifying plans are submitted as idempotent Alpaca Paper limit orders.

## Runtime flow

1. A content-addressed universe registry preserves stock, ETF, and fund
   metadata, including delisted and non-tradable assets. Deterministic focus
   pools rank only active/tradable assets using reliable holdout, liquidity,
   and data-quality evidence; failed rebuilds preserve the prior valid pool.
   A durable research-budget queue then limits symbols, estimated cost and
   runtime, processes stable priority batches, bounds retries/timeouts, and
   resumes without reclaiming completed work. Daily REAL/SYNTHETIC-separated
   observations gate screening coverage, research completion/failure, cost,
   and completion-window quality over 20 sessions.
2. The daily batch screens the configured focus symbols without AI and enriches the
   ranking with reliable holdout statistics.
3. TradingAgents analyzes only the deep shortlist. The immutable run and per-symbol
   results are written to ai_states.duckdb once for the target trading date.
4. Runtime reads that frozen daily shortlist; it does not rerun the full LLM agent
   set every tick. Intraday bars, strategy votes, positions, and orders remain live.
5. PaperDecision requires a current strategy signal, reliable holdout statistics
   for the current market regime, and a valid result from today's research run.
6. Runtime creates ATR plans, allocates positions, applies deterministic risk,
   persists signal lifecycle events, then records DRY_RUN or routes executable
   plans through `CandidatePlan → FinalTradePlan → OrderIntent` before submitting
   Paper LMT orders. New entries, monitored exits, invalidation adjustments, and
   recovered missing intents all use this same production pipeline.
   Confirmed cumulative fills are projected transactionally into one immutable
   `PositionPlan` version chain; partial, repeated, reduction, and closing fills
   cannot silently replace the original entry, stop, target, or quantity baseline.
   Potential invalidations are stored separately and only accepted from
   type-specific market, broker, corporate-action, exchange, or deterministic
   strategy facts with current timestamps and evidence references; free-form
   model text is not an invalidation event. A validated event can create one
   transactional `PositionAdjustment`: exit, reduce, or tighten the stop. Exit
   and reduction become durable, idempotent Paper LMT order intents through the
   same Runtime execution guard; long stops cannot be loosened.
   Startup reconciliation restores open adjustment orders, creates a missing
   intent for a durable `PLANNED` adjustment, replays only new cumulative fill
   deltas into the PositionPlan chain, and can rebuild a missed initial plan
   from its audited TradePlan. UNKNOWN orders are never guessed or resubmitted.
   Each successful broker reconciliation records a REAL daily position-quality
   observation and a 30-session report covering broker/local/plan quantities,
   unlinked version changes, and duplicate adjustments. REAL and SYNTHETIC
   evidence never mix. A one-time broker-authoritative portfolio baseline can
   retire legacy local-paper fills without deleting or fabricating fills; all
   later broker fills are replayed on top of that immutable baseline.
   PositionPlan fills also produce immutable trade-episode attribution snapshots
   for partial fills, reductions, closes, cross-day duration, realized PnL,
   limit slippage, invalidations, and adjustments. Frozen reviews keep facts,
   decisions, execution, and results separate under a stable error taxonomy;
   a losing trade is not automatically treated as strategy invalidation.
   A frozen review may append an immutable strategy candidate that records its
   production baseline, data/code/parameter versions, and strictly separated
   training and holdout windows. Repeated generation is content-idempotent and
   parameter changes create a linked child version; this candidate store has no
   API for modifying production parameters. A separate append-only release
   audit compares candidate and champion on frozen holdout, non-overlapping
   historical replay, and Paper evidence. Promotion requires minimum bar,
   session, and trade samples plus explicit fee, slippage, Sharpe, return, and
   drawdown gates. It preserves the prior champion as a rollback version, while
   Runtime does not consume or automatically apply the promoted parameters.
7. logs/runtime_status.json provides a lock-free live status feed rendered on the
   NiceGUI overview. Reconciliation failures and the kill switch block new orders.
8. Runtime registers every scheduled Paper session and freezes REAL maturity
   evidence after 20:00 ET. At the same cutoff it creates one idempotent,
   checksummed, read-verified backup of the trade and AI DuckDB files under
   `backups/YYYY-MM-DD/`; source databases are never overwritten.
9. All 24 rendered NiceGUI actions use the durable action-audit gateway;
   worker-backed actions remain BUSY until their real outcome is known.
   Discord delivery is routed through one authorization/redaction/idempotency
   gateway and remains externally disabled unless
   `DISCORD_EXTERNAL_SEND_ENABLED=true` or an explicit UI/manual send is used.
   The NiceGUI Paper auto-trade checkbox is session-scoped and defaults off on
   every app start; saved UI preferences cannot silently re-enable submission.
   Runtime status older than three minutes is rendered as stopped/stale rather
   than healthy.

The daily worker runs once in a pre-market window or once after 16:15 ET for the
next weekday. Research evidence is valid for 36 hours by default; intraday bar
freshness is checked separately in seconds/minutes.

TradingAgents runs in a dedicated external Python environment and is invoked
through a JSON subprocess worker; its LangChain/LangGraph dependencies are not
installed into the production Runtime environment. Set TRADINGAGENTS_PYTHON and
TRADINGAGENTS_PROJECT_DIR to that installation. Missing workers, provider
failures, and timeouts are recorded as FAILED; the application never fabricates
a fallback research score. Screening and snapshot capture share one injected
batch clock, stale RUNNING/PENDING work is failed after the configured worker
timeout, and every failed run/item persists a non-empty diagnostic code. Each
new worker call is bound to its immutable
research run, snapshot content hash, data version, and model/config version;
unmatched or stale output is rejected. The verified Ollama setup and upgrade procedure are
documented in [docs/TRADINGAGENTS_LOCAL_SETUP.md](docs/TRADINGAGENTS_LOCAL_SETUP.md).

## Safety

- Agents never call the broker.
- Automatic submission requires auto_trade_paper and alpaca_paper.
- Only LMT orders are submitted.
- Missing, stale, untrusted, or insufficient AI evidence rejects a plan.
- Missing valid strategy statistics produces no decision.
- Kill switch, reconciliation, durable order records, idempotency, and risk controls cannot be bypassed.

## Entrypoints

- python -m trader.monitor_nice: NiceGUI monitor
- python -m trader.main: production Runtime consuming the frozen daily research
- python -m trader.daily_research: manual/automation entrypoint for one daily batch
- python -m trader.strategy_statistics: build statistics from local cached bars
- python -m trader.paper_smoke: network-free execution/restart smoke harness
- python -m trader.audit_query: read-only plan/risk/order/fill trace query
- python -m trader.research_snapshot_quality: 10-day shadow snapshot quality report
- python -m trader.data_hub_shadow: read-only Alpaca/Data Hub double-read quality cycle
- python -m trader.data_hub_replay: 20-session local/Alpaca historical correctness replay
- notebooks/research.py: Marimo research notebook

Anthropic fallback is optional: run uv sync --extra anthropic before setting its key.

README is the user guide. AGENTS.md is the only engineering baseline.
The isolated execution drill is documented in
[docs/PAPER_SMOKE_RUNBOOK.md](docs/PAPER_SMOKE_RUNBOOK.md).
Research snapshot quality metrics are documented in
[docs/RESEARCH_SNAPSHOT_QUALITY.md](docs/RESEARCH_SNAPSHOT_QUALITY.md).
The Data Hub 20-trading-day observation procedure is documented in
[docs/DATA_HUB_SHADOW_RUNBOOK.md](docs/DATA_HUB_SHADOW_RUNBOOK.md).
The current autonomous Paper-loop and full NiceGUI action acceptance contract is
documented in
[docs/CLOSED_LOOP_ACCEPTANCE.md](docs/CLOSED_LOOP_ACCEPTANCE.md).
The H-stage recovery procedure and provisional Paper-loop delivery report are
documented in
[docs/OPERATIONS_RECOVERY_RUNBOOK.md](docs/OPERATIONS_RECOVERY_RUNBOOK.md) and
[docs/PAPER_CLOSED_LOOP_SIGNOFF.md](docs/PAPER_CLOSED_LOOP_SIGNOFF.md).
Long-running maturity evidence and the architecture/final sign-off distinction
are documented in
[docs/PAPER_MATURITY_RUNBOOK.md](docs/PAPER_MATURITY_RUNBOOK.md) and
[docs/PAPER_MIGRATION_SIGNOFF.md](docs/PAPER_MIGRATION_SIGNOFF.md).
NiceGUI web mode binds to `127.0.0.1` by default; set `QUANT_HOST` explicitly
only when a different interface is intentionally required.
The trading-record page renders the newest order's complete explanation chain.
Read-only integrations can use `/api/ui-actions` and
`/api/order-explanation/{plan_id}`.

## Verification

    .venv\Scripts\python.exe -m pytest tests -q
    .venv\Scripts\python.exe -m compileall -q trader tests
    .venv\Scripts\python.exe -m ruff check trader tests

This project is for research and learning, not investment advice.
