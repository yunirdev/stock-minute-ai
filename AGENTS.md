# AGENTS.md - stock-minute-ai engineering baseline

Read this file before any work. Update it after every code, test, or architecture change. It is the only authoritative engineering context.

## Product

- AI-assisted Alpaca Paper trading. No per-trade user approval. No automatic live trading.
- NiceGUI is for monitoring and research. Runtime is the only production trading loop.
- Agents create analysis only. Runtime is the only component that may submit orders.
- Without --auto-trade, plans are DRY_RUN. With it, qualifying plans submit Paper LMT orders.

## Single production path

    Runtime -> market data and candidate strategy votes
      -> background AgentManager -> ai_states.duckdb
      -> PaperDecision: holdout statistics + market regime + current strategy signal + AI evidence
      -> ATR TradePlan -> allocation -> deterministic risk
      -> DRY_RUN or durable/idempotent Alpaca Paper LMT
      -> order polling -> portfolio and audit persistence

Entrypoints:

- python -m trader.monitor_nice
- python -m trader.main [--auto-trade]
- python -m trader.strategy_statistics --symbols AAPL,MSFT --timeframe 5m

## Non-negotiable safety

1. Automatic submission requires broker_type=alpaca_paper and auto_trade_paper=True.
2. Automatic execution creates only LMT orders.
3. A plan needs a current BUY/SELL strategy signal and reliable matching holdout statistics.
4. AI evidence must exist, be fresh, trusted, and meet the configured score. Quant-only operation must be explicit.
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
- ai/manager.py and ai/agents: analysis system
- monitor_nice.py and monitor_data.py: UI and read models
- strategies, strategy_core.py, factors, backtest: research
- watchdog.py and kill_switch.py: runtime safety

Do not restore Scheduler, in-memory PaperBroker, separate yfinance feed, OrchestratorAgent, Streamlit preferences, per-trade approval, Protocol shells, PendingOrder, BrokerAdapter, or PaperDecision enabled/shadow dual paths.

## Data rules

Commit source, tests, README, this file, configuration templates, and active technical documentation.

Never commit caches, bytecode, test/lint caches, .nicegui, .tmp, egg-info, generated conf snapshots, strategy_statistics.json, downloaded external projects, secrets, databases, or logs.

Do not delete trade.duckdb, ai_states.duckdb, logs, or conf/ui_settings.json. They are user records or preferences.

## Working protocol

Before work: read this file, check git status, trace production callers.

After work: remove new cache files; run proportionate verification; update Current baseline and Recent changes below; inspect diff/status for secrets and generated data; update README if behavior changes.

## Verification

    .venv\Scripts\python.exe -m pytest tests -q
    .venv\Scripts\python.exe -m ruff check trader tests
    .venv\Scripts\python.exe -m compileall -q trader tests

## Current baseline

- Date: 2026-07-24
- Baseline: PaperDecision production-path consolidation (this changeset)
- Verification: 142 tests passed; Ruff, compileall, setup.bat, the monitor launcher, both CLI entrypoints, and NiceGUI HTTP smoke passed.
- Goal achieved: one auditable Paper-only path that fails closed on strategy statistics and AI evidence.

## Recent changes

- PaperDecision is the mandatory strategy gate; enabled/shadow and per-trade approval paths are gone.
- Runtime runs a rate-limited AdvisoryWorker; AI production no longer depends on the UI.
- strategy_statistics.py evaluates the final 30% holdout from the local bar cache.
- Strategy statistics reject future timestamps, non-finite metrics, and invalid ranges.
- Research calculations run outside the NiceGUI event loop and restore button state on failure.
- Both batch entrypoints are ASCII/CRLF; setup preserves .env, and the monitor launcher was verified through HTTP 200.
- Removed unused functions, the single-implementation BrokerAdapter, stale tests, wrappers, and caches.
- PyYAML is explicit, Anthropic is optional, and test tools are development dependencies.
