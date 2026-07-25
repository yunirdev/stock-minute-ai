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
    uv run python -m trader.main --symbols AAPL,MSFT
    uv run python -m trader.main --symbols AAPL,MSFT --auto-trade --min-ai-score 70

Without --auto-trade, plans are recorded as DRY_RUN. With it, qualifying plans are submitted as idempotent Alpaca Paper limit orders.

## Runtime flow

1. Runtime reads market data and candidates, then runs agents in the background.
2. AgentManager writes analyses to ai_states.duckdb. The UI is optional.
3. PaperDecision requires a current strategy signal, reliable holdout statistics for the current market regime, and valid AI evidence.
4. Runtime creates ATR plans, allocates positions, applies deterministic risk controls, then records DRY_RUN or submits Paper LMT orders.
5. Reconciliation failures and the kill switch block new orders.

## Safety

- Agents never call the broker.
- Automatic submission requires auto_trade_paper and alpaca_paper.
- Only LMT orders are submitted.
- Missing, stale, untrusted, or insufficient AI evidence rejects a plan.
- Missing valid strategy statistics produces no decision.
- Kill switch, reconciliation, durable order records, idempotency, and risk controls cannot be bypassed.

## Entrypoints

- python -m trader.monitor_nice: NiceGUI monitor
- python -m trader.main: production Runtime with background AgentManager
- python -m trader.strategy_statistics: build statistics from local cached bars
- notebooks/research.py: Marimo research notebook

Anthropic fallback is optional: run uv sync --extra anthropic before setting its key.

README is the user guide. AGENTS.md is the only engineering baseline.

## Verification

    .venv\Scripts\python.exe -m pytest tests -q
    .venv\Scripts\python.exe -m compileall -q trader tests
    .venv\Scripts\python.exe -m ruff check trader tests

This project is for research and learning, not investment advice.
