# PaperDecision

PaperDecision is the mandatory Runtime strategy gate. It is not manual approval and it does not submit orders.

Flow:

    candidate current strategy votes
      -> matching symbol, timeframe, and market-regime holdout statistics
      -> retain only strategies with a current BUY or SELL signal
      -> verify AI evidence
      -> StrategyDecision
      -> ATR plan -> allocation -> risk -> DRY_RUN or Alpaca Paper LMT

Each StrategyDecision stores strategy/version, direction, statistics record, AI run, data/universe version, expiry, and rejected alternatives. It links to the resulting TradePlan and durable OrderIntent.

## Statistics

Build statistics from the local cache:

    uv run python -m trader.strategy_statistics --symbols AAPL,MSFT --timeframe 5m

The generator computes signals on full history and evaluates only the final 30 percent holdout. A record requires at least 30 closed trades, fresh evaluation, and maximum drawdown no greater than 100 percent. Missing matching statistics fail closed.

Statistics are tied to the current T0 market regime. Regenerate them after a regime change.

## Agent production

Runtime uses a rate-limited, single-threaded AdvisoryWorker to run AgentManager and write ai_states.duckdb. The UI does not need to remain open. Stub, stale, or untrusted scores are rejected. Pure quantitative operation requires explicit ALLOW_QUANT_WITHOUT_AI=true.

Only --auto-trade can submit, and then only Alpaca Paper LMT orders. Reconciliation, idempotency, kill switch, and deterministic risk controls remain mandatory.
