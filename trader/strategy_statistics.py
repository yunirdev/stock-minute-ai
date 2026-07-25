"""Generate out-of-sample strategy statistics from the existing local bar cache."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .data_cache import get_bars
from .engine import simulate
from .paper_decision import StrategyStatistics
from .strategy_core import STRATEGY_OPTIONS, compute_signals

_BARS_PER_YEAR = {
    "1m": 252 * 390,
    "5m": 252 * 78,
    "15m": 252 * 26,
    "30m": 252 * 13,
    "1h": 252 * 7,
    "1d": 252,
}


def evaluate_strategy(
    bars: pd.DataFrame,
    *,
    symbol: str,
    strategy: str,
    timeframe: str,
    market_regime: str,
    now: datetime | None = None,
) -> StrategyStatistics | None:
    """Evaluate the last 30% of a series after computing signals on the full history."""
    if len(bars) < 200:
        return None
    frame = bars.copy()
    frame.columns = [str(column).lower() for column in frame.columns]
    required = {"timestamp_utc", "open", "high", "low", "close", "volume"}
    if not required.issubset(frame.columns):
        return None
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    signaled = compute_signals(frame, strategy)
    test = signaled.iloc[max(1, int(len(signaled) * 0.70)):].copy()
    if len(test) < 60:
        return None

    result = simulate(test, fee_bps=5, slippage_bps=5)
    returns = result.equity_curve.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    std = float(returns.std()) if len(returns) > 1 else 0.0
    sharpe = (
        float(returns.mean() / std * np.sqrt(_BARS_PER_YEAR.get(timeframe, 252)))
        if std > 0 else 0.0
    )
    peak = result.equity_curve.cummax()
    drawdown = ((peak - result.equity_curve) / peak.replace(0, np.nan)).fillna(0)
    closed = [trade for trade in result.trades if trade.side == "SELL"]
    evaluated = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    data_start = pd.Timestamp(test["timestamp_utc"].iloc[0]).to_pydatetime()
    data_end = pd.Timestamp(test["timestamp_utc"].iloc[-1]).to_pydatetime()
    fingerprint = json.dumps(
        {
            "symbol": symbol,
            "strategy": strategy,
            "timeframe": timeframe,
            "regime": market_regime,
            "data_end": data_end.isoformat(),
        },
        sort_keys=True,
    )
    return StrategyStatistics(
        statistics_id="stat-" + hashlib.sha256(fingerprint.encode()).hexdigest()[:20],
        symbol=symbol,
        strategy=strategy,
        strategy_version=hashlib.sha256(strategy.encode()).hexdigest()[:12],
        timeframe=timeframe,
        market_regime=market_regime,
        out_of_sample_net_return=result.total_return,
        sharpe=sharpe,
        max_drawdown=float(drawdown.max()),
        trade_count=len(closed),
        win_rate=(sum(trade.ret > 0 for trade in closed) / len(closed)) if closed else 0.0,
        average_trade_return=float(np.mean([trade.ret for trade in closed])) if closed else 0.0,
        fees=float(sum(trade.fee for trade in result.trades)),
        slippage=0.0005,
        data_start=data_start,
        data_end=data_end,
        evaluated_at=evaluated,
        statistics_version="holdout-70-30-v1",
    )


def generate_statistics(
    symbols: list[str],
    strategies: list[str],
    timeframe: str,
    market_regime: str,
) -> list[StrategyStatistics]:
    records = []
    for symbol in symbols:
        bars = get_bars(symbol, timeframe)
        for strategy in strategies:
            try:
                record = evaluate_strategy(
                    bars,
                    symbol=symbol,
                    strategy=strategy,
                    timeframe=timeframe,
                    market_regime=market_regime,
                )
                if record is not None:
                    records.append(record)
            except Exception as exc:
                print(f"skip {symbol} / {strategy}: {type(exc).__name__}")
    return records


def save_statistics(records: list[StrategyStatistics], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for record in records:
        row = asdict(record)
        for key in ("data_start", "data_end", "evaluated_at"):
            row[key] = row[key].isoformat()
        payload.append(row)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate PaperDecision strategy statistics")
    parser.add_argument("--symbols", required=True, help="Comma-separated cached symbols")
    parser.add_argument("--strategies", default="", help="Comma-separated names; default: all")
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--regime", default="")
    parser.add_argument("--output", default="conf/strategy_statistics.json")
    args = parser.parse_args()

    regime = args.regime
    if not regime:
        try:
            from .teams.market_env import read_regime_cache
            cached = read_regime_cache()
            regime = cached.regime.value if cached else "no_cache"
        except Exception:
            regime = "unknown"
    symbols = [value.strip().upper() for value in args.symbols.split(",") if value.strip()]
    strategies = [value.strip() for value in args.strategies.split(",") if value.strip()]
    records = generate_statistics(
        symbols,
        strategies or list(STRATEGY_OPTIONS),
        args.timeframe,
        regime,
    )
    save_statistics(records, args.output)
    reliable = sum(record.reliable(datetime.now(timezone.utc)) for record in records)
    print(f"wrote {len(records)} statistics ({reliable} reliable) to {args.output}")
    return 0 if reliable else 1


if __name__ == "__main__":
    raise SystemExit(main())
