"""
main.py
Entry point for the Alpaca trading engine.

Usage:
    python -m trader.main
    python -m trader.main --broker-type alpaca_live
    python -m trader.main --symbols AAPL,MSFT --tf 5m
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trader.config import RiskConfig, TradingConfig, settings
from trader.scheduler import Scheduler
from trader.strategy_core import STRATEGY_OPTIONS


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Alpaca trading engine")
    parser.add_argument("--symbols", default="AAPL", help="Comma-separated symbols, e.g. AAPL,MSFT")
    parser.add_argument(
        "--strategies",
        default="5/20均线金叉死叉",
        help="Comma-separated strategy names",
    )
    parser.add_argument("--tf", default="5m", help="Bar timeframe: 1m | 5m | 15m | 30m | 1h | 1d")
    parser.add_argument("--interval", type=int, default=30, help="Polling interval in seconds")
    parser.add_argument(
        "--broker-type",
        dest="broker_type",
        default=None,
        choices=["alpaca_paper", "alpaca_live"],
        help="Execution broker. Defaults to .env BROKER_TYPE.",
    )
    parser.add_argument(
        "--data-feed",
        dest="data_feed",
        default=None,
        choices=["alpaca"],
        help="Market data feed. Defaults to .env DATA_FEED_TYPE.",
    )
    parser.add_argument("--list-strategies", action="store_true", help="List supported strategies and exit")
    parser.add_argument("--db-path", dest="db_path", default="", help="Override trading audit DuckDB path")
    return parser


def main() -> None:
    if sys.platform == "win32":
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = _build_parser().parse_args()

    if args.list_strategies:
        print("Supported strategies:")
        for name in STRATEGY_OPTIONS:
            print(f"  {name}")
        return

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    broker_type = args.broker_type or settings.broker_type or "alpaca_paper"
    data_feed = args.data_feed or os.getenv("DATA_FEED_TYPE", "alpaca")

    cfg = TradingConfig(
        symbols=symbols,
        strategies=strategies,
        timeframe=args.tf,
        broker_type=broker_type,
        data_feed_type=data_feed,
        poll_interval_secs=args.interval,
        risk=RiskConfig(),
        db_path=args.db_path or os.getenv("TRADE_DB_PATH", "trade.duckdb"),
    )

    mode_names = {
        "alpaca_paper": "ALPACA PAPER (paper account, no real money)",
        "alpaca_live": "ALPACA LIVE (real money)",
    }
    if broker_type == "alpaca_live":
        print("\nWARNING: live mode will place real-money orders. Ctrl-C stops the engine.")
    print(f"\nMode: {mode_names.get(broker_type, broker_type)}")
    print(f"Symbols: {cfg.symbols}")
    print(f"Strategies: {cfg.strategies}")
    print(f"Timeframe: {cfg.timeframe}  Interval: {cfg.poll_interval_secs}s\n")

    Scheduler(cfg).run()


if __name__ == "__main__":
    main()
