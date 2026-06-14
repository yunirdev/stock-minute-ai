"""
main.py
Entry point for the trading platform.

Usage:
    python -m trader.main                          # paper mode, defaults
    python -m trader.main --live                   # LIVE mode (needs .env creds)
    python -m trader.main --symbols AAPL,MSFT --tf 5m --strategies "5/20均线金叉死叉"
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trader.config import RiskConfig, TradingConfig
from trader.scheduler import Scheduler
from trader.strategy_core import STRATEGY_OPTIONS


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="股票自动交易平台")
    p.add_argument("--symbols", default="AAPL",
                   help="逗号分隔的股票代码，例如 AAPL,MSFT")
    p.add_argument("--strategies", default="5/20均线金叉死叉",
                   help="逗号分隔的策略名称（英文或中文）")
    p.add_argument("--tf", default="5m",
                   help="K线周期: 1m | 5m | 15m | 30m | 1h | 1d")
    p.add_argument("--capital", type=float, default=10_000.0,
                   help="初始本金（仅paper模式有效）")
    p.add_argument("--interval", type=int, default=30,
                   help="轮询间隔（秒）")
    p.add_argument(
        "--broker-type", dest="broker_type",
        default=None,
        choices=["local_paper"],
        help="执行适配器: local_paper (默认读取 .env BROKER_TYPE)",
    )
    p.add_argument(
        "--data-feed", dest="data_feed",
        default=None,
        choices=["yfinance", "alpaca"],
        help="行情数据源: yfinance | alpaca (默认读取 .env DATA_FEED_TYPE)",
    )
    p.add_argument("--list-strategies", action="store_true",
                   help="列出所有支持的策略名称后退出")
    p.add_argument("--db-path", dest="db_path", default="",
                   help="覆盖 DuckDB 交易数据库路径")
    return p


def main() -> None:
    # Ensure UTF-8 output on Windows (prevents UnicodeEncodeError with Chinese text)
    import sys
    if sys.platform == "win32":
        import os
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
        print("支持的策略:")
        for name in STRATEGY_OPTIONS:
            print(f"  {name}")
        return

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]

    # broker_type: CLI flag wins; if not given, falls back to .env BROKER_TYPE
    broker_type = args.broker_type  # may be None → TradingConfig reads .env
    data_feed   = args.data_feed    # may be None → TradingConfig reads .env

    cfg = TradingConfig(
        symbols=symbols,
        strategies=strategies,
        timeframe=args.tf,
        initial_capital=args.capital,
        paper_mode=True,
        broker_type="local_paper",
        data_feed_type=data_feed or __import__('os').getenv("DATA_FEED_TYPE", "yfinance"),
        poll_interval_secs=args.interval,
        risk=RiskConfig(),
        db_path=args.db_path or __import__('os').getenv("TRADE_DB_PATH", "trade.duckdb"),
    )

    mode_str = "LOCAL PAPER 🗒️"
    print(f"\n模式: {mode_str}")
    print(f"标的: {cfg.symbols}")
    print(f"策略: {cfg.strategies}")
    print(f"周期: {cfg.timeframe}  间隔: {cfg.poll_interval_secs}s\n")

    scheduler = Scheduler(cfg)
    scheduler.run()


if __name__ == "__main__":
    main()
