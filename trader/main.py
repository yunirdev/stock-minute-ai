"""
main.py
Entry point for the Alpaca trading engine (Runtime pipeline).

Usage:
    python -m trader.main
    python -m trader.main --symbols AAPL,MSFT --tf 5m
    python -m trader.main --auto-trade              # AI 自动虚拟盘，评分 ≥ 65 自动下单
    python -m trader.main --auto-trade --min-ai-score 70
    python -m trader.main --broker-type alpaca_live  # 实盘（谨慎）
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trader.config import RiskConfig, TradingConfig, settings
from trader.strategy_core import STRATEGY_OPTIONS


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Alpaca trading engine (Runtime pipeline)")
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

    # ── AI 自动交易 ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--auto-trade",
        dest="auto_trade",
        action="store_true",
        default=False,
        help=(
            "AI 自动虚拟盘交易：读取 ai_states.duckdb 的 AI 综合评分，"
            "分数与确定性风控达标后向 Alpaca paper 自动提交 LMT 单。"
        ),
    )
    parser.add_argument(
        "--min-ai-score",
        dest="min_ai_score",
        type=int,
        default=None,
        help="AI 综合评分门槛（0-100），≥ 此值才自动下单（默认 65）。仅 --auto-trade 时生效。",
    )
    parser.add_argument(
        "--ai-score-db",
        dest="ai_score_db",
        default=None,
        help="AI 评分数据库路径（默认 ai_states.duckdb）。",
    )

    parser.add_argument("--ai-score-max-age-minutes", type=float, default=None)
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
    # yfinance 对取不到数据的 ETF/指数（无 fundamentals/sector）会自行打 ERROR 日志，
    # 跟我们的 try/except 无关，单独调高级别屏蔽（不影响我们自己的报错处理）。
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)

    args = _build_parser().parse_args()

    if args.list_strategies:
        print("Supported strategies:")
        for name in STRATEGY_OPTIONS:
            print(f"  {name}")
        return

    symbols     = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    strategies  = [s.strip() for s in args.strategies.split(",") if s.strip()]
    broker_type = args.broker_type or settings.broker_type or "alpaca_paper"
    data_feed   = args.data_feed or os.getenv("DATA_FEED_TYPE", "alpaca")

    cfg = TradingConfig(
        symbols=symbols,
        strategies=strategies,
        timeframe=args.tf,
        broker_type=broker_type,
        data_feed_type=data_feed,
        poll_interval_secs=args.interval,
        risk=RiskConfig(),
        db_path=args.db_path or os.getenv("TRADE_DB_PATH", "trade.duckdb"),
        # AI 自动交易参数
        auto_trade_paper=args.auto_trade,
        min_ai_score=args.min_ai_score if args.min_ai_score is not None else settings.min_ai_score,
        ai_score_db=args.ai_score_db or settings.ai_score_db,
        ai_score_max_age_minutes=(
            args.ai_score_max_age_minutes
            if args.ai_score_max_age_minutes is not None
            else settings.ai_score_max_age_minutes
        ),
    )

    mode_names = {
        "alpaca_paper": "ALPACA PAPER (虚拟盘，不动真实资金)",
        "alpaca_live":  "ALPACA LIVE  (实盘，真实资金！)",
    }
    print(f"\n{'=' * 60}")
    print(f"  模式: {mode_names.get(broker_type, broker_type)}")
    print(f"  标的: {cfg.symbols}")
    print(f"  策略: {cfg.strategies}")
    print(f"  周期: {cfg.timeframe}  轮询: {cfg.poll_interval_secs}s")
    if args.auto_trade:
        print(f"  AI 自动交易: 开启（评分门槛 ≥ {cfg.min_ai_score}，读取 {cfg.ai_score_db}）")
    else:
        print("  AI 自动交易: 关闭（DRY-RUN，不实际下单）")
    if broker_type == "alpaca_live":
        print("\n  ⚠️  WARNING: live 模式将下真实资金订单！按 Ctrl-C 停止引擎。")
    print(f"{'=' * 60}\n")

    from trader.runtime import Runtime
    Runtime(cfg).run()


def _run_entrypoint() -> None:
    try:
        main()
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        try:
            from trader.bug_reporting import BugReporter

            BugReporter(
                os.getenv("TRADE_DB_PATH", "trade.duckdb"), "startup"
            ).capture_exception(exc, operation="trader.main.startup")
        except Exception:
            pass
        logging.getLogger(__name__).critical(
            "Trading engine startup failed: %s", type(exc).__name__
        )
        raise


if __name__ == "__main__":
    _run_entrypoint()
