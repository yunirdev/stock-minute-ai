"""
scheduler.py
Main trading event loop.

Every *poll_interval_secs* seconds:
  1. Fetch latest prices  →  equity check  →  risk circuit breaker
  2. For each symbol × strategy:
       fetch bars  →  compute_signals  →  evaluate risk  →  place order
  3. Apply paper fills immediately; live fills tracked asynchronously.
  4. Persist equity snapshot to DuckDB.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import List

import numpy as np
import pandas as pd

from .audit import AuditLog
from .broker.base import BrokerAdapter
from .broker.paper import PaperBroker
from .config import TradingConfig
from .data_cache import upsert_bars as _dc_upsert
from .data_feed import AlpacaDataFeed
from .models import OrderIntent, PendingOrder, Side, Signal, new_id
from .order_manager import OrderManager
from .portfolio import Portfolio
from .risk_engine import RiskEngine
from .strategies.registry import StrategyRegistry, build_default_registry

logger = logging.getLogger(__name__)


class Scheduler:
    """Orchestrates the complete data → signal → risk → execution pipeline."""

    def __init__(self, config: TradingConfig) -> None:
        self._cfg = config
        # Select data feed based on config
        if config.data_feed_type == "yfinance":
            from .data_feed_yfinance import YFinanceDataFeed
            self.feed = YFinanceDataFeed(config)
            logger.info("DataFeed: Yahoo Finance (yfinance)")
        else:  # "alpaca"
            self.feed = AlpacaDataFeed(config)
            logger.info("DataFeed: Alpaca")
        self.risk = RiskEngine(config)
        self.portfolio = Portfolio(config)
        self.audit = AuditLog(config)
        self.strategy_registry: StrategyRegistry = build_default_registry()

        if config.broker_type == "local_paper":
            self.broker: BrokerAdapter = PaperBroker()
            logger.info("📝 LOCAL PAPER MODE — 不会产生真实订单")
        else:
            # Live broker placeholder — extend BrokerAdapter and set broker_type in .env
            logger.warning("broker_type=%r 未实现，回退到 PaperBroker", config.broker_type)
            self.broker = PaperBroker()

        self.order_mgr = OrderManager(
            max_bars_alive=config.pending_order_max_bars,
        )

        self._running = False
        self._tick_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the trading loop (blocking). Press Ctrl-C to stop."""
        logger.info(
            "Scheduler 启动  symbols=%s  strategies=%s  tf=%s  interval=%ds",
            self._cfg.symbols, self._cfg.strategies,
            self._cfg.timeframe, self._cfg.poll_interval_secs,
        )
        self._running = True

        # Calibrate daily equity baseline
        prices = self.feed.get_latest_prices(self._cfg.symbols)
        equity = self.portfolio.snapshot_equity(prices)
        self.risk.set_daily_start(equity)
        logger.info("日初资产: %.2f", equity)

        while self._running:
            try:
                self._tick()
            except KeyboardInterrupt:
                logger.info("收到终止信号，调度器退出")
                break
            except Exception as exc:
                logger.error("Tick 异常: %s", exc, exc_info=True)

            time.sleep(self._cfg.poll_interval_secs)

        logger.info("Scheduler 已停止")

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # Internal tick
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        self._tick_count += 1
        prices = self.feed.get_latest_prices(self._cfg.symbols)
        equity = self.portfolio.get_equity(prices)
        price_str = "  ".join(f"{s}={v:.2f}" for s, v in prices.items()) if prices else "(无报价)"
        logger.info("⏱  Tick #%d  equity=%.2f  %s", self._tick_count, equity, price_str)

        self.risk.check_equity(equity)
        if self.risk.is_halted:
            logger.warning("⛔ 系统熔断: %s", self.risk.halt_reason)
            # Still write heartbeat so UI knows engine is alive (just halted)
            self.audit.log_heartbeat(self._tick_count, equity)
            return

        for symbol in self._cfg.symbols:
            bars = self.feed.fetch_bars(symbol, n_bars=self._cfg.bars_lookback)
            logger.info("  %s: %d bars", symbol, len(bars))
            if len(bars) < 30:
                logger.warning("  %s: bar 数量不足 (%d)，跳过本轮", symbol, len(bars))
                continue

            df = _bars_to_df(bars)
            _dc_upsert(symbol, self._cfg.timeframe, df)

            # ── Check pending (gap-fill) orders against current bar ──────
            last_bar = bars[-1]
            self.order_mgr.check_pending(
                symbol=symbol,
                bar_high=last_bar.high,
                bar_low=last_bar.low,
                bar_close=last_bar.close,
                on_fill=lambda pending, _eq=equity, _px=prices: self._fill_pending(pending, _eq, _px),
            )

            for strategy in self._cfg.strategies:
                try:
                    result = self.strategy_registry.compute(strategy, df)
                    self._route_signal(result, symbol, strategy, equity, prices, last_bar)
                except Exception as exc:
                    logger.error(
                        "策略计算失败 [%s / %s]: %s", strategy, symbol, exc, exc_info=True)

        self.portfolio.snapshot_equity(prices)
        self.audit.log_heartbeat(self._tick_count, equity)

    # ------------------------------------------------------------------
    # Direct signal → risk → order → fill (replaces the old event bus +
    # SignalRouter + ExecutionPipeline; same behaviour, reads top-to-bottom)
    # ------------------------------------------------------------------

    def _route_signal(self, result, symbol, strategy, equity, prices, last_bar) -> None:
        if result is None or result.empty:
            return
        last = result.iloc[-1]
        sig_val = int(last.get("strat_signal", 0))
        if sig_val == 0:
            return
        side = Side.BUY if sig_val > 0 else Side.SELL

        # buy-and-hold: don't re-buy if already long
        if strategy == "全仓买入并持有" and side == Side.BUY:
            pos = self.portfolio.positions.get(symbol)
            if pos is not None and pos.qty > 0:
                return

        exec_px = float(last.get("strat_exec_px", last["close"]))
        if not np.isfinite(exec_px) or exec_px <= 0:
            exec_px = float(prices.get(symbol, last["close"]))

        signal = Signal(
            signal_id=new_id(), symbol=symbol, strategy=strategy, side=side,
            exec_price=exec_px, timeframe=self._cfg.timeframe,
            signal_time=datetime.now(timezone.utc), bar_close=float(last["close"]),
            metadata={"close": round(float(last["close"]), 4), "exec_px": round(exec_px, 4)},
        )
        logger.info("📡 SIGNAL %s %s %s exec_px=%.4f", strategy, side.value, symbol, exec_px)
        self.audit.log_signal(signal)
        self.order_mgr.cancel_opposite(symbol, side)

        # gap-fill: only execute now if exec_px is reachable on this bar
        fill_now = self.order_mgr.maybe_enqueue(
            signal=signal, bar_open=last_bar.open, bar_high=last_bar.high,
            bar_low=last_bar.low, bar_close=last_bar.close,
        )
        if fill_now:
            self._execute(signal, equity, prices)

    def _fill_pending(self, pending: PendingOrder, equity: float, prices: dict) -> None:
        """A queued gap order became reachable — execute at its limit price."""
        src = pending.signal
        filled = Signal(
            signal_id=new_id(), symbol=src.symbol, strategy=src.strategy, side=src.side,
            exec_price=pending.limit_price, timeframe=src.timeframe,
            signal_time=datetime.now(timezone.utc), bar_close=src.bar_close,
            metadata={**src.metadata, "pending_id": pending.pending_id,
                      "bars_waited": pending.bars_alive},
        )
        logger.info("📬 PENDING→FILL %s %s limit=%.4f", src.side.value, src.symbol, pending.limit_price)
        self.audit.log_signal(filled)
        self._execute(filled, equity, prices)

    def _execute(self, signal: Signal, equity: float, prices: dict) -> None:
        """Risk-check, then place a paper order and apply the fill."""
        verdict = self.risk.evaluate(signal, equity, self.portfolio.positions)
        self.audit.log_risk_event(signal, verdict)
        if not verdict.approved:
            logger.info("🚫 BLOCKED [%s %s %s]: %s",
                        signal.strategy, signal.side.value, signal.symbol, verdict.reason)
            return
        intent = OrderIntent(
            intent_id=new_id(), signal_id=signal.signal_id, symbol=signal.symbol,
            side=signal.side, qty=verdict.suggested_qty, order_type=self._cfg.order_type,
            limit_price=signal.exec_price if self._cfg.order_type == "LMT" else None,
            reference_price=signal.exec_price, risk_tag=verdict.reason,
        )
        try:
            broker_id = self.broker.place_order(intent)
            self.audit.log_order(intent, broker_id, "SUBMITTED")
            self.risk.record_success()
            if self._cfg.paper_mode:
                fill = self.broker.get_fill(broker_id)
                if fill:
                    fill.intent_id = intent.intent_id
                    self.portfolio.apply_fill(fill)
            else:
                logger.info("⏳ 等待成交回报 broker_id=%s", broker_id)
        except Exception as exc:
            self.risk.record_failure()
            self.audit.update_order_status(intent.intent_id, "FAILED")
            logger.error("❌ 下单失败 [%s %s]: %s", intent.symbol, intent.side.value, exc)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _bars_to_df(bars: List) -> pd.DataFrame:
    rows = [
        {
            "timestamp_utc": b.timestamp,
            "open": b.open,
            "high": b.high,
            "low": b.low,
            "close": b.close,
            "volume": b.volume,
        }
        for b in bars
    ]
    df = pd.DataFrame(rows)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df.reset_index(drop=True)
