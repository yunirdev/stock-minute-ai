"""
scheduler.py
Main Alpaca trading event loop.

Every poll interval:
  1. Fetch latest prices and broker account state.
  2. Poll previously submitted Alpaca orders.
  3. For each symbol and strategy, fetch bars, compute signals, run risk checks,
     and submit approved orders to Alpaca.
  4. Persist account snapshots and audit events to DuckDB.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np
import pandas as pd

from .audit import AuditLog
from .broker.alpaca import AlpacaBroker
from .broker.base import BrokerAdapter
from .config import TradingConfig
from .data_cache import upsert_bars as _dc_upsert
from .data_feed import AlpacaDataFeed
from .models import OrderIntent, OrderStatus, Side, Signal, new_id
from .portfolio import Portfolio
from .risk_engine import RiskEngine
from .strategies.registry import StrategyRegistry, build_default_registry

logger = logging.getLogger(__name__)


class Scheduler:
    """Orchestrates the complete data -> signal -> risk -> Alpaca execution path."""

    def __init__(self, config: TradingConfig) -> None:
        self._cfg = config

        if config.data_feed_type != "alpaca":
            raise ValueError(
                f"Unsupported data_feed_type={config.data_feed_type!r}; use alpaca"
            )
        self.feed = AlpacaDataFeed(config)
        logger.info("DataFeed: Alpaca")

        if config.broker_type not in ("alpaca_paper", "alpaca_live"):
            raise ValueError(
                f"Unsupported broker_type={config.broker_type!r}; "
                "use alpaca_paper or alpaca_live"
            )
        is_paper = config.broker_type != "alpaca_live"
        self.broker: BrokerAdapter = AlpacaBroker(
            config.alpaca_api_key,
            config.alpaca_secret_key,
            paper=is_paper,
        )
        logger.info("Broker: Alpaca %s", "PAPER" if is_paper else "LIVE")

        self.risk = RiskEngine(config)
        self.portfolio = Portfolio(config)
        self.audit = AuditLog(config)
        self.strategy_registry: StrategyRegistry = build_default_registry()

        self._open_orders: Dict[str, OrderIntent] = {}
        self._running = False
        self._tick_count = 0

    def run(self) -> None:
        """Start the trading loop. Press Ctrl-C to stop."""
        logger.info(
            "Scheduler start symbols=%s strategies=%s tf=%s interval=%ds",
            self._cfg.symbols,
            self._cfg.strategies,
            self._cfg.timeframe,
            self._cfg.poll_interval_secs,
        )
        self._running = True

        prices = self.feed.get_latest_prices(self._cfg.symbols)
        equity, _ = self._equity_and_positions(prices)
        self.risk.set_daily_start(equity)
        logger.info("Daily starting equity: %.2f", equity)

        while self._running:
            try:
                self._tick()
            except KeyboardInterrupt:
                logger.info("Stop signal received; scheduler exiting")
                break
            except Exception as exc:
                logger.error("Tick failed: %s", exc, exc_info=True)

            time.sleep(self._cfg.poll_interval_secs)

        logger.info("Scheduler stopped")

    def stop(self) -> None:
        self._running = False

    def _equity_and_positions(self, prices: dict):
        """Broker account is the source of truth for equity and positions."""
        equity = self.broker.get_account_equity()
        positions = {p.symbol: p for p in self.broker.get_positions()}
        return equity, positions

    def _tick(self) -> None:
        self._tick_count += 1
        prices = self.feed.get_latest_prices(self._cfg.symbols)
        equity, positions = self._equity_and_positions(prices)
        price_str = "  ".join(f"{s}={v:.2f}" for s, v in prices.items()) if prices else "(no quote)"
        logger.info("Tick #%d equity=%.2f %s", self._tick_count, equity, price_str)

        self.risk.check_equity(equity)
        if self.risk.is_halted:
            logger.warning("Risk halt: %s", self.risk.halt_reason)
            self.audit.log_heartbeat(self._tick_count, equity)
            return

        self._poll_alpaca_orders()

        for symbol in self._cfg.symbols:
            bars = self.feed.fetch_bars(symbol, n_bars=self._cfg.bars_lookback)
            logger.info("%s: %d bars", symbol, len(bars))
            if len(bars) < 30:
                logger.warning("%s: insufficient bars (%d), skipped", symbol, len(bars))
                continue

            df = _bars_to_df(bars)
            _dc_upsert(symbol, self._cfg.timeframe, df)
            last_bar = bars[-1]

            for strategy in self._cfg.strategies:
                try:
                    result = self.strategy_registry.compute(strategy, df)
                    self._route_signal(result, symbol, strategy, equity, positions, prices, last_bar)
                except Exception as exc:
                    logger.error(
                        "Strategy failed [%s / %s]: %s",
                        strategy,
                        symbol,
                        exc,
                        exc_info=True,
                    )

        self.portfolio.snapshot_external_equity(equity)
        self.audit.log_heartbeat(self._tick_count, equity)

    def _poll_alpaca_orders(self) -> None:
        if not self._open_orders:
            return
        done: List[str] = []
        for broker_id, intent in list(self._open_orders.items()):
            status = self.broker.get_order_status(broker_id)
            if status == OrderStatus.FILLED:
                fill = self.broker.get_fill(broker_id)
                if fill is not None:
                    fill.intent_id = intent.intent_id
                    self.audit.update_order_status(intent.intent_id, "FILLED")
                    self.portfolio.apply_fill(fill)
                    logger.info(
                        "ALPACA filled %s %s qty=%.0f @ %.4f",
                        fill.side.value,
                        fill.symbol,
                        fill.filled_qty,
                        fill.avg_price,
                    )
                done.append(broker_id)
            elif status in (OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.FAILED):
                self.audit.update_order_status(intent.intent_id, status.value)
                logger.info("ALPACA order closed %s status=%s", broker_id, status.value)
                done.append(broker_id)
        for broker_id in done:
            self._open_orders.pop(broker_id, None)

    def _route_signal(self, result, symbol, strategy, equity, positions, prices, last_bar) -> None:
        if result is None or result.empty:
            return
        last = result.iloc[-1]
        sig_val = int(last.get("strat_signal", 0))
        if sig_val == 0:
            return

        side = Side.BUY if sig_val > 0 else Side.SELL
        if strategy == "全仓买入并持有" and side == Side.BUY:
            pos = positions.get(symbol)
            if pos is not None and pos.qty > 0:
                return

        exec_px = float(last.get("strat_exec_px", last["close"]))
        if not np.isfinite(exec_px) or exec_px <= 0:
            exec_px = float(prices.get(symbol, last["close"]))

        signal = Signal(
            signal_id=new_id(),
            symbol=symbol,
            strategy=strategy,
            side=side,
            exec_price=exec_px,
            timeframe=self._cfg.timeframe,
            signal_time=datetime.now(timezone.utc),
            bar_close=float(last["close"]),
            metadata={
                "close": round(float(last["close"]), 4),
                "exec_px": round(exec_px, 4),
            },
        )
        logger.info("SIGNAL %s %s %s exec_px=%.4f", strategy, side.value, symbol, exec_px)
        self.audit.log_signal(signal)
        self._execute(signal, equity, positions)

    def _execute(self, signal: Signal, equity: float, positions: dict) -> None:
        verdict = self.risk.evaluate(signal, equity, positions)
        self.audit.log_risk_event(signal, verdict)
        if not verdict.approved:
            logger.info(
                "BLOCKED [%s %s %s]: %s",
                signal.strategy,
                signal.side.value,
                signal.symbol,
                verdict.reason,
            )
            return

        intent = OrderIntent(
            intent_id=new_id(),
            signal_id=signal.signal_id,
            symbol=signal.symbol,
            side=signal.side,
            qty=verdict.suggested_qty,
            order_type=self._cfg.order_type,
            limit_price=signal.exec_price if self._cfg.order_type == "LMT" else None,
            reference_price=signal.exec_price,
            risk_tag=verdict.reason,
        )
        try:
            broker_id = self.broker.place_order(intent)
            self.audit.log_order(intent, broker_id, "SUBMITTED")
            self.risk.record_success()
            self._open_orders[broker_id] = intent
            logger.info("ALPACA submitted %s", broker_id)
        except Exception as exc:
            self.risk.record_failure()
            self.audit.update_order_status(intent.intent_id, "FAILED")
            logger.error("Order failed [%s %s]: %s", intent.symbol, intent.side.value, exc)


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
