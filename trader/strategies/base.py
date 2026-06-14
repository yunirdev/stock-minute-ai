"""Strategy contracts used by live, paper, and backtest runtimes."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from ..models import Bar, Signal, Side, new_id


@dataclass
class StrategyContext:
    """Runtime context passed to event-driven script strategies."""

    symbol: str
    timeframe: str
    strategy_name: str
    params: Dict[str, Any] = field(default_factory=dict)
    current_bar: Optional[Bar] = None
    signals: List[Signal] = field(default_factory=list)

    def buy(
        self,
        exec_price: Optional[float] = None,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Signal:
        return self._emit(Side.BUY, exec_price, confidence, metadata)

    def sell(
        self,
        exec_price: Optional[float] = None,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Signal:
        return self._emit(Side.SELL, exec_price, confidence, metadata)

    def _emit(
        self,
        side: Side,
        exec_price: Optional[float],
        confidence: float,
        metadata: Optional[Dict[str, Any]],
    ) -> Signal:
        if self.current_bar is None:
            raise RuntimeError("StrategyContext.current_bar is required before emitting signals")
        price = float(exec_price if exec_price is not None else self.current_bar.close)
        signal = Signal(
            signal_id=new_id(),
            symbol=self.symbol,
            strategy=self.strategy_name,
            side=side,
            exec_price=price,
            timeframe=self.timeframe,
            signal_time=datetime.now(timezone.utc),
            bar_close=float(self.current_bar.close),
            confidence=confidence,
            metadata=metadata or {},
        )
        self.signals.append(signal)
        return signal


class Strategy(ABC):
    """Base strategy contract."""

    name: str

    @abstractmethod
    def compute(self, df: pd.DataFrame, **params: Any) -> pd.DataFrame:
        """Return a dataframe with strat_signal and strat_exec_px columns."""


@dataclass
class IndicatorStrategy(Strategy):
    """Adapter for dataframe-based signal functions."""

    name: str
    compute_fn: Callable[..., pd.DataFrame]

    def compute(self, df: pd.DataFrame, **params: Any) -> pd.DataFrame:
        return self.compute_fn(df, self.name, **params)


class ScriptStrategy(Strategy):
    """Base for event-driven on_bar strategies."""

    name = "script"

    def on_init(self, ctx: StrategyContext) -> None:
        """Optional initialization hook."""

    @abstractmethod
    def on_bar(self, ctx: StrategyContext, bar: Bar) -> None:
        """Handle one bar and optionally emit ctx.buy()/ctx.sell()."""

    def compute(self, df: pd.DataFrame, **params: Any) -> pd.DataFrame:
        out = df.copy()
        out["strat_signal"] = 0
        out["strat_exec_px"] = pd.NA
        return out
