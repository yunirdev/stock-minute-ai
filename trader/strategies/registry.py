"""Strategy registry used by runtimes to resolve strategy names."""
from __future__ import annotations

from typing import Dict, Iterable

import pandas as pd

from .base import IndicatorStrategy, Strategy


class StrategyRegistry:
    """In-memory catalog of available strategy implementations."""

    def __init__(self) -> None:
        self._items: Dict[str, Strategy] = {}

    def register(self, strategy: Strategy) -> None:
        self._items[strategy.name] = strategy

    def names(self) -> list[str]:
        return list(self._items.keys())

    def get(self, name: str) -> Strategy:
        try:
            return self._items[name]
        except KeyError as exc:
            available = ", ".join(self.names())
            raise KeyError(f"Unknown strategy '{name}'. Available: {available}") from exc

    def compute(self, name: str, df: pd.DataFrame, **params) -> pd.DataFrame:
        return self.get(name).compute(df, **params)


def build_default_registry(names: Iterable[str] | None = None) -> StrategyRegistry:
    """Register the current built-in dataframe strategies."""
    from ..strategy_core import STRATEGY_OPTIONS, compute_signals

    registry = StrategyRegistry()
    for name in names or STRATEGY_OPTIONS:
        registry.register(IndicatorStrategy(name=name, compute_fn=compute_signals))
    return registry
