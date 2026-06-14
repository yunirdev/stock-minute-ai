"""
broker/base.py
Abstract interface that all broker adapters must implement.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from ..models import Fill, OrderIntent, OrderStatus, Position


class BrokerAdapter(ABC):
    """
    Minimal interface for order execution adapters.

    Concrete implementations: AlpacaBroker (and future broker adapters).
    """

    @abstractmethod
    def place_order(self, intent: OrderIntent) -> str:
        """
        Submit the order to the broker.

        Returns
        -------
        broker_order_id : str
            Broker-assigned order identifier.

        Raises
        ------
        Exception on hard failure (network error, rejection before ack).
        """

    @abstractmethod
    def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel an open order. Returns True if cancellation was accepted."""

    @abstractmethod
    def get_order_status(self, broker_order_id: str) -> OrderStatus:
        """Query the current status of an order."""

    @abstractmethod
    def get_fill(self, broker_order_id: str) -> Optional[Fill]:
        """Return fill details once an order is filled/partially filled."""

    @abstractmethod
    def get_positions(self) -> List[Position]:
        """Return broker-side open positions (used for reconciliation)."""

    @abstractmethod
    def get_account_equity(self) -> float:
        """Return broker-reported net liquidation value."""
