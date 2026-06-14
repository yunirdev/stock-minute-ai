"""
config.py
Typed configuration.

Two layers, on purpose:
  - Settings (pydantic-settings): the SINGLE typed source for .env-driven values
    (secrets, feed selection). Only the keys declared here are read from .env —
    unrelated keys like SYMBOLS are ignored, so the engine universe stays under
    explicit CLI/code control.
  - RiskConfig / TradingConfig (pydantic models): validated runtime config,
    constructed explicitly (CLI args in main.py). Their .env-derived defaults
    pull from the Settings instance.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from dotenv import load_dotenv
from pydantic import AliasChoices, BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
# Keep python-dotenv loading too, so plain os.getenv(...) elsewhere still works.
load_dotenv(_ENV_PATH)


class Settings(BaseSettings):
    """Single typed source for .env-driven configuration."""
    model_config = SettingsConfigDict(env_file=str(_ENV_PATH), extra="ignore", case_sensitive=False)

    broker_type: str = Field("alpaca_paper", validation_alias=AliasChoices("BROKER_TYPE"))
    data_feed_type: str = Field("alpaca", validation_alias=AliasChoices("DATA_FEED_TYPE"))
    alpaca_api_key: str = Field("", validation_alias=AliasChoices("ALPACA_API_KEY"))
    alpaca_secret_key: str = Field(
        "", validation_alias=AliasChoices("ALPACA_API_SECRET", "ALPACA_SECRET_KEY"))
    alpaca_feed: str = Field(
        "iex", validation_alias=AliasChoices("ALPACA_DATA_FEED", "ALPACA_FEED"))


# Loaded once at import — the one place .env is parsed for engine config.
settings = Settings()


class RiskConfig(BaseModel):
    """Pre-trade and real-time risk limits."""

    max_position_pct: float = 0.20          # single symbol max fraction of capital
    max_trade_risk_pct: float = 0.005       # max capital risk per trade (stop ~1%)
    daily_drawdown_limit_pct: float = 0.03  # halt if day PnL <= -3%
    max_consecutive_failures: int = 3       # halt after N consecutive order errors
    allow_short: bool = False               # long-only by default


class TradingConfig(BaseModel):
    """Master configuration for the trading platform."""

    # ---- Trading universe (CLI/code-controlled, NOT auto-read from .env) -----
    symbols: List[str] = Field(default_factory=lambda: ["AAPL"])
    strategies: List[str] = Field(default_factory=lambda: ["5/20均线金叉死叉"])
    timeframe: str = "5m"                    # "1m" | "5m" | "30m" | "1h" | "1d"

    # ---- Research/backtest capital -----------------------------------------
    initial_capital: float = 10_000.0
    leverage: float = 1.0
    order_type: str = "LMT"                  # "LMT" | "MKT"

    # ---- Mode ---------------------------------------------------------------
    broker_type: str = Field(default_factory=lambda: settings.broker_type)

    # ---- Risk ---------------------------------------------------------------
    risk: RiskConfig = Field(default_factory=RiskConfig)

    # ---- Data feed ----------------------------------------------------------
    data_feed_type: str = Field(default_factory=lambda: settings.data_feed_type)
    alpaca_api_key: str = Field(default_factory=lambda: settings.alpaca_api_key)
    alpaca_secret_key: str = Field(default_factory=lambda: settings.alpaca_secret_key)
    alpaca_feed: str = Field(default_factory=lambda: settings.alpaca_feed)

    # ---- Storage ------------------------------------------------------------
    db_path: str = "trade.duckdb"

    # ---- Scheduler ----------------------------------------------------------
    poll_interval_secs: int = 30
    bars_lookback: int = 120                 # bars fetched per tick for indicator warm-up

    # ---- Pending-order / gap-fill -------------------------------------------
    pending_order_max_bars: int = 10

    # ---- Execution gate (红线：默认关；只挂 LMT；需人工确认后再开) -----------
    execution_enabled: bool = False
