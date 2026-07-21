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
# override=True: .env 是本项目唯一权威配置源。不加 override 时
# python-dotenv 默认让已存在的 OS/用户级环境变量"赢"，曾导致一个误设的系统级
# FINNHUB_API_KEY（实际是粘错的 OpenAI key）静默覆盖 .env 里正确的值，且无任何报错提示。
load_dotenv(_ENV_PATH, override=True)


class Settings(BaseSettings):
    """Single typed source for .env-driven configuration."""
    model_config = SettingsConfigDict(env_file=str(_ENV_PATH), extra="ignore", case_sensitive=False)

    broker_type: str = Field("alpaca_paper", validation_alias=AliasChoices("BROKER_TYPE"))
    data_feed_type: str = Field("alpaca", validation_alias=AliasChoices("DATA_FEED_TYPE"))
    alpaca_api_key: str = Field("", validation_alias=AliasChoices("ALPACA_API_KEY"))
    alpaca_secret_key: str = Field(
        "", validation_alias=AliasChoices("ALPACA_API_SECRET", "ALPACA_SECRET_KEY"))
    alpaca_feed: str = Field(
        "sip", validation_alias=AliasChoices("ALPACA_DATA_FEED", "ALPACA_FEED"))


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

    # ---- AI 自动虚拟盘交易 -------------------------------------------------------
    # auto_trade_paper=True：
    #   · 从 ai_score_db 读取最新 AI 综合分（0-100）
    #   · 分数 ≥ min_ai_score 且确定性风控通过后下 LMT 单到 Alpaca paper
    #   · 分数 < min_ai_score 或 DB 无数据 → REJECTED（不执行）
    auto_trade_paper: bool = False
    min_ai_score: int = 65               # AI 综合分达标线（≥ 此分才下单）
    ai_score_db: str = "ai_states.duckdb"  # AI 评分来源（monitor 写，runtime 读）


    ai_score_max_age_minutes: float = 30.0
    paper_decision_enabled: bool = False
    paper_decision_shadow_mode: bool = True
    allow_quant_without_ai: bool = False
    strategy_statistics_path: str = "conf/strategy_statistics.json"
    universe_max_symbols: int = 20
    universe_max_age_minutes: int = 1440

    def model_post_init(self, __context: object) -> None:
        if self.auto_trade_paper and self.broker_type != 'alpaca_paper':
            raise ValueError('AUTO_TRADE_REQUIRES_ALPACA_PAPER')
