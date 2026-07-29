"""Whitelisted .env writer for NiceGUI-editable operational/scheduling settings.

Deliberately excludes anything that touches trading authority: broker_type,
auto_trade_paper, Alpaca credentials, and risk thresholds stay read-only in
the UI and are not writable through this module.
"""
from __future__ import annotations

from typing import Any

from dotenv import set_key

from .config import _ENV_PATH

WRITABLE_ENV_KEYS: frozenset[str] = frozenset(
    {
        "OLLAMA_MODEL",
        "LLM_PROVIDER",
        "DAILY_RESEARCH_CLOSE_HOUR_ET",
        "DAILY_RESEARCH_CLOSE_MINUTE_ET",
        "MORNING_BRIEF_HOUR_ET",
        "DAILY_REVIEW_HOUR_ET",
        "AGENT_CYCLE_INTERVAL_MINUTES",
        "MIN_AI_SCORE",
        "AI_SCORE_MAX_AGE_MINUTES",
        "AI_MIN_CONTRIBUTORS",
        "AI_MIN_WEIGHT_COVERAGE",
        "DAILY_RESEARCH_SCREEN_LIMIT",
        "DAILY_RESEARCH_DEEP_LIMIT",
        "UNIVERSE_MAX_SYMBOLS",
        "UNIVERSE_MAX_AGE_MINUTES",
    }
)


class EnvKeyNotWritableError(ValueError):
    """Raised when a caller tries to write a key outside WRITABLE_ENV_KEYS."""


class EnvValueRequiredError(ValueError):
    """Raised when a caller tries to write None/empty as a value.

    Without this guard, a caller passing None (e.g. a cleared UI number
    field) would silently write the literal string "None" into .env, which
    then fails pydantic-settings parsing for the next app start.
    """


def write_env_setting(key: str, value: Any) -> None:
    """Persist one operational setting to .env, in place, preserving other lines.

    Rejects any key not in WRITABLE_ENV_KEYS (trading authority, broker, and
    risk keys included) so this module can never become a second way to
    change trading permissions from the UI.
    """
    normalized_key = key.strip().upper()
    if normalized_key not in WRITABLE_ENV_KEYS:
        raise EnvKeyNotWritableError(normalized_key)
    if value is None or (isinstance(value, str) and not value.strip()):
        raise EnvValueRequiredError(normalized_key)
    set_key(str(_ENV_PATH), normalized_key, str(value))
