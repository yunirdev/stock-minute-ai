"""
trader/exploration.py
Canonical import path for the exploration/backtest panel inside the trader package.
Delegates to app/exploration.py which holds the full implementation.
"""
from app.exploration import render_exploration_tab  # noqa: F401
