"""
Persistent UI preferences — survives Streamlit app restarts.

Call load_prefs(st.session_state) once at startup (before widgets are rendered)
to restore saved values.  Call save_prefs(st.session_state) after widgets are
rendered so the latest selections are written to disk.
"""

import json
from pathlib import Path
from typing import Any

PREFS_PATH = Path("conf/ui_settings.json")

# All session_state keys that should be persisted.
PREF_KEYS: list[str] = [
    # monitor.py sidebar
    "cfg_broker_type",
    "cfg_symbols",
    "cfg_strategies",
    "cfg_tf",
    "cfg_interval",
    "cfg_capital",
    "cfg_data_feed",
    "cfg_hours",
    # 行情图 tab (inside @st.fragment)
    "chart_sym",
    "chart_tf",
    # exploration.py inline controls
    "exp_initial_capital",
    "exp_symbols",
    "exp_strat_cat",
    "exp_tf",
    "exp_xmode",
    "exp_day_sep",
    "exp_leverage",
    "exp_autoscale",
    "exp_window",
    "exp_ind_list",
    "exp_sma_n",
    "exp_ema_fast",
    "exp_ema_slow",
    "exp_bb_n",
    "exp_bb_k",
    "exp_rsi_n",
    "exp_macd_f",
    "exp_macd_s",
    "exp_macd_sg",
    "exp_atr_n",
    "exp_ta_n",
    "exp_strategy",
    "exp_kdj_n",
    "exp_cci_n",
    "exp_dc_n",
    "exp_week_n",
    "exp_bbi_eps",
    "exp_bbi_bo",
    "exp_bbi_feps",
]


def load_prefs(ss: Any) -> None:
    """
    Read preferences from disk and inject into *ss* (st.session_state) for
    every key that is not already present.  Safe to call on every rerun —
    the guard ``key not in ss`` prevents overwriting values the user has
    already changed during the current session.
    """
    if not PREFS_PATH.exists():
        return
    try:
        data: dict = json.loads(PREFS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return
    for key, val in data.items():
        if key in PREF_KEYS and key not in ss:
            ss[key] = val


def save_prefs(ss: Any) -> None:
    """Write all tracked session_state keys that exist to the prefs file."""
    data = {k: _serialisable(ss[k]) for k in PREF_KEYS if k in ss}
    try:
        PREFS_PATH.parent.mkdir(parents=True, exist_ok=True)
        PREFS_PATH.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _serialisable(v: Any) -> Any:
    """Convert types that are not JSON-serialisable to plain Python types."""
    if isinstance(v, (list, tuple)):
        return [_serialisable(i) for i in v]
    if isinstance(v, dict):
        return {str(k): _serialisable(val) for k, val in v.items()}
    # numpy / pandas scalars
    try:
        import numpy as np  # type: ignore
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
    except ImportError:
        pass
    return v
