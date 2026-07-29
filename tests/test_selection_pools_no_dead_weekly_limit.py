"""rebuild_selection_pipeline() only ever built LONG_TERM and DAILY_DECISION
pools -- there is no weekly-pool build step in its body, and its one caller
(monitor_nice.py) never passed weekly_limit. It was a leftover parameter
from an earlier pipeline shape. Pin the signature so it doesn't quietly
reappear as dead surface area.
"""
import inspect

from trader.selection_pools import rebuild_selection_pipeline


def test_rebuild_selection_pipeline_has_no_weekly_limit_param():
    params = inspect.signature(rebuild_selection_pipeline).parameters
    assert "weekly_limit" not in params
