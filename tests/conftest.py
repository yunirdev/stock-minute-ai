"""
conftest.py
M0 公共测试 fixtures —— 提供数据模型和 stub 模块实例。
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict

import pandas as pd
import pytest

from trader.models import (
    Advisory,
    Alert,
    Bar,
    Candidate,
    Fill,
    NewsEvent,
    Notification,
    Position,
    ReviewReport,
    Side,
    Signal,
    TradePlan,
    new_id,
    utc_now,
)


# ---------------------------------------------------------------------------
# 基础数据
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_bar() -> Bar:
    return Bar(
        symbol="AAPL",
        timestamp=datetime(2026, 6, 14, 14, 30, tzinfo=timezone.utc),
        open=190.0, high=192.0, low=188.0, close=191.0, volume=1_000_000,
    )


@pytest.fixture
def sample_candidate() -> Candidate:
    return Candidate(
        symbol="AAPL",
        score=70.0,
        rank=1,
        reasons={"votes": {"MACD零轴战法": 1, "RSI震荡战法(60买40卖)": -1},
                 "total_strategies": 2},
        as_of=utc_now(),
    )


@pytest.fixture
def sample_trade_plan() -> TradePlan:
    return TradePlan(
        plan_id=new_id(),
        symbol="AAPL",
        side=Side.BUY,
        action="OPEN",
        entry_price=191.0,
        stop_loss=188.0,
        take_profit=197.0,
        confidence=0.7,
        rationale="test plan",
        status="DRAFT",
    )


@pytest.fixture
def sample_position() -> Position:
    return Position(symbol="AAPL", qty=10.0, avg_entry_px=190.0)


@pytest.fixture
def sample_positions(sample_position) -> Dict[str, Position]:
    return {"AAPL": sample_position}


@pytest.fixture
def sample_news_event() -> NewsEvent:
    return NewsEvent(
        event_id=new_id(),
        kind="price_move",
        symbol="AAPL",
        title="AAPL 上涨 5%",
        severity=0.5,
        source="price_move",
    )


@pytest.fixture
def sample_notification() -> Notification:
    return Notification(
        title="测试推送",
        body="这是一条测试消息",
        kind="info",
    )


@pytest.fixture
def sample_alert() -> Alert:
    return Alert(level="warn", source="test", message="测试告警")


# ---------------------------------------------------------------------------
# Stub 模块实例
# ---------------------------------------------------------------------------

@pytest.fixture
def console_notifier():
    from trader.notify import ConsoleNotifier
    return ConsoleNotifier()


@pytest.fixture
def equal_weight_allocator():
    from trader.allocator import EqualWeightAllocator
    return EqualWeightAllocator(max_position_pct=0.25)


@pytest.fixture
def atr_planner():
    from trader.plan import ATRPlanner
    return ATRPlanner()


@pytest.fixture
def auto_approver_pending():
    from trader.approval import AutoApprover
    return AutoApprover(auto_approve=False)


@pytest.fixture
def auto_approver_auto():
    from trader.approval import AutoApprover
    return AutoApprover(auto_approve=True, min_confidence=0.5)


@pytest.fixture
def file_kill_switch():
    import os
    import tempfile
    from pathlib import Path
    from trader.watchdog import FileKillSwitch
    d = tempfile.mkdtemp()
    ks = FileKillSwitch(path=Path(d) / "kill_switch.json")
    yield ks
    p = Path(d) / "kill_switch.json"
    if p.exists():
        p.unlink()
    try:
        os.rmdir(d)
    except OSError:
        pass


@pytest.fixture
def orchestrator():
    from trader.ai.agents.orchestrator import OrchestratorAgent
    # 测试用 stub 模式，不需要 Ollama 在线
    return OrchestratorAgent(use_real_agents=False)


@pytest.fixture
def agent_context(sample_candidate, sample_trade_plan, sample_news_event,
                  sample_positions) -> "AgentContext":
    from trader.contracts import AgentContext
    return AgentContext(
        candidates=[sample_candidate],
        plans=[sample_trade_plan],
        news=[sample_news_event],
        positions=sample_positions,
        equity=100_000.0,
        as_of=utc_now(),
    )


# ---------------------------------------------------------------------------
# 通用 DataFrame fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df(n: int = 140) -> pd.DataFrame:
    rows = []
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(n):
        close = 100.0 + i * 0.1
        rows.append({
            "timestamp_utc": base + pd.Timedelta(minutes=i),
            "open": close - 0.2,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 1000 + i,
        })
    return pd.DataFrame(rows)
