"""
test_m0_contracts.py
M0 契约层验收测试。

覆盖：
- 所有新数据模型可 import + 实例化
- contracts.py 中所有 Protocol 可 import
- 各 stub 模块实例化 + 主路径 + 空数据/异常
"""
from __future__ import annotations

import pytest
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# 1. 新数据模型（models.py M0 扩展）
# ---------------------------------------------------------------------------

class TestNewModels:
    def test_candidate_instantiation(self, sample_candidate):
        from trader.models import Candidate
        assert isinstance(sample_candidate, Candidate)
        assert sample_candidate.score == 70.0
        assert sample_candidate.rank == 1

    def test_trade_plan_instantiation(self, sample_trade_plan):
        from trader.models import TradePlan
        assert isinstance(sample_trade_plan, TradePlan)
        assert sample_trade_plan.status == "DRAFT"
        assert sample_trade_plan.entry_price == 191.0

    def test_advisory_instantiation(self):
        from trader.models import Advisory, new_id
        adv = Advisory(advisory_id=new_id(), kind="plan", agent="scout",
                       payload={"test": 1})
        assert adv.agent == "scout"

    def test_news_event_instantiation(self, sample_news_event):
        from trader.models import NewsEvent
        assert isinstance(sample_news_event, NewsEvent)
        assert sample_news_event.kind == "price_move"

    def test_review_report_instantiation(self):
        from trader.models import ReviewReport, new_id
        r = ReviewReport(report_id=new_id(), period="daily",
                         market_summary="ok", portfolio_pnl=100.0,
                         attribution={})
        assert r.portfolio_pnl == 100.0

    def test_alert_instantiation(self, sample_alert):
        from trader.models import Alert
        assert isinstance(sample_alert, Alert)

    def test_notification_instantiation(self, sample_notification):
        from trader.models import Notification
        assert isinstance(sample_notification, Notification)


# ---------------------------------------------------------------------------
# 2. contracts.py —— 所有 Protocol 可 import
# ---------------------------------------------------------------------------

class TestContractsImport:
    def test_all_protocols_importable(self):
        from trader.contracts import (
            Selector, Planner, Allocator, PortfolioManager,
            PositionMonitor, PlanRiskChecker, Notifier,
            Agent, AgentContext,
            NewsSource, Reviewer, Watchdog, KillSwitch,
            UniverseProvider, MarketCalendar, Approver,
        )

    def test_agent_context_creation(self, agent_context):
        from trader.contracts import AgentContext
        assert isinstance(agent_context, AgentContext)
        assert len(agent_context.candidates) == 1


# ---------------------------------------------------------------------------
# 3. notify
# ---------------------------------------------------------------------------

class TestNotify:
    def test_console_notifier_always_succeeds(self, console_notifier, sample_notification):
        result = console_notifier.send(sample_notification)
        assert result is True

    def test_discord_notifier_degrades_without_url(self, sample_notification):
        from trader.notify import DiscordNotifier
        n = DiscordNotifier(webhook_url="")
        result = n.send(sample_notification)  # should fall back to console
        assert result is True


# ---------------------------------------------------------------------------
# 4. allocator
# ---------------------------------------------------------------------------

class TestAllocator:
    def test_single_plan_allocation(self, equal_weight_allocator,
                                    sample_trade_plan, sample_positions):
        plans = [sample_trade_plan]
        result = equal_weight_allocator.allocate(plans, equity=100_000.0,
                                                  positions=sample_positions)
        assert len(result) == 1
        assert 0 < result[0].target_weight <= 0.25
        assert result[0].qty > 0

    def test_total_weight_not_exceed_one(self, equal_weight_allocator,
                                          sample_trade_plan, sample_positions):
        from trader.models import TradePlan, Side, new_id
        plans = [
            TradePlan(plan_id=new_id(), symbol=f"SYM{i}", side=Side.BUY,
                      action="OPEN", entry_price=100.0, stop_loss=95.0,
                      take_profit=110.0, confidence=0.8)
            for i in range(8)
        ]
        result = equal_weight_allocator.allocate(plans, equity=100_000.0,
                                                  positions={})
        total = sum(p.target_weight for p in result)
        assert total <= 1.0 + 1e-6

    def test_empty_plans_returns_empty(self, equal_weight_allocator, sample_positions):
        result = equal_weight_allocator.allocate([], 100_000.0, sample_positions)
        assert result == []


# ---------------------------------------------------------------------------
# 5. plan (ATRPlanner)
# ---------------------------------------------------------------------------

class TestATRPlanner:
    def test_plan_has_valid_prices(self, atr_planner, sample_candidate, sample_bar):
        plan = atr_planner.make_plan(sample_candidate, sample_bar, {})
        # 多头：stop < entry < tp
        assert plan.stop_loss < plan.entry_price < plan.take_profit
        assert plan.status == "DRAFT"
        assert plan.rationale != ""

    def test_plan_symbol_matches_candidate(self, atr_planner, sample_candidate, sample_bar):
        plan = atr_planner.make_plan(sample_candidate, sample_bar, {})
        assert plan.symbol == sample_candidate.symbol


# ---------------------------------------------------------------------------
# 6. approval
# ---------------------------------------------------------------------------

class TestApproval:
    def test_pending_by_default(self, auto_approver_pending, sample_trade_plan):
        result = auto_approver_pending.decide(sample_trade_plan)
        assert result == "PENDING"

    def test_auto_approve_high_confidence(self, auto_approver_auto, sample_trade_plan):
        sample_trade_plan.confidence = 0.8
        result = auto_approver_auto.decide(sample_trade_plan)
        assert result == "APPROVED"

    def test_auto_reject_low_confidence(self, auto_approver_auto, sample_trade_plan):
        sample_trade_plan.confidence = 0.3
        result = auto_approver_auto.decide(sample_trade_plan)
        assert result == "REJECTED"


# ---------------------------------------------------------------------------
# 7. watchdog / kill_switch
# ---------------------------------------------------------------------------

class TestKillSwitch:
    def test_not_engaged_by_default(self, file_kill_switch):
        assert file_kill_switch.engaged() is False

    def test_engage_and_disengage(self, file_kill_switch):
        file_kill_switch.engage("test reason")
        assert file_kill_switch.engaged() is True
        file_kill_switch.disengage()
        assert file_kill_switch.engaged() is False


# ---------------------------------------------------------------------------
# 8. universe + market_calendar
# ---------------------------------------------------------------------------

class TestUniverse:
    def test_default_returns_list(self):
        from trader.universe import get_universe
        syms = get_universe("default")
        assert isinstance(syms, list)
        assert len(syms) > 0

    def test_mega_cap_universe(self):
        from trader.universe import get_universe
        syms = get_universe("mega_cap")
        assert "AAPL" in syms


class TestMarketCalendar:
    def test_session_returns_valid_value(self):
        from trader.market_calendar import session_now
        result = session_now()
        assert result in ("pre", "open", "post", "closed")

    def test_weekend_is_closed(self):
        from trader.market_calendar import session_at
        from datetime import datetime, timezone
        # 2026-06-14 is a Sunday
        sunday_utc = datetime(2026, 6, 14, 15, 0, tzinfo=timezone.utc)
        assert session_at(sunday_utc) == "closed"

    def test_weekday_market_hours_is_open(self):
        from trader.market_calendar import session_at
        # 2026-06-15 (Monday) 14:30 UTC = 09:30 ET = market open
        monday_open_utc = datetime(2026, 6, 15, 14, 30, tzinfo=timezone.utc)
        assert session_at(monday_open_utc) == "open"


# ---------------------------------------------------------------------------
# 9. portfolio_manager + position_monitor
# ---------------------------------------------------------------------------

class TestPortfolioManager:
    def test_passthrough_returns_same_plans(self, sample_trade_plan, sample_positions):
        from trader.portfolio_manager import PassthroughPortfolioManager
        pm = PassthroughPortfolioManager()
        plans = [sample_trade_plan]
        result = pm.reconcile(plans, sample_positions)
        assert result is plans


class TestPositionMonitor:
    def test_stop_loss_triggers_close_plan(self, sample_bar, sample_position,
                                            sample_trade_plan):
        from trader.position_monitor import StopTakeProfitMonitor
        monitor = StopTakeProfitMonitor()
        # 设置 bar 价格跌破止损
        sample_bar.close = 185.0  # below stop_loss=188.0
        result = monitor.check(
            positions={"AAPL": sample_position},
            live_plans={"AAPL": sample_trade_plan},
            latest={"AAPL": sample_bar},
        )
        assert len(result) == 1
        assert result[0].action == "CLOSE"
        assert result[0].status == "APPROVED"

    def test_no_trigger_in_range(self, sample_bar, sample_position, sample_trade_plan):
        from trader.position_monitor import StopTakeProfitMonitor
        monitor = StopTakeProfitMonitor()
        sample_bar.close = 192.0  # between stop=188 and tp=197
        result = monitor.check(
            positions={"AAPL": sample_position},
            live_plans={"AAPL": sample_trade_plan},
            latest={"AAPL": sample_bar},
        )
        assert result == []


# ---------------------------------------------------------------------------
# 10. ai/agents — orchestrator + stub
# ---------------------------------------------------------------------------

class TestAgents:
    def test_orchestrator_returns_advisory(self, orchestrator, agent_context):
        result = orchestrator.run(agent_context)
        assert len(result) >= 1
        # summary advisory 始终是第一个，agent 为 "orchestrator"
        assert result[0].agent == "orchestrator"

    def test_stub_agent_returns_empty(self, agent_context):
        from trader.ai.agents.base import StubAgent
        stub = StubAgent("scout")
        result = stub.run(agent_context)
        assert result == []

    def _import_lines(self, src: str):
        """只提取真实的 import 语句行（跳过注释和文档字符串）。"""
        in_docstring = False
        lines = []
        for line in src.splitlines():
            stripped = line.strip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                in_docstring = not in_docstring
                continue
            if in_docstring:
                continue
            if stripped.startswith("#"):
                continue
            if stripped.startswith("import ") or stripped.startswith("from "):
                lines.append(stripped)
        return lines

    def test_agent_base_does_not_import_broker(self):
        import inspect
        import trader.ai.agents.base as base_mod
        imports = self._import_lines(inspect.getsource(base_mod))
        for line in imports:
            assert "broker" not in line, f"agent base imports broker: {line}"
            assert "order_manager" not in line, f"agent base imports order_manager: {line}"

    def test_orchestrator_does_not_import_broker_or_scheduler(self):
        import inspect
        import trader.ai.agents.orchestrator as orch_mod
        imports = self._import_lines(inspect.getsource(orch_mod))
        for line in imports:
            assert "broker" not in line, f"orchestrator imports broker: {line}"
            assert "scheduler" not in line, f"orchestrator imports scheduler: {line}"
