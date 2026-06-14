"""
contracts.py
所有模块的 Protocol 契约定义（PLAN.md §5，M0 冻结）。

规则：
- 任何模块只 import 这里的 Protocol + trader/models.py 的数据模型。
- 不得 import 其他模块的实现细节。
- 契约变更须在 PLAN.md §10「契约变更记录」追加说明。
"""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    pass

from .models import (
    Advisory,
    Alert,
    Bar,
    Candidate,
    NewsEvent,
    Notification,
    Position,
    ReviewReport,
    RiskVerdict,
    Signal,
    TradePlan,
)


# ---------------------------------------------------------------------------
# 5.1  selection — 选股
# ---------------------------------------------------------------------------

@runtime_checkable
class Selector(Protocol):
    """对 universe 跑策略共识打分，输出按 score 降序的 Candidate 列表。"""
    def select(
        self,
        universe: List[str],
        timeframe: str,
        as_of: datetime,
    ) -> List[Candidate]: ...


# ---------------------------------------------------------------------------
# 5.2  plan — 交易计划
# ---------------------------------------------------------------------------

@runtime_checkable
class Planner(Protocol):
    """给定候选标的 + 最新 Bar，生成纪律化 TradePlan(DRAFT)。"""
    def make_plan(
        self,
        cand: Candidate,
        latest_bar: Bar,
        params: Dict[str, Any],
    ) -> TradePlan: ...


# ---------------------------------------------------------------------------
# 5.3  allocator — 仓位分配
# ---------------------------------------------------------------------------

@runtime_checkable
class Allocator(Protocol):
    """把 N 个 TradePlan 的 qty/target_weight 填好，满足总权重 ≤ 1 + 单标的上限。"""
    def allocate(
        self,
        plans: List[TradePlan],
        equity: float,
        positions: Dict[str, Position],
    ) -> List[TradePlan]: ...


# ---------------------------------------------------------------------------
# 5.4  portfolio_manager — 组合协调
# ---------------------------------------------------------------------------

@runtime_checkable
class PortfolioManager(Protocol):
    """相关性/集中度检查，可 pass-through（基础版）。"""
    def reconcile(
        self,
        plans: List[TradePlan],
        positions: Dict[str, Position],
    ) -> List[TradePlan]: ...


# ---------------------------------------------------------------------------
# 5.5  position_monitor — 盯盘守护
# ---------------------------------------------------------------------------

@runtime_checkable
class PositionMonitor(Protocol):
    """实时检查是否触发止损/止盈，返回需要处理的平仓/调整计划列表。"""
    def check(
        self,
        positions: Dict[str, Position],
        live_plans: Dict[str, TradePlan],
        latest: Dict[str, Bar],
    ) -> List[TradePlan]: ...


# ---------------------------------------------------------------------------
# 5.6  risk — 多层风控（扩展 risk_engine.py）
# （接口通过 RiskEngine 类方法扩展，此处只定义新增方法的 Protocol）
# ---------------------------------------------------------------------------

@runtime_checkable
class PlanRiskChecker(Protocol):
    """计划级风控审查（pre-trade + 止损/止盈合理性）。"""
    def evaluate_plan(
        self,
        plan: TradePlan,
        equity: float,
        positions: Dict[str, Position],
    ) -> RiskVerdict: ...

    def check_portfolio(
        self,
        positions: Dict[str, Position],
        equity: float,
    ) -> Optional[Alert]: ...


# ---------------------------------------------------------------------------
# 5.7  execution 接口已由 broker/base.py BrokerAdapter 承载
# OrderManager 强化在 order_manager.py 内部扩展，此处不重复
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 5.8  notify — 推送
# ---------------------------------------------------------------------------

@runtime_checkable
class Notifier(Protocol):
    """统一推送接口；实现：ConsoleNotifier | DiscordNotifier。"""
    def send(self, note: Notification) -> bool: ...


# ---------------------------------------------------------------------------
# 5.9  ai/agents — 多 agent 分析
# ---------------------------------------------------------------------------

class AgentContext:
    """传给每个 agent 的上下文数据包（agent 只读）。"""
    def __init__(
        self,
        candidates: Optional[List[Candidate]] = None,
        plans: Optional[List[TradePlan]] = None,
        news: Optional[List[NewsEvent]] = None,
        positions: Optional[Dict[str, Position]] = None,
        equity: float = 0.0,
        as_of: Optional[datetime] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.candidates = candidates or []
        self.plans = plans or []
        self.news = news or []
        self.positions = positions or {}
        self.equity = equity
        self.as_of = as_of
        self.extra = extra or {}


@runtime_checkable
class Agent(Protocol):
    """AI agent 角色接口。只产出 Advisory，绝不调用 broker/order_manager。"""
    role: str   # scout | sentiment | planner | risk_reviewer | reviewer | orchestrator

    def run(self, ctx: AgentContext) -> List[Advisory]: ...


# ---------------------------------------------------------------------------
# 5.10  news — 新闻/异动
# ---------------------------------------------------------------------------

@runtime_checkable
class NewsSource(Protocol):
    """轮询新闻/异动事件；实现：PriceMoveSource | NewsSourceStub。"""
    def poll(self, since: datetime) -> List[NewsEvent]: ...


# ---------------------------------------------------------------------------
# 5.11  review — 复盘归因
# ---------------------------------------------------------------------------

@runtime_checkable
class Reviewer(Protocol):
    """从账本/equity 数据生成盏后复盘报告。"""
    def review(self, period: str, as_of: datetime) -> ReviewReport: ...


# ---------------------------------------------------------------------------
# 5.12  watchdog + kill_switch
# ---------------------------------------------------------------------------

@runtime_checkable
class Watchdog(Protocol):
    """检查数据新鲜度/心跳/broker 连接/异常波动，返回 Alert 列表。"""
    def check(self) -> List[Alert]: ...


@runtime_checkable
class KillSwitch(Protocol):
    """急停开关；engage() 后 runtime 跳过所有执行。"""
    def engaged(self) -> bool: ...
    def engage(self, reason: str) -> None: ...
    def disengage(self) -> None: ...


# ---------------------------------------------------------------------------
# 5.13  universe + calendar
# ---------------------------------------------------------------------------

@runtime_checkable
class UniverseProvider(Protocol):
    """返回指定标的池名称对应的 symbol 列表。"""
    def get_universe(self, name: str) -> List[str]: ...


@runtime_checkable
class MarketCalendar(Protocol):
    """返回当前美股交易时段。"""
    def session_now(self) -> str: ...   # "pre" | "open" | "post" | "closed"


# ---------------------------------------------------------------------------
# 5.15  approval — 人在回路
# ---------------------------------------------------------------------------

@runtime_checkable
class Approver(Protocol):
    """判断 TradePlan 是否放行；默认 PENDING（不下单）。"""
    def decide(self, plan: TradePlan) -> str: ...  # APPROVED | REJECTED | PENDING
