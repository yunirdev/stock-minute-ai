# trader/ai/agents — 多 agent 角色
from .base import AgentBase, StubAgent
from .technical import TechnicalAgent
from .news import NewsAgent
from .bull_bear import BullBearDebate
from .web_research import WebResearchAgent
from .macro import MacroAgent
from .fundamental import FundamentalAgent
from .quant import QuantAgent
from .etf_flow import ETFFlowAgent
from .options import OptionsAgent
from .elite_holdings import EliteHoldingsAgent

__all__ = [
    "AgentBase", "StubAgent",
    "TechnicalAgent",
    "NewsAgent",
    "BullBearDebate",
    "WebResearchAgent",
    "MacroAgent",
    "FundamentalAgent",
    "QuantAgent",
    "ETFFlowAgent",
    "OptionsAgent",
    "EliteHoldingsAgent",
]
