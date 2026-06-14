# trader/ai/agents — 多 agent 角色
from .base import AgentBase, StubAgent
from .orchestrator import OrchestratorAgent
from .technical import TechnicalAgent
from .news import NewsAgent
from .bull_bear import BullBearDebate
from .web_research import WebResearchAgent

__all__ = [
    "AgentBase", "StubAgent",
    "OrchestratorAgent",
    "TechnicalAgent",
    "NewsAgent",
    "BullBearDebate",
    "WebResearchAgent",
]
