# trader/ai — AI 旁路模块（agent 只产出 Advisory/TradePlan(DRAFT)，绝不下单）
from .manager import AgentManager, get_manager

__all__ = ["AgentManager", "get_manager"]
