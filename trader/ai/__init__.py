# trader/ai — AI 旁路模块（agent 只产出 Advisory/TradePlan(DRAFT)，绝不下单）
#
# get_manager()（模块级单例 + 调度入口）已删除：在整个代码库里已经零调用方，
# ai_advisories 表最后一条记录停留在很多天前——是一条"写者已经没了、读者
# 还在悄悄用过期数据"的孤儿依赖。AgentManager 类本身还在（还有测试覆盖，
# 需要的话可以手动实例化），只是不再有任何东西定时调度它。
from .manager import AgentManager

__all__ = ["AgentManager"]
