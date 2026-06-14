"""
trader/ai/agents/orchestrator.py
Orchestrator：调度 technical/news/bull_bear 等 sub-agent，
聚合产出最终 Advisory 列表，供 runtime 使用。

流程（每轮）：
  1. TechnicalAgent  → 各 symbol 技术打分
  2. NewsAgent       → 各 symbol 新闻情绪
  3. BullBearDebate  → top candidates 辩论 + 裁决
  4. orchestrator    → 汇总，产出综合 summary Advisory

红线：只产出 Advisory；不调用 broker / order_manager / scheduler。
"""
from __future__ import annotations

import logging
from typing import List

from trader.contracts import AgentContext
from trader.models import Advisory, new_id, utc_now
from .base import AgentBase, StubAgent

logger = logging.getLogger(__name__)


class OrchestratorAgent(AgentBase):
    """汇总各子 agent 的 Advisory，产出综合建议。"""

    role = "orchestrator"

    def __init__(
        self,
        sub_agents: List[AgentBase] | None = None,
        client=None,
        use_real_agents: bool = True,
    ) -> None:
        """
        use_real_agents=True：自动实例化 TechnicalAgent + NewsAgent + BullBearDebate。
        use_real_agents=False：全部 stub（离线/测试用）。
        sub_agents：手动注入 agent 列表（覆盖 use_real_agents）。
        """
        if sub_agents is not None:
            self._agents = sub_agents
        elif use_real_agents:
            self._agents = self._build_real_agents(client)
        else:
            self._agents = [
                StubAgent("technical"),
                StubAgent("news"),
                StubAgent("bull_bear"),
            ]

    # ── 公共接口 ─────────────────────────────────────────────────────────────

    def run(self, ctx: AgentContext) -> List[Advisory]:
        all_advisories: List[Advisory] = []
        tech_advs: List[Advisory] = []
        news_advs: List[Advisory] = []

        for agent in self._agents:
            try:
                results = agent.run(ctx)
                all_advisories.extend(results)
                # 把 technical/news advisory 存入 ctx.extra，供 bull_bear 使用
                if agent.role == "technical":
                    tech_advs = results
                elif agent.role == "news":
                    news_advs = results
                elif agent.role == "bull_bear" and results:
                    # bull_bear 需要前两步结果，注入到 ctx.extra 后再运行
                    pass
                logger.info(
                    "orchestrator: [%s] → %d advisory",
                    agent.role, len(results),
                )
            except Exception as exc:
                logger.error("orchestrator: agent[%s] 失败: %s", agent.role, exc, exc_info=True)

        # 如果有 bull_bear agent，给它注入 tech/news 结果后重跑
        bb_agents = [a for a in self._agents if a.role == "bull_bear"]
        if bb_agents and (tech_advs or news_advs):
            enriched_extra = dict(ctx.extra or {})
            enriched_extra["technical_advisories"] = tech_advs
            enriched_extra["news_advisories"] = news_advs
            enriched_ctx = AgentContext(
                candidates=ctx.candidates,
                plans=ctx.plans,
                news=ctx.news,
                positions=ctx.positions,
                equity=ctx.equity,
                as_of=ctx.as_of,
                extra=enriched_extra,
            )
            # 移除前面跑过的 bull_bear 结果（重跑更准确）
            all_advisories = [a for a in all_advisories if a.agent != "bull_bear"]
            for bb in bb_agents:
                try:
                    bb_results = bb.run(enriched_ctx)
                    all_advisories.extend(bb_results)
                    logger.info("orchestrator: [bull_bear] re-run → %d advisory", len(bb_results))
                except Exception as exc:
                    logger.error("orchestrator: bull_bear re-run 失败: %s", exc)

        summary = self._summarize(all_advisories, ctx)
        logger.info(
            "orchestrator 完成: %d sub-advisories, %d candidates analyzed",
            len(all_advisories), len(ctx.candidates),
        )
        return [summary] + all_advisories

    # ── 私有 ─────────────────────────────────────────────────────────────────

    def _summarize(self, advisories: List[Advisory], ctx: AgentContext) -> Advisory:
        # 提取各 symbol 的综合得分
        scores_by_symbol: dict = {}
        verdicts_by_symbol: dict = {}

        for adv in advisories:
            sym = adv.payload.get("symbol", "")
            if not sym:
                continue
            kind = adv.kind
            if kind == "technical":
                scores_by_symbol.setdefault(sym, {})["technical"] = adv.payload.get("technical_score", 50)
            elif kind == "news":
                scores_by_symbol.setdefault(sym, {})["news"] = adv.payload.get("news_score", 50)
            elif kind == "bull_bear_debate":
                scores_by_symbol.setdefault(sym, {})["debate"] = adv.payload.get("final_score", 50)
                verdicts_by_symbol[sym] = adv.payload.get("verdict", "WATCHLIST")

        # 合成最终推荐
        recommendations = []
        for sym, scores in scores_by_symbol.items():
            avg = sum(scores.values()) / len(scores) if scores else 50
            recommendations.append({
                "symbol": sym,
                "composite_score": round(avg, 1),
                "verdict": verdicts_by_symbol.get(sym, "WATCHLIST"),
                "scores": scores,
            })
        recommendations.sort(key=lambda x: x["composite_score"], reverse=True)

        return Advisory(
            advisory_id=new_id(),
            kind="orchestrator_summary",
            agent=self.role,
            payload={
                "sub_advisory_count": len(advisories),
                "candidate_count": len(ctx.candidates),
                "recommendations": recommendations,
                "top_pick": recommendations[0] if recommendations else None,
            },
            confidence=0.6 if recommendations else 0.0,
            model="orchestrator_v1",
            created_at=utc_now(),
        )

    @staticmethod
    def _build_real_agents(client=None, use_web_research: bool = True) -> List[AgentBase]:
        from trader.ai.client import make_client
        from .technical import TechnicalAgent
        from .news import NewsAgent
        from .bull_bear import BullBearDebate
        from .web_research import WebResearchAgent
        c = client or make_client()
        agents: List[AgentBase] = [
            TechnicalAgent(client=c),
            NewsAgent(client=c),
        ]
        if use_web_research:
            agents.append(WebResearchAgent(client=c, max_symbols=3))
        agents.append(BullBearDebate(client=c, min_score=55.0, max_symbols=3))
        return agents
