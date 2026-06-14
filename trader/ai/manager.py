"""
trader/ai/manager.py
AgentManager: 并行调度 agent，写入 DuckDB，返回 Advisory 列表。

红线：只产出 Advisory；绝不调用 broker / order_manager / scheduler。

流程：
  1. ThreadPoolExecutor 并行运行 technical / news / web_research
  2. 串行运行 bull_bear（需要前两步结果作为上下文）
  3. 写入 DuckDB: agent_states + ai_advisories 两张表
  4. 合成各 symbol 综合分，供 UI 展示
"""
from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional

from trader.contracts import AgentContext
from trader.models import Advisory, utc_now

logger = logging.getLogger(__name__)

_AGENT_TIMEOUT = 120
_MAX_WORKERS = 4


class AgentManager:
    """
    统一 agent 调度层。
    - 并行: technical / news / web_research (ThreadPoolExecutor)
    - 串行: bull_bear（依赖前面结果）
    - 持久化: agent_states + ai_advisories → DuckDB
    """

    def __init__(
        self,
        agents: list | None = None,
        client=None,
        use_real_agents: bool = True,
    ) -> None:
        if agents is not None:
            self._agents = agents
        elif use_real_agents:
            self._agents = self._build_agents(client)
        else:
            from trader.ai.agents.base import StubAgent
            self._agents = [StubAgent("technical"), StubAgent("news"), StubAgent("bull_bear")]
        self._lock = threading.Lock()

    # ── 公共接口 ─────────────────────────────────────────────────────────────

    def run_cycle(self, ctx: AgentContext, db_path: str = "") -> List[Advisory]:
        """运行一轮：并行 → bull_bear → 写 DuckDB → 返回所有 Advisory。"""
        if db_path:
            self._init_db(db_path)

        parallel = [a for a in self._agents if a.role != "bull_bear"]
        bb_list = [a for a in self._agents if a.role == "bull_bear"]

        all_advs: List[Advisory] = []
        tech_advs: List[Advisory] = []
        news_advs: List[Advisory] = []

        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
            futures = {ex.submit(self._run_one, a, ctx, db_path): a for a in parallel}
            for fut in as_completed(futures):
                agent = futures[fut]
                try:
                    results = fut.result(timeout=_AGENT_TIMEOUT)
                    all_advs.extend(results)
                    if agent.role == "technical":
                        tech_advs = results
                    elif agent.role == "news":
                        news_advs = results
                except Exception as exc:
                    logger.error("AgentManager [%s] failed: %s", agent.role, exc, exc_info=True)
                    if db_path:
                        self._write_state(db_path, agent.role, "error", 0.0, None, {"error": str(exc)})

        if bb_list:
            enriched_extra = dict(ctx.extra or {})
            enriched_extra["technical_advisories"] = tech_advs
            enriched_extra["news_advisories"] = news_advs
            ectx = AgentContext(
                candidates=ctx.candidates, plans=ctx.plans, news=ctx.news,
                positions=ctx.positions, equity=ctx.equity, as_of=ctx.as_of,
                extra=enriched_extra,
            )
            for bb in bb_list:
                all_advs.extend(self._run_one(bb, ectx, db_path))

        if db_path and all_advs:
            self._write_advisories(all_advs, db_path)

        logger.info("AgentManager done: %d advisories, %d agents", len(all_advs), len(self._agents))
        return all_advs

    def get_agent_states(self, db_path: str) -> List[Dict[str, Any]]:
        """读取各 agent 最近状态（供 UI 展示）。"""
        with self._lock:
            try:
                import duckdb
                con = duckdb.connect(db_path)
                rows = con.execute(
                    "SELECT role, status, last_score, last_run, summary_json, updated_at "
                    "FROM agent_states ORDER BY role"
                ).fetchall()
                con.close()
                return [
                    {"role": r[0], "status": r[1], "last_score": r[2],
                     "last_run": r[3], "summary": json.loads(r[4] or "{}"), "updated_at": r[5]}
                    for r in rows
                ]
            except Exception:
                return []

    def get_recent_advisories(self, db_path: str, n: int = 30) -> List[Dict[str, Any]]:
        """读取最近 n 条 Advisory（供 UI 活动流）。"""
        with self._lock:
            try:
                import duckdb
                con = duckdb.connect(db_path)
                rows = con.execute(
                    "SELECT advisory_id, kind, agent, payload_json, confidence, created_at "
                    "FROM ai_advisories ORDER BY created_at DESC LIMIT ?", [n]
                ).fetchall()
                con.close()
                return [
                    {"advisory_id": r[0], "kind": r[1], "agent": r[2],
                     "payload": json.loads(r[3] or "{}"), "confidence": r[4], "created_at": r[5]}
                    for r in rows
                ]
            except Exception:
                return []

    def get_composite_scores(self, db_path: str) -> List[Dict[str, Any]]:
        """从最近 advisory 合成各 symbol 综合分（供 Manager 决策区）。"""
        advisories = self.get_recent_advisories(db_path, n=100)
        scores_by_sym: Dict[str, Dict] = {}
        for a in advisories:
            sym = a["payload"].get("symbol", "")
            if not sym:
                continue
            scores_by_sym.setdefault(sym, {})
            k = a["kind"]
            p = a["payload"]
            if k == "technical":
                scores_by_sym[sym]["technical"] = p.get("technical_score", 50)
            elif k == "news":
                scores_by_sym[sym]["news"] = p.get("news_score", 50)
            elif k == "bull_bear_debate":
                scores_by_sym[sym]["debate"] = p.get("final_score", 50)
                scores_by_sym[sym]["verdict"] = p.get("verdict", "WATCHLIST")
            elif k == "web_research":
                scores_by_sym[sym]["web"] = p.get("hotspot_score", 50)

        result = []
        for sym, sc in scores_by_sym.items():
            nums = {k: v for k, v in sc.items() if isinstance(v, (int, float))}
            avg = sum(nums.values()) / len(nums) if nums else 50.0
            result.append({
                "symbol": sym,
                "composite_score": round(avg, 1),
                "verdict": sc.get("verdict", "WATCHLIST"),
                "scores": nums,
            })
        return sorted(result, key=lambda x: x["composite_score"], reverse=True)

    # ── 内部 ─────────────────────────────────────────────────────────────────

    def _run_one(self, agent, ctx: AgentContext, db_path: str) -> List[Advisory]:
        if db_path:
            self._write_state(db_path, agent.role, "running", 0.0, None, {})
        try:
            results = agent.run(ctx)
            score = float(results[0].confidence * 100) if results else 0.0
            summary = results[0].payload if results else {}
            if db_path:
                self._write_state(db_path, agent.role, "done", score, utc_now(), summary)
            logger.info("AgentManager [%s] → %d advisory", agent.role, len(results))
            return results
        except Exception as exc:
            logger.error("AgentManager [%s]: %s", agent.role, exc, exc_info=True)
            if db_path:
                self._write_state(db_path, agent.role, "error", 0.0, None, {"error": str(exc)})
            return []

    def _init_db(self, db_path: str) -> None:
        with self._lock:
            try:
                import duckdb
                con = duckdb.connect(db_path)
                con.execute("""
                    CREATE TABLE IF NOT EXISTS agent_states (
                        role        VARCHAR PRIMARY KEY,
                        status      VARCHAR DEFAULT 'idle',
                        last_score  FLOAT DEFAULT 0,
                        last_run    TIMESTAMP,
                        summary_json VARCHAR DEFAULT '{}',
                        updated_at  TIMESTAMP
                    )
                """)
                con.execute("""
                    CREATE TABLE IF NOT EXISTS ai_advisories (
                        advisory_id VARCHAR PRIMARY KEY,
                        kind        VARCHAR,
                        agent       VARCHAR,
                        payload_json VARCHAR,
                        confidence  FLOAT,
                        created_at  TIMESTAMP
                    )
                """)
                con.commit()
                con.close()
            except Exception as exc:
                logger.error("AgentManager: init_db failed: %s", exc)

    def _write_state(
        self, db_path: str, role: str, status: str,
        score: float, last_run: Optional[datetime], summary: Dict[str, Any],
    ) -> None:
        with self._lock:
            try:
                import duckdb
                con = duckdb.connect(db_path)
                con.execute(
                    "INSERT OR REPLACE INTO agent_states "
                    "(role, status, last_score, last_run, summary_json, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    [role, status, score, last_run, json.dumps(summary, default=str), utc_now()],
                )
                con.commit()
                con.close()
            except Exception as exc:
                logger.warning("AgentManager: write_state failed: %s", exc)

    def _write_advisories(self, advisories: List[Advisory], db_path: str) -> None:
        with self._lock:
            try:
                import duckdb
                con = duckdb.connect(db_path)
                for adv in advisories:
                    con.execute(
                        "INSERT OR IGNORE INTO ai_advisories "
                        "(advisory_id, kind, agent, payload_json, confidence, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        [adv.advisory_id, adv.kind, adv.agent,
                         json.dumps(adv.payload, default=str), adv.confidence, adv.created_at],
                    )
                con.commit()
                con.close()
            except Exception as exc:
                logger.warning("AgentManager: write_advisories failed: %s", exc)

    @staticmethod
    def _build_agents(client=None) -> list:
        from trader.ai.client import make_client
        from trader.ai.agents import TechnicalAgent, NewsAgent, BullBearDebate, WebResearchAgent
        c = client or make_client()
        return [
            TechnicalAgent(client=c),
            NewsAgent(client=c),
            WebResearchAgent(client=c, max_symbols=3),
            BullBearDebate(client=c, min_score=55.0, max_symbols=3),
        ]


# 模块级单例，避免重复初始化 LLM 客户端
_manager: Optional[AgentManager] = None


def get_manager(use_real_agents: bool = True) -> AgentManager:
    global _manager
    if _manager is None:
        _manager = AgentManager(use_real_agents=use_real_agents)
    return _manager
