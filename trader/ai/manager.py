"""
trader/ai/manager.py
AgentManager: 调度 agent，写入 DuckDB，返回 Advisory 列表。

红线：只产出 Advisory；绝不调用 broker / order_manager / scheduler。

执行架构（双轨并行）：
  轨道 A（算法 agents，可并行）：quant / etf_flow / options / elite_holdings
  轨道 B（LLM agents，GPU 串行）：macro / fundamental / technical / news / web_research
  两轨同时跑，互不等待，结束后汇合。

  Phase 2（串行，依赖全部结果）：bull_bear

加权综合分（各 agent 独立输出 → 加权平均）：
  macro           25%
  fundamental     20%
  quant           15%
  options         12%
  etf_flow        10%
  elite_holdings  10%
  technical        5%
  news             2%
  web_research     1%

  bull_bear_debate 单独显示（综合裁判），不参与加权。
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

_AGENT_TIMEOUT = 900   # 单 agent 最长 15 分钟
_LLM_WORKERS   = 1     # LLM 串行（避免 GPU 争抢）
_ALGO_WORKERS  = 4     # 算法 agent 可并行

# 算法 agent（无 LLM，可并行）
_ALGO_ROLES = {"quant", "etf_flow", "options", "elite_holdings"}

# 加权系数（和 = 1.0），bull_bear_debate 单独展示
_AGENT_WEIGHTS: Dict[str, float] = {
    "macro":          0.25,
    "fundamental":    0.20,
    "quant":          0.15,
    "options":        0.12,
    "etf_flow":       0.10,
    "elite_holdings": 0.10,
    "technical":      0.05,
    "news":           0.02,
    "web_research":   0.01,
}

# 每个 kind 对应的分数字段
_SCORE_FIELD: Dict[str, str] = {
    "technical":      "technical_score",
    "news":           "news_score",
    "bull_bear_debate": "final_score",
    "web_research":   "hotspot_score",
    "macro":          "macro_score",
    "fundamental":    "fundamental_score",
    "quant":          "quant_score",
    "etf_flow":       "etf_score",
    "options":        "options_score",
    "elite_holdings": "elite_score",
}


class AgentManager:
    """
    统一 agent 调度层。
    - 双轨并行: algo（无 LLM）与 LLM agents 同时跑
    - 串行 Phase 2: bull_bear（依赖全部 Phase 1 结果）
    - 持久化: agent_states + ai_advisories → DuckDB
    """

    def __init__(
        self,
        agents: list | None = None,
        client=None,
        use_real_agents: bool = True,
    ) -> None:
        self._client = client
        if agents is not None:
            self._agents = agents
        elif use_real_agents:
            if self._client is None:
                from trader.ai.client import make_client
                self._client = make_client()
            self._agents = self._build_agents(self._client)
        else:
            from trader.ai.agents.base import StubAgent
            self._agents = [StubAgent("technical"), StubAgent("news"), StubAgent("bull_bear")]
        self._lock = threading.Lock()

    # ── 公共接口 ─────────────────────────────────────────────────────────────

    def run_cycle(self, ctx: AgentContext, db_path: str = "") -> List[Advisory]:
        """运行一轮：双轨并行 Phase 1 → bull_bear → 写 DuckDB → 返回全部 Advisory。"""
        if db_path:
            self._init_db(db_path)

        # 分轨
        algo_agents = [a for a in self._agents if a.role in _ALGO_ROLES]
        llm_agents  = [a for a in self._agents if a.role not in _ALGO_ROLES and a.role != "bull_bear"]
        bb_list     = [a for a in self._agents if a.role == "bull_bear"]

        all_advs:          List[Advisory] = []
        algo_advs:         List[Advisory] = []
        tech_advs:         List[Advisory] = []
        news_advs:         List[Advisory] = []
        # 其他 phase-1 结果分类，供 BullBear 使用
        macro_advs:        List[Advisory] = []
        fundamental_advs:  List[Advisory] = []
        quant_advs:        List[Advisory] = []
        etf_flow_advs:     List[Advisory] = []
        options_advs:      List[Advisory] = []
        elite_advs:        List[Advisory] = []

        # ── 轨道 A：算法 agents（并行） ────────────────────────────────────
        algo_errors: List[Exception] = []
        def _run_algo():
            with ThreadPoolExecutor(max_workers=_ALGO_WORKERS) as ex:
                futs = {ex.submit(self._run_one, a, ctx, db_path): a for a in algo_agents}
                for fut in as_completed(futs):
                    agent = futs[fut]
                    try:
                        results = fut.result(timeout=_AGENT_TIMEOUT)
                        algo_advs.extend(results)
                        if agent.role == "quant":
                            quant_advs.extend(results)
                        elif agent.role == "etf_flow":
                            etf_flow_advs.extend(results)
                        elif agent.role == "options":
                            options_advs.extend(results)
                        elif agent.role == "elite_holdings":
                            elite_advs.extend(results)
                    except Exception as exc:
                        logger.error("AgentManager [%s] failed: %s", agent.role, exc, exc_info=True)
                        algo_errors.append(exc)
                        if db_path:
                            self._write_state(db_path, agent.role, "error", 0.0, None, {"error": str(exc)})

        algo_thread = threading.Thread(target=_run_algo, daemon=True)
        algo_thread.start()

        # ── 轨道 B：LLM agents（GPU 串行） ─────────────────────────────────
        with ThreadPoolExecutor(max_workers=_LLM_WORKERS) as ex:
            futs = {ex.submit(self._run_one, a, ctx, db_path): a for a in llm_agents}
            for fut in as_completed(futs):
                agent = futs[fut]
                try:
                    results = fut.result(timeout=_AGENT_TIMEOUT)
                    all_advs.extend(results)
                    if agent.role == "technical":
                        tech_advs.extend(results)
                    elif agent.role == "news":
                        news_advs.extend(results)
                    elif agent.role == "macro":
                        macro_advs.extend(results)
                    elif agent.role == "fundamental":
                        fundamental_advs.extend(results)
                except Exception as exc:
                    logger.error("AgentManager [%s] failed: %s", agent.role, exc, exc_info=True)
                    if db_path:
                        self._write_state(db_path, agent.role, "error", 0.0, None, {"error": str(exc)})

        # 等待算法轨完成
        algo_thread.join(timeout=_AGENT_TIMEOUT)
        all_advs.extend(algo_advs)

        # ── Phase 2：BullBear（依赖全部 Phase 1 结果） ─────────────────────
        if bb_list:
            enriched_extra = dict(ctx.extra or {})
            enriched_extra.update({
                "technical_advisories":   tech_advs,
                "news_advisories":        news_advs,
                "macro_advisories":       macro_advs,
                "fundamental_advisories": fundamental_advs,
                "quant_advisories":       quant_advs,
                "etf_flow_advisories":    etf_flow_advs,
                "options_advisories":     options_advs,
                "elite_advisories":       elite_advs,
            })
            ectx = AgentContext(
                candidates=ctx.candidates, plans=ctx.plans, news=ctx.news,
                positions=ctx.positions, equity=ctx.equity, as_of=ctx.as_of,
                extra=enriched_extra,
            )
            for bb in bb_list:
                bb_results = self._run_one(bb, ectx, db_path)
                all_advs.extend(bb_results)

        if db_path and all_advs:
            self._write_advisories(all_advs, db_path)

        logger.info("AgentManager done: %d advisories, %d agents", len(all_advs), len(self._agents))
        return all_advs

    def get_agent_states(self, db_path: str) -> List[Dict[str, Any]]:
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
        """
        从最近 advisory 合成各 symbol 加权综合分。
        权重见 _AGENT_WEIGHTS；bull_bear_debate 单独附加，不参与加权。
        """
        advisories = self.get_recent_advisories(db_path, n=300)
        scores_by_sym: Dict[str, Dict] = {}

        for a in advisories:
            sym = a["payload"].get("symbol", "")
            if not sym:
                continue
            scores_by_sym.setdefault(sym, {})
            kind = a["kind"]
            p    = a["payload"]
            field = _SCORE_FIELD.get(kind)
            if field and field in p:
                score_val = p[field]
                # 同一 symbol + kind 取最新（advisories 已按 created_at DESC）
                if kind not in scores_by_sym[sym]:
                    scores_by_sym[sym][kind] = score_val
            # 附加 verdict
            if kind == "bull_bear_debate":
                scores_by_sym[sym]["verdict"] = p.get("verdict", "WATCHLIST")

        result = []
        for sym, sc in scores_by_sym.items():
            # 加权平均（只用有数据的 agent）
            total_w = sum(_AGENT_WEIGHTS.get(k, 0) for k in sc if k in _AGENT_WEIGHTS)
            if total_w > 0:
                weighted = sum(
                    sc[k] * _AGENT_WEIGHTS[k]
                    for k in sc
                    if k in _AGENT_WEIGHTS and isinstance(sc[k], (int, float))
                ) / total_w
            else:
                weighted = 50.0

            # 收集所有分数快照
            scores_snapshot = {k: v for k, v in sc.items() if isinstance(v, (int, float))}

            result.append({
                "symbol":          sym,
                "composite_score": round(weighted, 1),
                "verdict":         sc.get("verdict", "WATCHLIST"),
                "scores":          scores_snapshot,
                "weights_applied": {k: _AGENT_WEIGHTS[k] for k in scores_snapshot if k in _AGENT_WEIGHTS},
            })
        return sorted(result, key=lambda x: x["composite_score"], reverse=True)

    # ── 内部 ─────────────────────────────────────────────────────────────────

    def _run_one(self, agent, ctx: AgentContext, db_path: str) -> List[Advisory]:
        if db_path:
            self._write_state(db_path, agent.role, "running", 0.0, None, {})
        try:
            results = agent.run(ctx)
            score   = float(results[0].confidence * 100) if results else 0.0
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
                for column, sql_type in {
                    "run_id": "VARCHAR", "provider": "VARCHAR", "model": "VARCHAR",
                    "is_stub": "BOOLEAN", "source": "VARCHAR", "generated_by": "VARCHAR",
                    "schema_version": "INTEGER", "created_at_utc": "TIMESTAMPTZ",
                }.items():
                    con.execute(f"ALTER TABLE ai_advisories ADD COLUMN IF NOT EXISTS {column} {sql_type}")
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
                run_id = f"ai-run-{utc_now().strftime('%Y%m%dT%H%M%S%fZ')}"
                for adv in advisories:
                    provider = ("deterministic" if adv.agent in _ALGO_ROLES
                                else type(self._client).__name__.removesuffix("Client").lower())
                    model = adv.model or (f"{adv.agent}:v1" if provider == "deterministic"
                                          else getattr(self._client, "_model", ""))
                    is_stub = type(self._client).__name__ == "StubLLMClient"
                    con.execute(
                        "INSERT OR IGNORE INTO ai_advisories "
                        "(advisory_id, kind, agent, payload_json, confidence, created_at, "
                        "run_id, provider, model, is_stub, source, generated_by, schema_version, created_at_utc) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        [adv.advisory_id, adv.kind, adv.agent,
                         json.dumps(adv.payload, default=str), adv.confidence, adv.created_at,
                         run_id, "stub" if is_stub else provider, model, is_stub,
                         "agent_manager", adv.agent, 1, adv.created_at],
                    )
                con.commit()
                con.close()
            except Exception as exc:
                logger.warning("AgentManager: write_advisories failed: %s", exc)

    @staticmethod
    def _build_agents(client=None) -> list:
        from trader.ai.client import make_client
        from trader.ai.agents import (
            TechnicalAgent, NewsAgent, BullBearDebate, WebResearchAgent,
            MacroAgent, FundamentalAgent, QuantAgent, ETFFlowAgent,
            OptionsAgent, EliteHoldingsAgent,
        )
        c = client or make_client()
        return [
            # ── 算法轨（无 LLM，并行）──────────────────────
            QuantAgent(),
            ETFFlowAgent(),
            OptionsAgent(),
            EliteHoldingsAgent(),
            # ── LLM 轨（GPU 串行）──────────────────────────
            MacroAgent(client=c),
            FundamentalAgent(client=c),
            TechnicalAgent(client=c),
            NewsAgent(client=c),
            WebResearchAgent(client=c, max_symbols=3),
            # ── Phase 2（等全部 Phase 1 完成）──────────────
            BullBearDebate(client=c, min_score=55.0, max_symbols=3),
        ]


def get_composite_scores_from_db(db_path: str, limit: int = 300) -> Dict[str, float]:
    """
    直接从 ai_states.duckdb 读取 AI 综合加权分，不需要 AgentManager 实例。

    用于 runtime.py 在自动交易模式下读取 monitor 写入的最新 AI 评分。
    返回：{symbol: composite_score}，score 范围 0-100。
    无数据或读取失败时返回空字典。
    """
    try:
        import duckdb
        con = duckdb.connect(db_path, read_only=True)
        rows = con.execute(
            "SELECT advisory_id, kind, agent, payload_json, confidence, created_at "
            "FROM ai_advisories ORDER BY created_at DESC LIMIT ?",
            [limit],
        ).fetchall()
        con.close()
    except Exception as exc:
        logger.debug("get_composite_scores_from_db: %s", exc)
        return {}

    scores_by_sym: Dict[str, Dict] = {}
    # 只用 kind + payload_json；advisory_id/agent/confidence/created_at 在这里不需要
    # （composite score 用的是下面按 agent 类型的静态权重 _AGENT_WEIGHTS，不是逐条 confidence）。
    for _advisory_id, kind, _agent, payload_json, _confidence, _created_at in rows:
        try:
            p = json.loads(payload_json or "{}")
        except Exception:
            continue
        sym = p.get("symbol", "")
        if not sym:
            continue
        scores_by_sym.setdefault(sym, {})
        field = _SCORE_FIELD.get(kind)
        if field and field in p and kind not in scores_by_sym[sym]:
            scores_by_sym[sym][kind] = p[field]

    result: Dict[str, float] = {}
    for sym, sc in scores_by_sym.items():
        total_w = sum(_AGENT_WEIGHTS.get(k, 0.0) for k in sc if k in _AGENT_WEIGHTS)
        if total_w > 0:
            weighted = sum(
                sc[k] * _AGENT_WEIGHTS[k]
                for k in sc
                if k in _AGENT_WEIGHTS and isinstance(sc[k], (int, float))
            ) / total_w
        else:
            weighted = 50.0
        result[sym] = round(weighted, 1)
    return result


# 模块级单例
_manager: Optional[AgentManager] = None


def get_manager(use_real_agents: bool = True) -> AgentManager:
    global _manager
    if _manager is None:
        _manager = AgentManager(use_real_agents=use_real_agents)
    return _manager


def get_score_snapshots_from_db(db_path: str, limit: int = 300):
    """Read composite scores with verifiable provenance; legacy rows remain incomplete."""
    from .safety import AIScoreSnapshot
    try:
        import duckdb
        con = duckdb.connect(db_path, read_only=True)
        columns = {row[1] for row in con.execute("PRAGMA table_info('ai_advisories')").fetchall()}
        provenance_columns = {"run_id", "provider", "model", "is_stub", "source", "generated_by", "created_at_utc"}
        if provenance_columns <= columns:
            rows = con.execute(
                "SELECT kind, payload_json, confidence, created_at_utc, run_id, provider, model, is_stub, source, generated_by "
                "FROM ai_advisories ORDER BY created_at_utc DESC LIMIT ?", [limit]
            ).fetchall()
        else:
            rows = [(*row, None, None, None, False, None, None) for row in con.execute(
                "SELECT kind, payload_json, confidence, created_at FROM ai_advisories ORDER BY created_at DESC LIMIT ?", [limit]
            ).fetchall()]
        con.close()
    except Exception as exc:
        logger.debug("get_score_snapshots_from_db: %s", exc)
        return {}

    grouped: Dict[str, Dict[str, tuple]] = {}
    for row in rows:
        kind, payload_json, confidence, created_at, run_id, provider, model, is_stub, source, generated_by = row
        try:
            payload = json.loads(payload_json or "{}")
        except Exception:
            continue
        symbol = payload.get("symbol")
        field = _SCORE_FIELD.get(kind)
        score = payload.get(field) if field else None
        if not symbol or not isinstance(score, (int, float)):
            continue
        grouped.setdefault(symbol, {}).setdefault(kind, (score, created_at, run_id, provider, model, bool(is_stub), source, generated_by))

    snapshots = {}
    for symbol, values in grouped.items():
        weighted_values = [(value, _AGENT_WEIGHTS[kind]) for kind, value in values.items() if kind in _AGENT_WEIGHTS]
        total_weight = sum(weight for _, weight in weighted_values)
        if not total_weight:
            continue
        score = sum(value[0] * weight for value, weight in weighted_values) / total_weight
        contributors = [
            {"agent_name": kind, "score": value[0], "created_at": value[1], "provider": value[3],
             "model": value[4], "is_stub": value[5]} for kind, value in values.items()
        ]
        newest = max(values.values(), key=lambda value: value[1])
        providers = {value[3] for value in values.values() if value[3]}
        models = {value[4] for value in values.values() if value[4]}
        run_ids = {value[2] for value in values.values() if value[2]}
        snapshots[symbol] = AIScoreSnapshot(
            symbol=symbol, score=round(score, 1), created_at=newest[1],
            run_id=run_ids.pop() if len(run_ids) == 1 else None,
            provider=",".join(sorted(providers)) or None,
            model=",".join(sorted(models)) or None,
            source=newest[6], generated_by=newest[7],
            is_stub=any(value[5] for value in values.values()),
            contributors=contributors,
        )
    return snapshots
