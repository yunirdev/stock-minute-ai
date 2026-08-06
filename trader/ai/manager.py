"""
trader/ai/manager.py
AgentManager: 调度 agent，写入 DuckDB，返回 Advisory 列表。

红线：只产出 Advisory；绝不调用 broker / order execution。

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

from trader.models import AgentContext
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
        if algo_thread.is_alive():
            # 超时后主线程会继续往下走，而算法轨还在后台改 algo_advs —— 综合分
            # 会少算几个 agent（权重重新归一化，分数含义就变了）。静默降级不
            # 可接受，至少要留痕。
            logger.error(
                "AgentManager: 算法轨超时（%ds）仍未完成，本轮综合分缺少 algo agent 贡献",
                _AGENT_TIMEOUT,
            )
        all_advs.extend(list(algo_advs))

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
            if total_w <= 0:
                # 之前这里是 weighted = 50.0 —— 一个没有任何参与加权的 agent
                # 产出的标的（比如只有 bull_bear_debate，而它不参与加权）会拿到
                # 一个凭空捏造的 50 分"中性"综合分，跟真实算出来的 50 分完全
                # 无法区分。宁可不给分，也不给假分。
                logger.warning(
                    "composite score: %s 没有任何参与加权的 agent 分数，跳过（不再伪造 50 分）",
                    sym,
                )
                continue
            weighted = sum(
                sc[k] * _AGENT_WEIGHTS[k]
                for k in sc
                if k in _AGENT_WEIGHTS and isinstance(sc[k], (int, float))
            ) / total_w

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

    _LEGACY_TS_SUFFIX = "__legacy_naive_ts"

    @staticmethod
    def _table_columns(con, table: str) -> Dict[str, str]:
        return {
            row[0]: row[1]
            for row in con.execute(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_name = ?",
                [table],
            ).fetchall()
        }

    def _needs_ts_migration(self, con, table: str, ts_columns: List[str]) -> bool:
        cols = self._table_columns(con, table)
        if not cols:
            return False   # 表还不存在，等下按新 schema 建即可
        return any(
            cols.get(c) is not None and cols[c] != "TIMESTAMP WITH TIME ZONE"
            for c in ts_columns
        )

    def _rename_legacy(self, con, table: str) -> None:
        con.execute(f"ALTER TABLE {table} RENAME TO {table}{self._LEGACY_TS_SUFFIX}")

    def _copy_from_legacy(self, con, table: str, ts_columns: List[str]) -> None:
        """把旧表数据搬回新表：时间列按写入时的本地时区还原成正确 UTC 时刻，
        其余列**原样保留**。列名逐个对齐，旧表没有的列填 NULL —— 早期版本
        少了 provenance 列，写死列清单会把已有的 run_id/provider 等值清空。"""
        legacy = f"{table}{self._LEGACY_TS_SUFFIX}"
        legacy_cols = self._table_columns(con, legacy)
        new_cols = self._table_columns(con, table)
        select_parts, insert_parts = [], []
        for col in new_cols:
            insert_parts.append(col)
            if col not in legacy_cols:
                select_parts.append("NULL")
            elif col in ts_columns:
                select_parts.append(f"timezone(current_setting('TimeZone'), {col})")
            else:
                select_parts.append(col)
        con.execute(
            f"INSERT INTO {table} ({', '.join(insert_parts)}) "
            f"SELECT {', '.join(select_parts)} FROM {legacy}"
        )
        moved = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        con.execute(f"DROP TABLE {legacy}")
        logger.info(
            "AgentManager: %s 已迁移为 TIMESTAMPTZ，保留 %d 行（含全部 provenance 列）",
            table, moved,
        )

    def _init_db(self, db_path: str) -> None:
        with self._lock:
            try:
                import duckdb
                con = duckdb.connect(db_path)
                # 这几列必须是 TIMESTAMPTZ：DuckDB 把带时区的 Python datetime
                # 写进 naive TIMESTAMP 列时会静默转成本地墙钟时间并丢掉时区，
                # 读回来再当 UTC 用就会凭空差出一个时区偏移（audit.py 的
                # heartbeat 表就是这么产生了每 tick 一次的假 CRITICAL 告警）。
                # 已有的旧表原地迁移，按写入时的本地时区还原成正确的 UTC 时刻。
                states_ts = ["last_run", "updated_at"]
                states_legacy = self._needs_ts_migration(con, "agent_states", states_ts)
                if states_legacy:
                    self._rename_legacy(con, "agent_states")
                con.execute("""
                    CREATE TABLE IF NOT EXISTS agent_states (
                        role        VARCHAR PRIMARY KEY,
                        status      VARCHAR DEFAULT 'idle',
                        last_score  FLOAT DEFAULT 0,
                        last_run    TIMESTAMPTZ,
                        summary_json VARCHAR DEFAULT '{}',
                        updated_at  TIMESTAMPTZ
                    )
                """)
                if states_legacy:
                    self._copy_from_legacy(con, "agent_states", states_ts)

                adv_ts = ["created_at"]
                adv_legacy = self._needs_ts_migration(con, "ai_advisories", adv_ts)
                if adv_legacy:
                    self._rename_legacy(con, "ai_advisories")
                con.execute("""
                    CREATE TABLE IF NOT EXISTS ai_advisories (
                        advisory_id VARCHAR PRIMARY KEY,
                        kind        VARCHAR,
                        agent       VARCHAR,
                        payload_json VARCHAR,
                        confidence  FLOAT,
                        created_at  TIMESTAMPTZ
                    )
                """)
                # provenance 列必须在搬数据之前补齐，否则旧表里已有的
                # run_id/provider/is_stub 等值会因为新表还没有这些列而丢失。
                for column, sql_type in {
                    "run_id": "VARCHAR", "provider": "VARCHAR", "model": "VARCHAR",
                    "is_stub": "BOOLEAN", "source": "VARCHAR", "generated_by": "VARCHAR",
                    "schema_version": "INTEGER", "created_at_utc": "TIMESTAMPTZ",
                    "is_fallback": "BOOLEAN",
                }.items():
                    con.execute(f"ALTER TABLE ai_advisories ADD COLUMN IF NOT EXISTS {column} {sql_type}")
                if adv_legacy:
                    self._copy_from_legacy(con, "ai_advisories", adv_ts)
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
                client_name = type(self._client).__name__
                # 客户端缺失时必须 fail closed。原来的写法在 self._client is None
                # 时会算出 provider="nonetype"、is_stub=False —— 而 AIScoreValidator
                # 只检查 provider 非空且 is_stub 为 False，于是一条来源不明的
                # advisory 会顶着一个假的 provider 名字通过 AI 安全门。
                client_missing = self._client is None
                is_stub_client = client_name == "StubLLMClient"
                for adv in advisories:
                    if adv.agent in _ALGO_ROLES:
                        provider = "deterministic"          # 算法 agent 不用 LLM
                    elif client_missing:
                        # 如实标 unknown，而不是笼统写成 "stub"——两者的排查
                        # 方向完全不同（一个是没配客户端，一个是明确用了 stub）
                        provider = "unknown"
                    elif is_stub_client:
                        provider = "stub"
                    else:
                        provider = client_name.removesuffix("Client").lower()
                    model = adv.model or (f"{adv.agent}:v1" if provider == "deterministic"
                                          else getattr(self._client, "_model", ""))
                    # LLM agent 但客户端来源不明 → 也按 stub 处理，让安全门拒掉
                    is_stub = is_stub_client or (
                        client_missing and adv.agent not in _ALGO_ROLES
                    )
                    con.execute(
                        "INSERT OR IGNORE INTO ai_advisories "
                        "(advisory_id, kind, agent, payload_json, confidence, created_at, "
                        "run_id, provider, model, is_stub, source, generated_by, schema_version, created_at_utc, is_fallback) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        [adv.advisory_id, adv.kind, adv.agent,
                         json.dumps(adv.payload, default=str), adv.confidence, adv.created_at,
                         run_id, provider, model, is_stub,
                         "agent_manager", adv.agent, 1, adv.created_at, adv.is_fallback],
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
        if total_w <= 0:
            # 同 get_composite_scores：不给分好过给一个伪造的中性 50 分。
            # 这个函数喂的是 daily_candidates / selection_pools 的候选筛选，
            # 假分会直接把标的送进 AI 深度研究名单。
            logger.warning(
                "composite score: %s 没有任何参与加权的 agent 分数，跳过（不再伪造 50 分）",
                sym,
            )
            continue
        weighted = sum(
            sc[k] * _AGENT_WEIGHTS[k]
            for k in sc
            if k in _AGENT_WEIGHTS and isinstance(sc[k], (int, float))
        ) / total_w
        result[sym] = round(weighted, 1)
    return result


def get_score_snapshots_from_db(db_path: str, limit: int = 300):
    """Read the newest single-run composite for each symbol.

    Legacy rows without verifiable UTC timestamps/run IDs are ignored instead of
    being allowed to invalidate otherwise current symbols.
    """
    from .safety import AIScoreSnapshot
    try:
        import duckdb
        con = duckdb.connect(db_path, read_only=True)
        columns = {row[1] for row in con.execute("PRAGMA table_info('ai_advisories')").fetchall()}
        provenance_columns = {"run_id", "provider", "model", "is_stub", "source", "generated_by", "created_at_utc"}
        if not provenance_columns <= columns:
            con.close()
            return {}
        fallback_expr = "COALESCE(is_fallback, FALSE)" if "is_fallback" in columns else "FALSE"
        rows = con.execute(
            "SELECT kind, payload_json, confidence, created_at_utc, run_id, provider, "
            f"model, is_stub, source, generated_by, {fallback_expr} "
            "FROM ai_advisories "
            "WHERE created_at_utc IS NOT NULL AND run_id IS NOT NULL "
            "ORDER BY created_at_utc DESC LIMIT ?",
            [limit],
        ).fetchall()
        con.close()
    except Exception as exc:
        logger.debug("get_score_snapshots_from_db: %s", exc)
        return {}

    grouped: Dict[str, Dict[str, Dict[str, tuple]]] = {}
    for row in rows:
        kind, payload_json, _confidence, created_at, run_id, provider, model, is_stub, source, generated_by, is_fallback = row
        try:
            payload = json.loads(payload_json or "{}")
        except Exception:
            continue
        symbol = payload.get("symbol")
        field = _SCORE_FIELD.get(kind)
        score = payload.get(field) if field else None
        if not symbol or not run_id or created_at is None or not isinstance(score, (int, float)):
            continue
        run_values = grouped.setdefault(symbol, {}).setdefault(run_id, {})
        run_values.setdefault(
            kind,
            (score, created_at, provider, model, bool(is_stub), source, generated_by, bool(is_fallback)),
        )

    snapshots = {}
    for symbol, runs in grouped.items():
        run_id, values = max(
            runs.items(),
            key=lambda item: max(value[1] for value in item[1].values()),
        )
        weighted_values = [
            (kind, value, _AGENT_WEIGHTS[kind])
            for kind, value in values.items()
            if kind in _AGENT_WEIGHTS and not value[7]
        ]
        total_weight = sum(weight for _kind, _value, weight in weighted_values)
        if not total_weight:
            continue
        score = sum(value[0] * weight for _kind, value, weight in weighted_values) / total_weight
        contributors = [
            {"agent_name": kind, "score": value[0], "created_at": value[1], "provider": value[2],
             "model": value[3], "is_stub": value[4], "is_fallback": value[7]}
            for kind, value in values.items()
        ]
        newest = max((value for _kind, value, _weight in weighted_values), key=lambda value: value[1])
        providers = {value[2] for _kind, value, _weight in weighted_values if value[2]}
        models = {value[3] for _kind, value, _weight in weighted_values if value[3]}
        has_llm = any(kind not in _ALGO_ROLES for kind, _value, _weight in weighted_values)
        snapshots[symbol] = AIScoreSnapshot(
            symbol=symbol, score=round(score, 1), created_at=newest[1],
            run_id=run_id,
            provider=",".join(sorted(providers)) or None,
            model=",".join(sorted(models)) or None,
            source=newest[5], generated_by=newest[6],
            is_stub=any(value[4] for _kind, value, _weight in weighted_values),
            contributors=contributors,
            contributor_count=len(weighted_values),
            weight_coverage=round(total_weight, 4),
            has_llm=has_llm,
            fallback_count=sum(1 for value in values.values() if value[7]),
            # 这条 score 是多个 agent 分数的加权平均，从来不是 daily_research 那种
            # "50=中性、双极编码方向"的量表——直接除以 100 当方向无关的信心强度，
            # 不能套 conviction_of() 对双极分数做的 abs(score-50)/50 反变换（那套
            # 变换是 daily_research._score() 专属的，套在这里数值会算错）。
            confidence=round(score, 1) / 100.0,
        )
    return snapshots
