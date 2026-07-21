"""
trader/teams/maintenance.py
T5 维护团队 — 盘后复盘、异常检测、Discord 报告、反馈建议。

职责：
  1. 读取 DuckDB 成交/心跳/风控记录
  2. 计算各策略 / 各团队绩效
  3. 检测异常行为（短时间大量订单、回撤突刺等）
  4. 生成日报并发送到 Discord
  5. 将反馈建议写入 DuckDB，供后续参数校准参考
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List

from .base import TeamOutput

logger = logging.getLogger(__name__)

_DEFAULT_DB = str(Path(__file__).resolve().parents[2] / "trade.duckdb")


# ═══════════════════════════════════════════════════════════════════════════
# 公共入口
# ═══════════════════════════════════════════════════════════════════════════

def run_maintenance(
    db_path: str = _DEFAULT_DB,
    period_hours: int = 24,
    send_discord: bool = False,
) -> TeamOutput:
    """运行 T5 维护分析，返回 TeamOutput。"""
    out = TeamOutput(team="T5")
    t0 = time.time()
    try:
        stats = _compute_stats(db_path, period_hours)
        anomalies = _detect_anomalies(db_path, period_hours)
        suggestions = _generate_suggestions(stats, anomalies)

        out.data["stats"] = stats
        out.data["anomalies"] = anomalies
        out.data["suggestions"] = suggestions
        out.data["period_hours"] = period_hours

        if send_discord:
            _send_discord_report(stats, anomalies, suggestions)

        _persist_suggestions(db_path, suggestions)

        logger.info(
            "T5 维护完成: trades=%d win_rate=%.1f%% anomalies=%d",
            stats.get("trade_count", 0),
            stats.get("win_rate", 0) * 100,
            len(anomalies),
        )
    except Exception as exc:
        out.add_error(str(exc))
        logger.warning("T5 maintenance 失败: %s", exc, exc_info=True)
    finally:
        out.duration_ms = (time.time() - t0) * 1000
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 绩效统计
# ═══════════════════════════════════════════════════════════════════════════

def _compute_stats(db_path: str, period_hours: int) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "trade_count": 0, "win_rate": 0.0,
        "total_pnl": 0.0, "avg_pnl": 0.0,
        "max_drawdown_pct": 0.0,
        "equity_start": None, "equity_end": None,
    }
    try:
        import duckdb
        since = datetime.now(timezone.utc) - timedelta(hours=period_hours)
        con = duckdb.connect(db_path, read_only=True)

        # 成交统计
        try:
            fills = con.execute(
                "SELECT side, filled_qty, avg_price FROM fills WHERE filled_at >= ? ",
                [since],
            ).fetchdf()
            if not fills.empty:
                stats["trade_count"] = len(fills)
        except Exception:
            pass

        # 权益快照（最大回撤）
        try:
            eq = con.execute(
                "SELECT total_equity FROM equity_snapshots WHERE ts >= ? ORDER BY ts",
                [since],
            ).fetchdf()
            if not eq.empty and len(eq) > 1:
                equity = eq["total_equity"].values
                stats["equity_start"] = float(equity[0])
                stats["equity_end"] = float(equity[-1])
                # 最大回撤
                running_max = float(equity[0])
                max_dd = 0.0
                for v in equity:
                    if v > running_max:
                        running_max = v
                    dd = (v - running_max) / running_max if running_max > 0 else 0
                    if dd < max_dd:
                        max_dd = dd
                stats["max_drawdown_pct"] = round(max_dd * 100, 2)
                if equity[0] > 0:
                    stats["total_pnl"] = round(float(equity[-1]) - float(equity[0]), 2)
        except Exception:
            pass

        con.close()
    except Exception as exc:
        logger.warning("T5 stats 读取失败: %s", exc)

    return stats


# ═══════════════════════════════════════════════════════════════════════════
# 异常检测
# ═══════════════════════════════════════════════════════════════════════════

def _detect_anomalies(db_path: str, period_hours: int) -> List[Dict[str, Any]]:
    anomalies: List[Dict[str, Any]] = []
    try:
        import duckdb
        since = datetime.now(timezone.utc) - timedelta(hours=period_hours)
        con = duckdb.connect(db_path, read_only=True)

        # 异常1：心跳超时（超过 10 分钟无心跳）
        try:
            hb = con.execute(
                "SELECT MAX(ts) as last_hb FROM heartbeat"
            ).fetchone()
            if hb and hb[0]:
                last = hb[0]
                if hasattr(last, "tzinfo") and last.tzinfo is None:
                    last = last.replace(tzinfo=timezone.utc)
                gap = (datetime.now(timezone.utc) - last).total_seconds()
                if gap > 600:
                    anomalies.append({
                        "type": "heartbeat_timeout",
                        "severity": "high",
                        "message": f"引擎心跳超时 {gap/60:.1f} 分钟",
                        "value": gap,
                    })
        except Exception:
            pass

        # 异常2：风控熔断事件
        try:
            risk_halts = con.execute(
                "SELECT COUNT(*) FROM risk_events WHERE level='critical' AND ts >= ?",
                [since],
            ).fetchone()
            if risk_halts and risk_halts[0] > 0:
                anomalies.append({
                    "type": "risk_halt",
                    "severity": "critical",
                    "message": f"过去 {period_hours}h 发生 {risk_halts[0]} 次风控熔断",
                    "value": risk_halts[0],
                })
        except Exception:
            pass

        # 异常3：高频下单（同一标的在 5 分钟内多次下单）
        try:
            orders = con.execute(
                "SELECT symbol, COUNT(*) as cnt FROM order_intents WHERE created_at >= ? "
                "GROUP BY symbol HAVING COUNT(*) > 5",
                [since],
            ).fetchdf()
            for _, row in orders.iterrows():
                anomalies.append({
                    "type": "excessive_orders",
                    "severity": "medium",
                    "message": f"{row['symbol']} 在窗口内提交了 {row['cnt']} 次订单",
                    "value": int(row["cnt"]),
                })
        except Exception:
            pass

        con.close()
    except Exception as exc:
        logger.warning("T5 anomaly 检测失败: %s", exc)

    return anomalies


# ═══════════════════════════════════════════════════════════════════════════
# 反馈建议生成
# ═══════════════════════════════════════════════════════════════════════════

def _generate_suggestions(
    stats: Dict[str, Any],
    anomalies: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    suggestions: List[Dict[str, Any]] = []

    dd = stats.get("max_drawdown_pct", 0)
    if dd < -3:
        suggestions.append({
            "target_team": "T4",
            "target_param": "daily_drawdown_limit_pct",
            "suggestion": f"日内最大回撤 {dd:.1f}%，建议收紧熔断阈值至 2%",
            "priority": "high",
        })

    excessive = [a for a in anomalies if a["type"] == "excessive_orders"]
    if excessive:
        suggestions.append({
            "target_team": "T3",
            "target_param": "open_order_dedup",
            "suggestion": "检测到重复挂单异常，建议检查防重逻辑（_open_orders 过滤是否生效）",
            "priority": "high",
        })

    hb_timeout = [a for a in anomalies if a["type"] == "heartbeat_timeout"]
    if hb_timeout:
        suggestions.append({
            "target_team": "system",
            "target_param": "watchdog",
            "suggestion": "引擎心跳超时，建议检查 Runtime 是否崩溃或被系统挂起",
            "priority": "critical",
        })

    return suggestions


# ═══════════════════════════════════════════════════════════════════════════
# Discord 报告
# ═══════════════════════════════════════════════════════════════════════════

def _send_discord_report(
    stats: Dict[str, Any],
    anomalies: List[Dict[str, Any]],
    suggestions: List[Dict[str, Any]],
) -> None:
    try:
        from trader.notify import DiscordNotifier
        from trader.models import Notification

        pnl = stats.get("total_pnl", 0)
        pnl_str = f"${pnl:+,.2f}" if pnl else "N/A"
        dd = stats.get("max_drawdown_pct", 0)
        trades = stats.get("trade_count", 0)

        anom_lines = ""
        if anomalies:
            anom_lines = "\n".join(
                f"  [{a['severity'].upper()}] {a['message']}"
                for a in anomalies[:5]
            )
        else:
            anom_lines = "  无异常"

        sug_lines = ""
        if suggestions:
            sug_lines = "\n".join(
                f"  → [{s['priority'].upper()}] {s['suggestion']}"
                for s in suggestions[:3]
            )
        else:
            sug_lines = "  无建议"

        body = (
            f"盈亏: {pnl_str}  |  最大回撤: {dd:.2f}%  |  成交次数: {trades}\n"
            f"异常检测:\n{anom_lines}\n"
            f"整改建议:\n{sug_lines}"
        )
        notifier = DiscordNotifier()
        notifier.send(Notification(title="T5 维护日报", body=body, kind="system"))
    except Exception as exc:
        logger.warning("Discord 报告发送失败: %s", exc)


# ═══════════════════════════════════════════════════════════════════════════
# 持久化反馈建议
# ═══════════════════════════════════════════════════════════════════════════

def _persist_suggestions(db_path: str, suggestions: List[Dict[str, Any]]) -> None:
    if not suggestions:
        return
    try:
        import duckdb
        con = duckdb.connect(db_path)
        con.execute("""
            CREATE TABLE IF NOT EXISTS t5_suggestions (
                id          VARCHAR PRIMARY KEY,
                target_team VARCHAR,
                target_param VARCHAR,
                suggestion  TEXT,
                priority    VARCHAR,
                created_at  TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        import uuid
        for s in suggestions:
            con.execute(
                "INSERT OR IGNORE INTO t5_suggestions "
                "(id, target_team, target_param, suggestion, priority, created_at) "
                "VALUES (?,?,?,?,?,NOW())",
                [str(uuid.uuid4()), s.get("target_team"), s.get("target_param"),
                 s.get("suggestion"), s.get("priority")],
            )
        con.close()
    except Exception as exc:
        logger.warning("T5 suggestions 持久化失败: %s", exc)
