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


def _default_period_label(period_hours: int) -> str:
    """按周期粒度生成去重身份：周报按 ISO 周，日报按日期。"""
    now = datetime.now(timezone.utc)
    if period_hours >= 24 * 7:
        year, week, _ = now.isocalendar()
        return f"{year}-W{week:02d}"
    return now.strftime("%Y-%m-%d")


def should_send_weekly_report(
    now_utc: datetime,
    last_sent_label: str | None,
    *,
    weekday: int = 4,      # 0=周一 … 4=周五
    hour_et: int = 16,
    minute_et: int = 30,
) -> bool:
    """周五收盘后发一次策略体检。

    和开盘确认一样用"到点之后 + 窗口"而不是精确匹配某一分钟——引擎重启或数
    据源卡顿都可能刚好跳过那一分钟，那样这一周的体检就再也不会发了。
    """
    from zoneinfo import ZoneInfo

    et = now_utc.astimezone(ZoneInfo("America/New_York"))
    year, week, _ = et.isocalendar()
    label = f"{year}-W{week:02d}"
    if label == last_sent_label:
        return False
    if et.weekday() != weekday:
        return False
    minutes = et.hour * 60 + et.minute
    start = hour_et * 60 + minute_et
    return start <= minutes < start + 180


# ═══════════════════════════════════════════════════════════════════════════
# 公共入口
# ═══════════════════════════════════════════════════════════════════════════

def run_maintenance(
    db_path: str = _DEFAULT_DB,
    period_hours: int = 24,
    send_discord: bool = False,
    period_label: str = "",
) -> TeamOutput:
    """运行 T5 维护分析，返回 TeamOutput。

    period_label 用作推送的去重身份（例如 "2026-W31"），让周五自动触发的周
    报和当天手动点一次"运行维护分析"落在同一条消息上，不会推两遍。
    """
    out = TeamOutput(team="T5")
    t0 = time.time()
    period_label = period_label or _default_period_label(period_hours)
    try:
        stats = _compute_stats(db_path, period_hours)
        anomalies = _detect_anomalies(db_path, period_hours)
        suggestions = _generate_suggestions(stats, anomalies)

        out.data["stats"] = stats
        out.data["anomalies"] = anomalies
        out.data["suggestions"] = suggestions
        out.data["period_hours"] = period_hours

        if send_discord:
            _send_discord_report(
                stats, anomalies, suggestions, period_label=period_label
            )

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
        # 之前这里查询用的列名 filled_at 在 fills 表里根本不存在（真实列名
        # 是 fill_time），SQL 每次都直接报错，被下面的 bare except 吞掉——
        # 每一份维护日报里的"成交次数"因此永远是 0，不管实际交易了多少笔。
        try:
            fills = con.execute(
                "SELECT symbol, side, filled_qty, avg_price FROM fills WHERE fill_time >= ? ",
                [since],
            ).fetchdf()
            if not fills.empty:
                stats["trade_count"] = len(fills)
                per_symbol: Dict[str, Dict[str, float]] = {}
                for _, row in fills.iterrows():
                    sym = row["symbol"]
                    notional = float(row["filled_qty"]) * float(row["avg_price"])
                    bucket = per_symbol.setdefault(sym, {"buy": 0.0, "sell": 0.0})
                    if str(row["side"]).upper() == "SELL":
                        bucket["sell"] += notional
                    else:
                        bucket["buy"] += notional
                if per_symbol:
                    # win_rate 是"卖出净现金流为正的标的数 / 有成交的标的数"，
                    # 不是逐笔已实现盈亏（fills 表没有成本基础，算不出真正
                    # 的逐笔盈亏）——比之前硬编码永远 0.0 诚实，但仍是近似值。
                    wins = sum(1 for b in per_symbol.values() if b["sell"] - b["buy"] > 0)
                    stats["win_rate"] = round(wins / len(per_symbol), 4)  # 0-1 分数，跟下面 *100 的展示逻辑保持一致
        except Exception as exc:
            logger.warning("T5 stats: fills 查询失败: %s", exc)

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

        # 异常2：风控拒单事件
        # 原查询用的是 risk_events 表 + level='critical'，但 risk_events
        # 从来没有任何代码写入过（log_risk_event() 是死代码），且这张表
        # 根本没有 level 列——查询必报错，被 bare except 吞掉，这个异常
        # 检测永远不会触发。Runtime 实际写风控裁决用的是 plan_risk_events
        # 表（verdict='BLOCKED' 表示被拒），改用这张表。
        try:
            risk_blocks = con.execute(
                "SELECT COUNT(*) FROM plan_risk_events WHERE verdict='BLOCKED' AND ts >= ?",
                [since],
            ).fetchone()
            if risk_blocks and risk_blocks[0] > 0:
                anomalies.append({
                    "type": "risk_blocks",
                    "severity": "high",
                    "message": f"过去 {period_hours}h 有 {risk_blocks[0]} 个计划被风控拒绝",
                    "value": risk_blocks[0],
                })
        except Exception as exc:
            logger.warning("T5 anomaly: 风控事件查询失败: %s", exc)

        # 异常3：高频下单（同一标的在窗口内多次下单）
        # order_intents 没有 created_at 列（真实列名是 submitted_at），
        # 同样一直静默失败——这个本该用来抓"失控疯狂下单"的检测从写出来
        # 就没生效过。
        try:
            orders = con.execute(
                "SELECT symbol, COUNT(*) as cnt FROM order_intents WHERE submitted_at >= ? "
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
        except Exception as exc:
            logger.warning("T5 anomaly: 订单频率查询失败: %s", exc)

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
    period_label: str = "",
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

        win_rate = stats.get("win_rate", 0) * 100
        body = (
            f"盈亏: {pnl_str}  |  最大回撤: {dd:.2f}%  |  成交次数: {trades}"
            f"  |  胜率: {win_rate:.1f}%\n"
            f"异常检测:\n{anom_lines}\n"
            f"整改建议:\n{sug_lines}"
        )
        # 授权交给 DISCORD_EXTERNAL_SEND_ENABLED 总闸决定，这里不再硬编码
        # 绕过它：总闸关着时手动点击同样应该被拦下并如实告知，而不是悄悄放行。
        notifier = DiscordNotifier()
        notifier.send(
            Notification(
                title=f"🔧 周报 · 策略体检（近 {period_label}）",
                body=body,
                kind="system",
                fields={
                    "成交": str(trades),
                    "胜率": f"{win_rate:.1f}%",
                    "最大回撤": f"{dd:.2f}%",
                },
                # 一周一份。自动触发和手动点击落在同一个身份上，所以周五下午
                # 手动跑一次维护不会和自动周报重复推送。
                dedupe_key=f"maintenance:{period_label}",
            )
        )
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
