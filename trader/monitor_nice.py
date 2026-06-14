"""
trader/monitor_nice.py
美股K线 · 交易监控 / 决策前端 (NiceGUI)。

三项全局体验保证（本版重点）：
  A. 增量刷新（不闪烁）：每个实时页拆成 build（建结构一次）+ update（只改数据）。
     定时器只调当前页的 update，绝不 clear+rebuild，所以不闪。
  B. 图状态保持：所有 plotly 图设 uirevision，刷新数据时保留缩放/平移/视图。
  C. 持久化用户偏好：所有输入/选择 + 当前 tab 存到 conf/ui_settings.json，
     重启应用后自动恢复。

导航按数据成熟度分三组：实况（真实审计库）/ 研究（真回测）/ 规划中（示例，带徽章）。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from nicegui import ui

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from trader.monitor_data import (  # noqa: E402
    equity_df, fills_df, heartbeat, live_alpaca_equity, orders_df, risk_events_df, signals_df,
)

if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

_LOG_FILE = _ROOT / "trader_engine.log"
_PID_FILE = _ROOT / ".engine.pid"
_PREFS_PATH = _ROOT / "conf" / "ui_settings.json"
_REFRESH_SEC = 5.0
_AI_DB = str(_ROOT / "ai_states.duckdb")

# ═══════════════════════════════════════════════════════════════════════════
# 持久化用户偏好 (纯 JSON，跨重启)
# ═══════════════════════════════════════════════════════════════════════════

def _load_prefs() -> dict:
    try:
        return json.loads(_PREFS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_prefs() -> None:
    try:
        _PREFS_PATH.parent.mkdir(parents=True, exist_ok=True)
        _PREFS_PATH.write_text(json.dumps(_PREFS, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


_PREFS: dict = _load_prefs()


def _pref(key: str, default):
    v = _PREFS.get(key, default)
    return v if v is not None else default


def _set_pref(key: str, value) -> None:
    _PREFS[key] = value
    _save_prefs()


def _persist(widget, key: str):
    """控件值变化时自动存盘。控件初值应在创建时用 _pref(key, default) 读取。"""
    widget.on_value_change(lambda e, k=key: _set_pref(k, e.value))
    return widget

# ═══════════════════════════════════════════════════════════════════════════
# 引擎进程控制
# ═══════════════════════════════════════════════════════════════════════════

def _engine_running() -> bool:
    if not _PID_FILE.exists():
        return False
    try:
        pid = int(_PID_FILE.read_text().strip())
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"],
                             capture_output=True, text=True, timeout=3)
        return str(pid) in out.stdout
    except Exception:
        return False


def _start_engine(symbols: str, strategies: str, tf: str, interval) -> str:
    if _engine_running():
        return "引擎已在运行"
    syms = ",".join(s.strip().upper() for s in symbols.split(",") if s.strip())
    strats = ",".join(s.strip() for s in strategies.split(",") if s.strip())
    if not syms or not strats:
        return "❌ 请填写标的与策略"
    cmd = [sys.executable, "-m", "trader.main", "--symbols", syms,
           "--strategies", strats, "--tf", tf, "--interval", str(int(interval))]
    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        log_fh = open(_LOG_FILE, "w", encoding="utf-8", buffering=1)
        proc = subprocess.Popen(
            cmd, stdout=log_fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            env=env, cwd=str(_ROOT),
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        _PID_FILE.write_text(str(proc.pid))
        return f"✅ 引擎已启动 (PID {proc.pid})"
    except Exception as exc:
        return f"❌ 启动失败: {exc}"


def _stop_engine() -> str:
    pid = _PID_FILE.read_text().strip() if _PID_FILE.exists() else None
    if not pid:
        return "引擎未运行"
    try:
        if sys.platform == "win32":
            subprocess.call(["taskkill", "/F", "/PID", pid],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            os.kill(int(pid), 15)
        _PID_FILE.unlink(missing_ok=True)
        return "🛑 引擎已停止"
    except Exception as exc:
        _PID_FILE.unlink(missing_ok=True)
        return f"⚠️ {exc}"


def _tail_log(n: int = 40) -> str:
    if not _LOG_FILE.exists():
        return "（暂无日志）"
    try:
        return "\n".join(_LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]) or "（日志为空）"
    except Exception:
        return "读取失败"

# ═══════════════════════════════════════════════════════════════════════════
# 格式化辅助
# ═══════════════════════════════════════════════════════════════════════════

def _fmt_time(ts) -> str:
    if ts is None or (isinstance(ts, float) and pd.isna(ts)):
        return "—"
    try:
        return pd.to_datetime(ts).strftime("%m-%d %H:%M:%S")
    except Exception:
        return str(ts)


def _money(x) -> str:
    try:
        return f"${float(x):,.0f}"
    except Exception:
        return "—"

# ═══════════════════════════════════════════════════════════════════════════
# 设计系统 (CSS)
# ═══════════════════════════════════════════════════════════════════════════

_CSS = """
:root{
  --bg:#0d1117; --panel:#161b22; --panel2:#1c2128; --border:#30363d;
  --fg:#e6edf3; --fg2:#8b949e; --fg3:#6e7681;
  --pos:#3fb950; --neg:#f85149; --ai:#58a6ff; --warn:#d29922;
  --mono:'JetBrains Mono','Cascadia Code',Consolas,ui-monospace,monospace;
}
*{box-sizing:border-box;}
body{background:var(--bg);color:var(--fg);
  font-family:'Segoe UI','Microsoft YaHei',system-ui,sans-serif;}
.q-layout,.q-page-container,.q-page{padding:0!important;margin:0!important;min-height:0!important;}
.nicegui-content{padding:0!important;gap:0!important;width:100vw;height:100vh;
  display:flex;flex-direction:column;align-items:stretch;max-width:none!important;overflow:hidden;}
::-webkit-scrollbar{width:9px;height:9px;}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:5px;}
::-webkit-scrollbar-track{background:transparent;}

.num{font-family:var(--mono);font-variant-numeric:tabular-nums;}
.pos{color:var(--pos)!important;} .neg{color:var(--neg)!important;} .ai{color:var(--ai)!important;}

.qa-topbar{display:flex;width:100%;align-items:center;gap:30px;height:60px;min-height:60px;
  padding:0 22px;background:var(--panel);border-bottom:1px solid var(--border);}
.qa-brand{font-size:16px;font-weight:700;letter-spacing:.02em;}
.qa-brand .dot{color:var(--ai);}
.qa-spacer{flex:1;}
.qa-stat{display:flex;flex-direction:column;line-height:1.3;min-width:96px;}
.qa-stat .l{font-size:11px;color:var(--fg3);}
.qa-stat .v{font-size:15px;font-weight:600;}

.qa-body{flex:1;width:100%;min-height:0;display:flex;flex-direction:row;}
.qa-nav{width:160px;min-width:160px;background:var(--panel);
  border-right:1px solid var(--border);overflow-y:auto;padding:6px 0;
  display:flex;flex-direction:column;}
.qa-nav-group{font-size:10px;color:var(--fg3);text-transform:uppercase;
  letter-spacing:.06em;padding:14px 16px 5px;}
.qa-nav-item{display:flex;align-items:center;gap:9px;padding:9px 16px;
  color:var(--fg2);cursor:pointer;font-size:14px;border-left:2px solid transparent;
  user-select:none;}
.qa-nav-item:hover{background:var(--panel2);color:var(--fg);}
.qa-nav-item.active{background:var(--panel2);color:var(--fg);
  border-left-color:var(--ai);font-weight:600;}
.qa-nav-item .ico{font-size:15px;width:18px;text-align:center;}

.qa-content{flex:1;overflow-y:auto;padding:22px;display:flex;
  flex-direction:column;gap:16px;background:var(--bg);}
.qa-h{font-size:19px;font-weight:700;}
.qa-h-sub{font-size:13px;color:var(--fg3);margin-top:2px;}

.qa-kpi-row{display:flex;gap:14px;width:100%;flex-wrap:wrap;}
.qa-kpi{flex:1;min-width:150px;background:var(--panel);border:1px solid var(--border);
  border-radius:12px;padding:14px 16px;}
.qa-kpi .l{font-size:12px;color:var(--fg3);}
.qa-kpi .v{font-size:24px;font-weight:700;margin-top:4px;}
.qa-kpi .s{font-size:12px;color:var(--fg2);margin-top:3px;}

.qa-card{background:var(--panel);border:1px solid var(--border);
  border-radius:12px;padding:18px;width:100%;}
.qa-card-title{font-size:14px;font-weight:600;}
.qa-card-sub{font-size:12px;color:var(--fg3);margin-top:2px;margin-bottom:12px;}

.qa-card .q-table__container,.qa-card .q-table,.qa-card .q-table__top,
.qa-card .q-table thead tr,.qa-card .q-table tbody td,.qa-card .q-table th{
  background:transparent!important;color:var(--fg)!important;border-color:var(--border)!important;}
.qa-card .q-table th{color:var(--fg3)!important;font-size:11px;text-transform:uppercase;letter-spacing:.04em;}
.qa-card .q-table tbody tr:hover{background:var(--panel2)!important;}

.qa-badge{display:inline-flex;align-items:center;gap:6px;font-size:11px;
  padding:3px 10px;border-radius:999px;font-weight:600;}
.qa-badge.demo{background:rgba(210,153,34,.15);color:var(--warn);
  border:1px solid rgba(210,153,34,.35);}
.qa-badge.live{background:rgba(63,185,80,.15);color:var(--pos);
  border:1px solid rgba(63,185,80,.35);}

.qa-note{font-size:12px;color:var(--fg3);padding:10px 14px;background:var(--panel2);
  border:1px solid var(--border);border-radius:8px;border-left:3px solid var(--warn);
  line-height:1.5;}
.qa-empty{display:flex;flex-direction:column;align-items:center;justify-content:center;
  padding:46px;color:var(--fg3);gap:8px;}
.qa-empty .ico{font-size:30px;opacity:.5;}
.qa-empty .t{font-size:13px;}
.qa-code{font-family:var(--mono);font-size:13px;background:var(--bg);
  border:1px solid var(--border);border-radius:8px;padding:12px 14px;color:var(--ai);
  white-space:pre-wrap;word-break:break-all;}

/* 决策台 */
.cp-agent-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;width:100%;}
@keyframes cp-pulse{0%,100%{opacity:1}50%{opacity:.3}}
.cp-mgr{background:var(--panel);border:1px solid rgba(88,166,255,.4);
  border-radius:12px;padding:18px;width:100%;}
.cp-pick-row{display:flex;gap:10px;flex-wrap:wrap;margin-top:10px;}
.cp-pick{padding:9px 14px;border-radius:8px;min-width:88px;text-align:center;}
.cp-pick.buy{background:rgba(63,185,80,.12);border:1px solid rgba(63,185,80,.35);}
.cp-pick.watch{background:rgba(210,153,34,.1);border:1px solid rgba(210,153,34,.3);}
.cp-pick.avoid{background:rgba(248,81,73,.1);border:1px solid rgba(248,81,73,.25);}
.cp-feed{background:var(--panel);border:1px solid var(--border);
  border-radius:12px;padding:14px 16px;width:100%;}
.cp-feed-row{display:flex;align-items:baseline;gap:9px;padding:5px 0;
  border-bottom:1px solid #21262d;font-size:12.5px;}
.cp-feed-row:last-child{border:none;}
.cp-ts{color:var(--fg3);font-size:11px;min-width:65px;
  font-family:var(--mono);flex-shrink:0;}
.cp-tag{font-size:10px;font-weight:700;padding:2px 7px;border-radius:999px;flex-shrink:0;}
"""

# ═══════════════════════════════════════════════════════════════════════════
# 页面骨架
# ═══════════════════════════════════════════════════════════════════════════

ui.add_head_html("<style>" + _CSS + "</style>")

_state: dict = {"tab": "overview", "updater": None}
_nav_refs: dict = {}

with ui.element("div").classes("qa-topbar"):
    ui.html('<span class="qa-brand">美股<span class="dot">K线</span>'
            '<span style="font-weight:400;color:var(--fg3);font-size:12px;margin-left:6px">'
            'DuckDB · Alpaca 实时</span></span>')
    ui.element("div").classes("qa-spacer")

    def _topstat(label: str):
        with ui.element("div").classes("qa-stat"):
            ui.label(label).classes("l")
            v = ui.label("—").classes("v num")
        return v

    top_total = _topstat("总资产")
    top_pnl = _topstat("近 24h 盈亏")
    top_hb = _topstat("心跳")
    with ui.element("div").classes("qa-stat"):
        ui.label("引擎").classes("l")
        top_engine = ui.label("—").classes("v")

with ui.element("div").classes("qa-body"):
    with ui.element("div").classes("qa-nav"):

        def _nav_group(title: str):
            ui.label(title).classes("qa-nav-group")

        def _nav_item(name: str, icon: str, label: str):
            el = ui.element("div").classes("qa-nav-item")
            with el:
                ui.html(f'<span class="ico">{icon}</span><span>{label}</span>')
            el.on("click", lambda n=name: _select(n))
            _nav_refs[name] = el

        _nav_group("实况")
        _nav_item("overview", "📊", "总览")
        _nav_item("activity", "🧾", "交易记录")
        _nav_item("cockpit", "🤖", "决策台")
        _nav_item("system", "⚙️", "系统")
        _nav_group("研究")
        _nav_item("research", "🔬", "研究")
        _nav_group("规划中 · 示例")
        _nav_item("universe", "🔭", "选股池")
        _nav_item("models", "🧠", "模型")

    content = ui.element("div").classes("qa-content")

# ═══════════════════════════════════════════════════════════════════════════
# 通用 UI 组件
# ═══════════════════════════════════════════════════════════════════════════

def _page_head(title: str, sub: str = "", badge: str = ""):
    with ui.element("div"):
        with ui.row().classes("items-center gap-3").style("margin:0"):
            ui.label(title).classes("qa-h")
            if badge == "live":
                ui.html('<span class="qa-badge live">● 实时数据</span>')
            elif badge == "demo":
                ui.html('<span class="qa-badge demo">⚠ 示例数据</span>')
        if sub:
            ui.label(sub).classes("qa-h-sub")


def _kpi(label: str, value: str = "—", sub: str = "", tone: str = ""):
    with ui.element("div").classes("qa-kpi"):
        ui.label(label).classes("l")
        v = ui.label(value).classes(f"v num {tone}")
        if sub:
            ui.label(sub).classes("s")
    return v


def _empty(msg: str, icon: str = "∅"):
    with ui.element("div").classes("qa-empty"):
        ui.html(f'<span class="ico">{icon}</span>')
        ui.label(msg).classes("t")


def _make_table(col_specs: list):
    cols = [{"name": f, "label": l, "field": f, "align": a} for f, l, a in col_specs]
    return ui.table(columns=cols, rows=[], row_key="__i",
                    pagination=0).props("flat dense").classes("w-full")


def _fill_table(table, df: pd.DataFrame, col_specs: list, fmts=None, max_rows: int = 12):
    fmts = fmts or {}
    rows = []
    for i, (_, r) in enumerate(df.head(max_rows).iterrows()):
        row = {"__i": i}
        for f, _l, _a in col_specs:
            v = r.get(f) if f in r else None
            if f in fmts:
                try:
                    row[f] = fmts[f](v)
                except Exception:
                    row[f] = "—"
            else:
                row[f] = "—" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)
        rows.append(row)
    table.rows = rows
    table.update()


def _equity_fig(eq: pd.DataFrame, uirev: str) -> go.Figure:
    fig = go.Figure()
    if eq is not None and not eq.empty and "total_equity" in eq.columns:
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(eq["ts"]), y=eq["total_equity"],
            mode="lines", line=dict(width=2, color="#58a6ff"), name="权益",
            fill="tozeroy", fillcolor="rgba(88,166,255,0.08)"))
    fig.update_layout(
        height=300, margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8b949e", size=11), showlegend=False, uirevision=uirev,
        xaxis=dict(gridcolor="#21262d", showgrid=True),
        yaxis=dict(gridcolor="#21262d", showgrid=True))
    return fig

# ═══════════════════════════════════════════════════════════════════════════
# 各页渲染 —— 实时页返回 update 函数；静态页返回 None
# ═══════════════════════════════════════════════════════════════════════════

def _render_overview():
    _page_head("总览", "账户权益与最新成交 · 数据窗口 24 小时", badge="live")

    with ui.element("div").classes("qa-kpi-row"):
        k_total = _kpi("总资产")
        k_pnl = _kpi("近 24h 盈亏")
        k_cash = _kpi("现金")
        k_unreal = _kpi("浮动盈亏")

    with ui.element("div").classes("qa-card"):
        ui.label("权益曲线").classes("qa-card-title")
        ui.label("equity_snapshots · 最近 24 小时").classes("qa-card-sub")
        eq_plot = ui.plotly(_equity_fig(None, "ov-eq")).classes("w-full")
        eq_empty = ui.element("div")

    with ui.element("div").classes("qa-card"):
        ui.label("最近成交").classes("qa-card-title")
        ui.label("fills · 最近 24 小时").classes("qa-card-sub")
        fill_cols = [("fill_time", "时间", "left"), ("symbol", "标的", "left"),
                     ("side", "方向", "center"), ("filled_qty", "数量", "right"),
                     ("avg_price", "均价", "right")]
        ft = _make_table(fill_cols)
        ft_empty = ui.element("div")

    def update():
        live = live_alpaca_equity()
        eq = equity_df(24)
        has = not eq.empty and "total_equity" in eq.columns

        if live is not None:
            # Alpaca API 实时权益为主：总资产 + 现金来自 API
            k_total.set_text(_money(live["equity"]))
            k_cash.set_text(_money(live["cash"]))
            # 近 24h 盈亏：用 API 现值减 DuckDB 历史起点
            if has:
                pnl = live["equity"] - float(eq["total_equity"].iloc[0])
                k_pnl.set_text(f"{pnl:+,.0f}")
                k_pnl.classes(remove="pos neg", add="pos" if pnl >= 0 else "neg")
            else:
                k_pnl.set_text("—")
            unreal = (
                float(eq["unrealized_pnl"].iloc[-1])
                if has and "unrealized_pnl" in eq.columns else None
            )
            k_unreal.set_text(f"{unreal:+,.0f}" if unreal is not None else "—")
            if unreal is not None:
                k_unreal.classes(remove="pos neg", add="pos" if unreal >= 0 else "neg")
        elif has:
            # 降级到 DuckDB 历史数据
            total = float(eq["total_equity"].iloc[-1])
            pnl = total - float(eq["total_equity"].iloc[0])
            cash = float(eq["cash"].iloc[-1]) if "cash" in eq.columns else None
            unreal = float(eq["unrealized_pnl"].iloc[-1]) if "unrealized_pnl" in eq.columns else None
            k_total.set_text(_money(total))
            k_pnl.set_text(f"{pnl:+,.0f}")
            k_pnl.classes(remove="pos neg", add="pos" if pnl >= 0 else "neg")
            k_cash.set_text(_money(cash) if cash is not None else "—")
            k_unreal.set_text(f"{unreal:+,.0f}" if unreal is not None else "—")
            if unreal is not None:
                k_unreal.classes(remove="pos neg", add="pos" if unreal >= 0 else "neg")
        else:
            for k in (k_total, k_pnl, k_cash, k_unreal):
                k.set_text("—")

        eq_plot.figure = _equity_fig(eq, "ov-eq")
        eq_plot.update()
        eq_plot.set_visibility(has)
        eq_empty.clear()
        if not has:
            with eq_empty:
                _empty("暂无权益记录 — 在「系统」页启动引擎后将自动写入", "📈")

        fills = fills_df(24)
        ft.set_visibility(not fills.empty)
        ft_empty.clear()
        if fills.empty:
            with ft_empty:
                _empty("暂无成交记录", "🧾")
        else:
            _fill_table(ft, fills, fill_cols, fmts={
                "fill_time": _fmt_time,
                "filled_qty": lambda v: f"{float(v):,.0f}",
                "avg_price": lambda v: f"${float(v):,.2f}",
            })

    update()
    return update


def _render_activity():
    _page_head("交易记录", "信号、风控裁决与订单 · 数据窗口 24 小时", badge="live")

    sig_cols = [("signal_time", "时间", "left"), ("symbol", "标的", "left"),
                ("strategy", "策略", "left"), ("side", "方向", "center"),
                ("exec_price", "执行价", "right")]
    risk_cols = [("ts", "时间", "left"), ("symbol", "标的", "left"),
                 ("verdict", "裁决", "center"), ("reason", "原因", "left")]
    order_cols = [("created_at", "时间", "left"), ("symbol", "标的", "left"),
                  ("side", "方向", "center"), ("qty", "数量", "right"), ("status", "状态", "center")]

    with ui.element("div").classes("qa-card"):
        ui.label("策略信号").classes("qa-card-title")
        ui.label("signals").classes("qa-card-sub")
        sig_t = _make_table(sig_cols)
        sig_e = ui.element("div")
    with ui.element("div").classes("qa-card"):
        ui.label("风控事件").classes("qa-card-title")
        ui.label("risk_events").classes("qa-card-sub")
        risk_t = _make_table(risk_cols)
        risk_e = ui.element("div")
    with ui.element("div").classes("qa-card"):
        ui.label("订单").classes("qa-card-title")
        ui.label("orders").classes("qa-card-sub")
        order_t = _make_table(order_cols)
        order_e = ui.element("div")

    def _refresh_one(table, empty_box, df, cols, fmts, icon, msg):
        table.set_visibility(not df.empty)
        empty_box.clear()
        if df.empty:
            with empty_box:
                _empty(msg, icon)
        else:
            _fill_table(table, df, cols, fmts=fmts)

    def update():
        _refresh_one(sig_t, sig_e, signals_df(24), sig_cols,
                     {"signal_time": _fmt_time, "exec_price": lambda v: f"${float(v):,.2f}"},
                     "📡", "暂无信号")
        _refresh_one(risk_t, risk_e, risk_events_df(24), risk_cols,
                     {"ts": _fmt_time}, "🛡️", "暂无风控事件")
        _refresh_one(order_t, order_e, orders_df(24), order_cols,
                     {"created_at": _fmt_time, "qty": lambda v: f"{float(v):,.0f}"},
                     "📋", "暂无订单")

    update()
    return update


def _render_system():
    _page_head("系统", "引擎控制与运行健康", badge="live")

    with ui.element("div").classes("qa-card"):
        ui.label("引擎控制").classes("qa-card-title")
        ui.label("启动 / 停止实时交易引擎 (trader.main)").classes("qa-card-sub")
        with ui.row().classes("items-end gap-3 flex-wrap"):
            sym_in = _persist(ui.input("标的", value=_pref("sys_sym", "QQQ")).props("dark dense outlined").style("width:120px"), "sys_sym")
            strat_in = _persist(ui.input("策略", value=_pref("sys_strat", "上周高低点(周K突破)")).props("dark dense outlined").style("width:220px"), "sys_strat")
            tf_in = _persist(ui.select(["5m", "30m", "1h", "1d"], value=_pref("sys_tf", "30m"), label="周期").props("dark dense outlined").style("width:90px"), "sys_tf")
            int_in = _persist(ui.number("间隔(秒)", value=_pref("sys_int", 30)).props("dark dense outlined").style("width:110px"), "sys_int")
        ui.html('<div class="qa-note">总资产、现金、持仓全部以 Alpaca 账户为准；系统不会在本地覆盖账户权益。</div>')

        with ui.row().classes("gap-3").style("margin-top:14px"):
            def _do_start():
                ui.notify(_start_engine(sym_in.value, strat_in.value, tf_in.value, int_in.value))
            def _do_stop():
                ui.notify(_stop_engine())
            ui.button("▶ 启动引擎", on_click=_do_start, color="positive").props("unelevated")
            ui.button("■ 停止", on_click=_do_stop, color="negative").props("unelevated outline")

    with ui.element("div").classes("qa-kpi-row"):
        k_eng = _kpi("引擎进程")
        k_hb = _kpi("最近心跳")
        k_db = _kpi("审计库", "trade.duckdb")
        k_win = _kpi("数据窗口", "24h")

    with ui.element("div").classes("qa-card"):
        ui.label("引擎日志 · 最后 40 行").classes("qa-card-title")
        log_html = ui.html(f'<div class="qa-code">{_tail_log(40)}</div>')

    def update():
        running = _engine_running()
        k_eng.set_text("运行中" if running else "已停止")
        k_eng.classes(remove="pos neg", add="pos" if running else "neg")
        hb = heartbeat()
        if hb is not None:
            secs = (datetime.now(timezone.utc) - hb).total_seconds()
            k_hb.set_text(f"{secs:.0f} 秒前" if secs < 120 else f"{secs/60:.0f} 分钟前")
        else:
            k_hb.set_text("—")
        log_html.set_content(f'<div class="qa-code">{_tail_log(40)}</div>')

    update()
    return update


def _render_research():
    _page_head("研究", "策略回测 · 与实盘共用同一引擎 simulate()，结果与生产一致")

    try:
        from trader.data_cache import get_bars, list_cached_files
        from trader.strategy_core import (
            DEFAULT_STRATEGY_PARAMS, STRATEGY_OPTIONS, compute_signals,
        )
        from trader.engine import simulate
    except Exception as exc:
        _empty(f"无法加载策略引擎: {exc}", "⚠️")
        return None

    files = list_cached_files()
    symbols = sorted({f["文件"].rsplit("_", 1)[0] for f in files}) or ["QQQ"]
    tfs = sorted({f["文件"].rsplit("_", 1)[1].replace(".parquet", "") for f in files}) or ["30m"]
    strategies = list(STRATEGY_OPTIONS)

    def _valid(val, options, default):
        return val if val in options else default

    r_sym = _valid(_pref("r_sym", symbols[0]), symbols, symbols[0])
    r_tf = _valid(_pref("r_tf", tfs[0]), tfs, tfs[0])
    r_strat = _valid(_pref("r_strat", "5/20均线金叉死叉"), strategies, strategies[0])

    with ui.element("div").classes("qa-card"):
        ui.label("回测设置").classes("qa-card-title")
        ui.label("数据严格本地优先 (bars/ Parquet)，不自动联网").classes("qa-card-sub")
        with ui.row().classes("items-end gap-3 flex-wrap"):
            sym_sel = _persist(ui.select(symbols, value=r_sym, label="标的").props("dark dense outlined").style("width:130px"), "r_sym")
            tf_sel = _persist(ui.select(tfs, value=r_tf, label="周期").props("dark dense outlined").style("width:95px"), "r_tf")
            strat_sel = _persist(ui.select(strategies, value=r_strat, label="策略").props("dark dense outlined").style("width:250px"), "r_strat")
            cap_in = _persist(ui.number("本金", value=_pref("r_cap", 10000), format="%.0f").props("dark dense outlined").style("width:110px"), "r_cap")
            lev_in = _persist(ui.number("杠杆", value=_pref("r_lev", 1.0), step=0.5, min=1.0).props("dark dense outlined").style("width:90px"), "r_lev")
            fill_sel = _persist(ui.select({"next_open": "下一开盘", "close": "当根收盘"}, value=_pref("r_fill", "next_open"), label="成交").props("dark dense outlined").style("width:120px"), "r_fill")
            risk_sw = _persist(ui.switch("风控熔断", value=_pref("r_risk", False)), "r_risk")
            run_btn = ui.button("▶ 运行回测", color="primary").props("unelevated")

    result_area = ui.column().style("gap:16px;width:100%")

    def _run():
        result_area.clear()
        with result_area:
            df = get_bars(sym_sel.value, tf_sel.value)
            if df is None or df.empty:
                _empty(f"本地无 {sym_sel.value} {tf_sel.value} 数据 — 请先下载", "📭")
                return
            try:
                df_sig = compute_signals(df.copy(), strat_sel.value, **DEFAULT_STRATEGY_PARAMS)
                res = simulate(
                    df_sig, capital=float(cap_in.value or 10000),
                    leverage=float(lev_in.value or 1.0),
                    fill=fill_sel.value, risk_halt=bool(risk_sw.value),
                )
            except Exception as exc:
                _empty(f"回测失败: {exc}", "⚠️")
                return

            tr = res.total_return
            with ui.element("div").classes("qa-kpi-row"):
                _kpi("最终权益", _money(res.final_equity))
                _kpi("总收益", f"{tr:+.2%}", tone=("pos" if tr >= 0 else "neg"))
                _kpi("平仓次数", str(res.closed_trades))
                _kpi("胜率", f"{res.win_rate:.1%}" if res.closed_trades else "—")
                _kpi("数据根数", f"{len(df):,}")

            with ui.element("div").classes("qa-card"):
                ui.label("权益曲线").classes("qa-card-title")
                ui.label(f"{strat_sel.value} · {sym_sel.value} {tf_sel.value}").classes("qa-card-sub")
                if res.equity_curve is not None and not res.equity_curve.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=res.equity_curve.index.strftime("%Y-%m-%d %H:%M"),
                        y=res.equity_curve.values,
                        mode="lines", line=dict(width=2, color="#58a6ff"), name="权益",
                        fill="tozeroy", fillcolor="rgba(88,166,255,0.08)"))
                    fig.add_hline(y=res.initial_capital, line=dict(width=1, dash="dot", color="#6e7681"))
                    fig.update_layout(
                        height=260, margin=dict(l=8, r=8, t=8, b=8),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#8b949e", size=11), showlegend=False, uirevision="rs-eq",
                        xaxis=dict(gridcolor="#21262d"), yaxis=dict(gridcolor="#21262d"))
                    ui.plotly(fig).classes("w-full")
                else:
                    _empty("无权益曲线", "📈")

            with ui.element("div").classes("qa-card"):
                ui.label("K线与买卖点").classes("qa-card-title")
                n = min(len(df_sig), 320)
                d = df_sig.tail(n).reset_index(drop=True)
                ui.label(f"最近 {n} 根 · ▲买入 ▼卖出").classes("qa-card-sub")
                x = list(range(len(d)))
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=x, open=d["open"], high=d["high"], low=d["low"], close=d["close"],
                    name="OHLC", increasing_line_color="#3fb950", decreasing_line_color="#f85149"))
                buys = d.index[d["strat_signal"] == 1].tolist()
                sells = d.index[d["strat_signal"] == -1].tolist()
                if buys:
                    fig.add_trace(go.Scatter(
                        x=buys, y=d.loc[buys, "strat_exec_px"], mode="markers", name="买入",
                        marker=dict(symbol="triangle-up", size=11, color="#3fb950")))
                if sells:
                    fig.add_trace(go.Scatter(
                        x=sells, y=d.loc[sells, "strat_exec_px"], mode="markers", name="卖出",
                        marker=dict(symbol="triangle-down", size=11, color="#f85149")))
                fig.update_layout(
                    height=380, margin=dict(l=8, r=8, t=8, b=8),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#8b949e", size=11), uirevision="rs-kline",
                    legend=dict(orientation="h", yanchor="bottom", y=1.01),
                    xaxis=dict(gridcolor="#21262d", rangeslider=dict(visible=False)),
                    yaxis=dict(gridcolor="#21262d"))
                ui.plotly(fig).classes("w-full")

            with ui.element("div").classes("qa-card"):
                ui.label("深度研究 · Marimo").classes("qa-card-title")
                ui.label("需要响应式多单元格探索时，用 Marimo notebook（同一引擎）").classes("qa-card-sub")
                ui.html('<div class="qa-code">.venv\\Scripts\\marimo.exe edit notebooks/research.py</div>')

    run_btn.on_click(_run)
    _run()
    return None


def _render_cockpit():
    _page_head("决策台", "多 Agent 并行分析 · ThreadPoolExecutor + DuckDB 持久化")

    ui.html('<div class="qa-note">'
            'AI 只产出 Advisory / TradePlan(DRAFT)；执行需通过风控层并人工审批。'
            '点击「运行一轮」触发 4 个 agent 并行分析（需 Ollama 在线）。'
            '</div>')

    try:
        from trader.ai.manager import get_manager
        mgr = get_manager()
    except Exception as exc:
        _empty(f"无法加载 AgentManager: {exc}", "⚠️")
        return None

    # ── ① Manager 决策区 ────────────────────────────────────────────────────
    with ui.element("div").classes("cp-mgr"):
        with ui.row().classes("items-center gap-3").style("margin:0;flex-wrap:wrap"):
            ui.label("Manager 决策区").classes("qa-card-title").style("color:var(--ai)")
            status_lbl = ui.label("空闲").style("font-size:12px;color:var(--fg3)")
            ui.element("div").style("flex:1")
            sym_in = ui.input(
                "分析标的 (逗号分隔)",
                value=_pref("cp_syms", "SPY,AAPL,NVDA,MSFT"),
            ).props("dark dense outlined").style("width:240px")
            sym_in.on_value_change(lambda e: _set_pref("cp_syms", e.value))
            run_btn = ui.button("▶ 运行一轮", color="primary").props("unelevated dense")
        picks_html = ui.html(
            '<div class="cp-pick-row">'
            '<span style="color:var(--fg3);font-size:12px">运行后显示推荐</span>'
            '</div>'
        )

    # ── ② Agent 状态面板（6 blocks）───────────────────────────────────────
    ui.label("Agent 状态").classes("qa-card-title").style("margin-top:4px")
    agent_cards: dict = {}
    with ui.element("div").classes("cp-agent-grid"):
        for role in _AGENT_META:
            agent_cards[role] = ui.html(_agent_card_html(role, None))

    # ── ③ 活动流 ─────────────────────────────────────────────────────────
    ui.label("实时活动流").classes("qa-card-title").style("margin-top:4px")
    feed_el = ui.html(_feed_html([]))

    # ── 运行按钮逻辑 ─────────────────────────────────────────────────────
    def _do_run():
        if _cockpit_run["running"]:
            ui.notify("Agent 正在运行，请稍候")
            return

        raw_syms = (sym_in.value or "SPY,AAPL,NVDA").strip()
        symbols = [s.strip().upper() for s in raw_syms.split(",") if s.strip()]
        if not symbols:
            ui.notify("请填写至少一个标的")
            return

        _cockpit_run["running"] = True
        run_btn.props("disable")
        status_lbl.set_text("运行中…")
        status_lbl.style("color:var(--ai)")
        # 立即标记所有 real agent 为 running
        mgr._init_db(_AI_DB)
        for role in ("technical", "news", "web_research", "bull_bear"):
            mgr._write_state(_AI_DB, role, "running", 0.0, None, {})

        import threading
        from trader.contracts import AgentContext
        from trader.models import Candidate, utc_now as _now

        def _bg():
            import pandas as pd
            from trader.config import TradingConfig
            from trader.contracts import AgentContext
            from trader.data_cache import upsert_bars as _upsert
            from trader.data_feed import AlpacaDataFeed
            from trader.models import Candidate, utc_now as _now
            from trader.selection import ConsensusSelector

            try:
                now = _now()
                cfg = TradingConfig()   # 读 .env 默认值（API key / timeframe 等）

                # ① 先拉 K 线更新缓存，ConsensusSelector 依赖 data_cache
                _cockpit_run["stage"] = "拉取 K 线…"
                try:
                    feed = AlpacaDataFeed(cfg)
                    for sym in symbols:
                        try:
                            raw = feed.fetch_bars(sym, n_bars=cfg.bars_lookback)
                            if raw:
                                rows = [
                                    {"timestamp_utc": b.timestamp,
                                     "open": b.open, "high": b.high,
                                     "low": b.low, "close": b.close,
                                     "volume": b.volume}
                                    for b in raw
                                ]
                                _upsert(sym, cfg.timeframe, pd.DataFrame(rows))
                        except Exception as e:
                            logger.warning("fetch_bars %s: %s", sym, e)
                except Exception as e:
                    logger.warning("AlpacaDataFeed 初始化失败 (离线?): %s", e)

                # ② 用真实策略共识打分（0-100 多空票数比）
                _cockpit_run["stage"] = "策略打分…"
                candidates = []
                try:
                    selector = ConsensusSelector(strategies=cfg.strategies)
                    candidates = selector.select(
                        universe=symbols,
                        timeframe=cfg.timeframe,
                        as_of=now,
                    )
                    logger.info("Cockpit selection: %d scored", len(candidates))
                except Exception as e:
                    logger.warning("ConsensusSelector 失败: %s", e)

                # ③ 对没有缓存数据的标的用 50 兜底（标注 no-data）
                scored_syms = {c.symbol for c in candidates}
                for i, s in enumerate(symbols):
                    if s not in scored_syms:
                        candidates.append(
                            Candidate(symbol=s, score=50.0,
                                      rank=len(candidates) + 1,
                                      reasons={"note": "no bar data"}, as_of=now)
                        )
                candidates.sort(key=lambda c: c.score, reverse=True)
                for i, c in enumerate(candidates):
                    c.rank = i + 1

                # ④ 运行 Agent Manager（LLM 层在真实 TA 分之上再做深度分析）
                _cockpit_run["stage"] = "运行 Agent…"
                ctx = AgentContext(
                    candidates=candidates, plans=[], news=[],
                    positions={}, equity=0.0, as_of=now, extra={},
                )
                mgr.run_cycle(ctx, _AI_DB)
                _cockpit_run["last_run"] = _now()
                _cockpit_run["stage"] = ""
            except Exception as exc:
                logger.error("AgentManager run_cycle 失败: %s", exc)
                _cockpit_run["stage"] = "错误"
            finally:
                _cockpit_run["running"] = False

        threading.Thread(target=_bg, daemon=True).start()

    run_btn.on_click(_do_run)

    # ── 增量更新函数（每 5s 由定时器调用）─────────────────────────────────
    def update():
        # 更新 agent 状态卡片
        states = mgr.get_agent_states(_AI_DB)
        by_role = {s["role"]: s for s in states}
        for role in _AGENT_META:
            agent_cards[role].set_content(_agent_card_html(role, by_role.get(role)))

        # 更新 Manager 决策区 picks
        scores = mgr.get_composite_scores(_AI_DB)
        if scores:
            inner = ""
            for s in scores[:6]:
                v = s["verdict"].lower()
                cls = "buy" if v == "buy" else ("avoid" if v == "avoid" else "watch")
                vc = {"buy": "var(--pos)", "avoid": "var(--neg)", "watch": "var(--warn)"}.get(cls, "var(--fg2)")
                inner += (
                    f'<div class="cp-pick {cls}">'
                    f'<div style="font-size:15px;font-weight:800;color:var(--fg)">{s["symbol"]}</div>'
                    f'<div style="font-size:10.5px;font-weight:700;margin:2px 0;color:{vc}">{s["verdict"]}</div>'
                    f'<div style="font-size:11px;color:var(--fg3)">综合 {s["composite_score"]:.0f}</div>'
                    f'</div>'
                )
            picks_html.set_content(f'<div class="cp-pick-row">{inner}</div>')

        # 更新运行状态 UI
        if _cockpit_run["running"]:
            stage = _cockpit_run.get("stage", "运行中…") or "运行中…"
            status_lbl.set_text(stage)
            status_lbl.style("color:var(--ai)")
            run_btn.props("disable")
        else:
            lr = _cockpit_run.get("last_run")
            stage = _cockpit_run.get("stage", "")
            if stage == "错误":
                status_lbl.set_text("运行出错，见日志")
                status_lbl.style("color:var(--neg)")
            elif lr:
                status_lbl.set_text(f"完成 {lr.strftime('%H:%M:%S')}")
                status_lbl.style("color:var(--fg3)")
            else:
                status_lbl.set_text("空闲")
                status_lbl.style("color:var(--fg3)")
            run_btn.props(remove="disable")

        # 更新活动流
        feed_el.set_content(_feed_html(mgr.get_recent_advisories(_AI_DB, n=20)))

    update()
    return update


def _render_universe():
    _page_head("选股池", "全市场横截面打分 · 由 DuckDB 直查 bars/ 实现", badge="demo")
    ui.html('<div class="qa-note">⚠ 占位示例。选股层将用 DuckDB 对全市场 Parquet 做横截面打分，'
            '此处展示最终形态。数据非真实，<b>不构成投资建议</b>。</div>')
    with ui.element("div").classes("qa-card"):
        ui.label("横截面因子打分（示例）").classes("qa-card-title")
        cols = [("code", "标的", "left"), ("ai", "AI 评分", "right"), ("mom", "动量", "center"),
                ("val", "估值", "center"), ("qual", "质量", "center"), ("vol", "波动", "center")]
        t = _make_table(cols)
        _fill_table(t, pd.DataFrame([
            {"code": "示例-A", "ai": 92, "mom": "强", "val": "弱", "qual": "强", "vol": "中"},
            {"code": "示例-B", "ai": 87, "mom": "强", "val": "中", "qual": "强", "vol": "低"},
            {"code": "示例-C", "ai": 74, "mom": "中", "val": "优", "qual": "强", "vol": "低"},
            {"code": "示例-D", "ai": 58, "mom": "弱", "val": "中", "qual": "中", "vol": "高"},
            {"code": "示例-E", "ai": 34, "mom": "弱", "val": "弱", "qual": "中", "vol": "高"},
        ]), cols)
    return None


def _render_models():
    _page_head("模型", "信号源管理 · 策略权重与表现", badge="demo")
    ui.html('<div class="qa-note">⚠ 占位示例。模型层将管理各信号源（TA 策略 / 未来 AI 模型）的权重与表现归因。'
            '数据非真实。</div>')
    with ui.element("div").classes("qa-card"):
        ui.label("信号源权重（示例）").classes("qa-card-title")
        cols = [("name", "信号源", "left"), ("w", "权重", "right"),
                ("acc", "准确率", "right"), ("sharpe", "夏普", "right")]
        t = _make_table(cols)
        _fill_table(t, pd.DataFrame([
            {"name": "5/20 均线金叉死叉", "w": "0.15", "acc": "58%", "sharpe": "1.34"},
            {"name": "RSI 震荡战法", "w": "0.12", "acc": "64%", "sharpe": "1.78"},
            {"name": "MACD 零轴战法", "w": "0.10", "acc": "52%", "sharpe": "0.96"},
            {"name": "布林带突破", "w": "0.08", "acc": "60%", "sharpe": "1.45"},
            {"name": "其余 20 个策略", "w": "0.55", "acc": "~55%", "sharpe": "~1.12"},
        ]), cols)
    return None


# ═══════════════════════════════════════════════════════════════════════════
# 决策台辅助函数
# ═══════════════════════════════════════════════════════════════════════════

_AGENT_META = {
    "technical":    ("📈", "Technical Agent",  "bars + TA 指标 → LLM 打分", False),
    "news":         ("📰", "News Agent",       "RSS + yfinance → 情绪评分", False),
    "web_research": ("🌐", "WebResearch",      "Agent-Reach + LLM → 热点报告", False),
    "bull_bear":    ("⚖️",  "Bull/Bear Debate", "LLM 三轮辩论 → 最终裁决", False),
    "fundamental":  ("📊", "Fundamental",      "yfinance P/E EPS → 基本面", True),
    "sentiment":    ("💬", "Sentiment Agent",  "Reddit/Twitter → 社区情绪", True),
}

_TAG_COLOR = {
    "technical": "#3fb950", "news": "#58a6ff", "web_research": "#79c0ff",
    "bull_bear": "#d29922", "orchestrator": "#9370db",
    "fundamental": "#f0883e", "sentiment": "#ea4aaa",
}


def _agent_card_html(role: str, state: dict | None) -> str:
    icon, name, desc, stub = _AGENT_META.get(role, ("?", role, "", False))
    if state is None:
        status, score, last_run_str, summary = "idle", 0.0, "—", {}
    else:
        status = state.get("status", "idle")
        score = float(state.get("last_score") or 0)
        lr = state.get("last_run")
        last_run_str = lr.strftime("%H:%M:%S") if hasattr(lr, "strftime") else (str(lr)[:8] if lr else "—")
        summary = state.get("summary") or {}

    color = {
        "done": "#3fb950", "running": "#58a6ff",
        "error": "#f85149", "timeout": "#d29922",
    }.get(status, "#6e7681")
    border = color if status not in ("idle",) else "var(--border)"
    pulse = "animation:cp-pulse 1.2s infinite" if status == "running" else ""

    bar_w = min(int(score), 100)
    sym = summary.get("symbol", "")
    if sym:
        score_key = next(
            (k for k in ("technical_score", "news_score", "hotspot_score", "final_score") if k in summary),
            "",
        )
        score_val = summary.get(score_key, "")
        trend = summary.get("trend", summary.get("sentiment", summary.get("verdict", "")))
        label = score_key.replace("_score", "")
        summary_txt = f"{sym}: {label}={score_val}{f' {trend}' if trend else ''}"[:48]
    else:
        summary_txt = "🔧 待实现" if stub else ("待运行" if status == "idle" else status)

    stub_note = '<div style="color:#d29922;font-size:10px;margin-bottom:5px">🔧 本轮待实现</div>' if stub else ""

    return (
        f'<div style="background:var(--panel);border:1px solid {border};'
        f'border-radius:12px;padding:15px;min-height:148px;box-sizing:border-box">'
        f'<div style="display:flex;align-items:center;gap:7px;margin-bottom:8px">'
        f'<span style="font-size:16px">{icon}</span>'
        f'<span style="font-size:12.5px;font-weight:700;color:var(--fg)">{name}</span>'
        f'<span style="width:7px;height:7px;border-radius:50%;background:{color};'
        f'display:inline-block;margin-left:auto;{pulse}"></span>'
        f'</div>'
        f'<div style="font-size:11px;color:var(--fg3);margin-bottom:8px;line-height:1.4">{desc}</div>'
        f'{stub_note}'
        f'<div style="font-size:11.5px;color:var(--fg2);margin-bottom:8px;word-break:break-all">{summary_txt}</div>'
        f'<div style="background:var(--border);border-radius:3px;height:3px;margin-bottom:7px">'
        f'<div style="background:{color};border-radius:3px;height:3px;width:{bar_w}%;transition:width .5s"></div>'
        f'</div>'
        f'<div style="display:flex;justify-content:space-between;font-size:10px;color:var(--fg3)">'
        f'<span>{status}</span><span>{last_run_str}</span>'
        f'</div>'
        f'</div>'
    )


def _feed_html(advisories: list) -> str:
    if not advisories:
        return ('<div class="cp-feed">'
                '<span style="color:var(--fg3);font-size:12px">运行后显示活动记录</span>'
                '</div>')
    items = ""
    for a in advisories:
        ts = a["created_at"]
        ts_str = ts.strftime("%H:%M:%S") if hasattr(ts, "strftime") else (str(ts)[:8] if ts else "—")
        agent = a["agent"]
        p = a["payload"]
        sym = p.get("symbol", "")
        k = a["kind"]
        if k == "technical":
            text = f"{sym} 技术分={p.get('technical_score','?')} {p.get('trend','')}"
        elif k == "news":
            text = f"{sym} 新闻分={p.get('news_score','?')} ({p.get('sentiment','?')})"
        elif k == "bull_bear_debate":
            text = f"{sym} → {p.get('verdict','?')} score={p.get('final_score','?')}"
        elif k == "web_research":
            text = f"{sym} 热点分={p.get('hotspot_score','?')} {p.get('sentiment','')}"
        elif k == "orchestrator_summary":
            top = p.get("top_pick") or {}
            text = f"汇总 {p.get('sub_advisory_count',0)} 条 top={top.get('symbol','—')}"
        else:
            text = str(p)[:70]
        color = _TAG_COLOR.get(agent, "#8b949e")
        items += (
            f'<div class="cp-feed-row">'
            f'<span class="cp-ts">{ts_str}</span>'
            f'<span class="cp-tag" style="background:{color}22;color:{color}">{agent}</span>'
            f'<span style="color:var(--fg)">{text[:80]}</span>'
            f'</div>'
        )
    return f'<div class="cp-feed">{items}</div>'


# 决策台运行状态（模块级，跨导航切换保持）
_cockpit_run = {"running": False, "last_run": None, "stage": ""}


_RENDERERS = {
    "overview": _render_overview,
    "activity": _render_activity,
    "system": _render_system,
    "research": _render_research,
    "cockpit": _render_cockpit,
    "universe": _render_universe,
    "models": _render_models,
}


def _select(name: str):
    if name not in _RENDERERS:
        name = "overview"
    _state["tab"] = name
    _state["updater"] = None
    _set_pref("active_tab", name)
    for k, el in _nav_refs.items():
        if k == name:
            el.classes(add="active")
        else:
            el.classes(remove="active")
    content.clear()
    with content:
        _state["updater"] = _RENDERERS[name]()

# ═══════════════════════════════════════════════════════════════════════════
# 定时器：顶栏 + 当前页增量更新（绝不 clear+rebuild，所以不闪）
# ═══════════════════════════════════════════════════════════════════════════

def _update_topbar():
    running = _engine_running()
    top_engine.set_text("● 运行中" if running else "○ 停止")
    top_engine.classes(remove="pos neg", add="pos" if running else "neg")
    hb = heartbeat()
    if hb is not None:
        secs = (datetime.now(timezone.utc) - hb).total_seconds()
        top_hb.set_text(f"{secs:.0f}s" if secs < 120 else f"{secs/60:.0f}m")
    else:
        top_hb.set_text("—")
    # 优先用 Alpaca API 实时权益；DuckDB 仅提供 24h 盈亏起点
    live = live_alpaca_equity()
    if live is not None:
        top_total.set_text(_money(live["equity"]))
        eq = equity_df(24)
        if not eq.empty and "total_equity" in eq.columns:
            pnl = live["equity"] - float(eq["total_equity"].iloc[0])
            top_pnl.set_text(f"{pnl:+,.0f}")
            top_pnl.classes(remove="pos neg", add="pos" if pnl >= 0 else "neg")
        else:
            top_pnl.set_text("—")
    else:
        eq = equity_df(24)
        if not eq.empty and "total_equity" in eq.columns:
            total = float(eq["total_equity"].iloc[-1])
            pnl = total - float(eq["total_equity"].iloc[0])
            top_total.set_text(_money(total))
            top_pnl.set_text(f"{pnl:+,.0f}")
            top_pnl.classes(remove="pos neg", add="pos" if pnl >= 0 else "neg")
        else:
            top_total.set_text("—")
            top_pnl.set_text("—")


def _tick():
    try:
        _update_topbar()
    except Exception:
        pass
    upd = _state.get("updater")
    if upd:
        try:
            upd()
        except Exception:
            pass


_select(_pref("active_tab", "overview"))
_update_topbar()
ui.timer(_REFRESH_SEC, _tick)

if __name__ in {"__main__", "__mp_main__"}:
    _web = os.getenv("QUANT_WEB") == "1" or "--web" in sys.argv
    ui.run(
        title="美股K线 · DuckDB + Alpaca 实时",
        reload=False,
        native=not _web,
        port=8080 if _web else None,
        show=_web,
        window_size=(1600, 1000) if not _web else None,
        dark=True,
    )
