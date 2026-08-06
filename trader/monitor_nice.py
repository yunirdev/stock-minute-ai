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

import html
import json
import inspect
import logging
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# yfinance 默认 hide_exceptions=True：取不到数据（如 ETF/指数没有 fundamentals、
# sector 等字段）时不抛异常，而是自己用 logging.getLogger("yfinance").error(...)
# 直接打到控制台 —— 跟我们代码里的 try/except 无关，必须单独调高它的日志级别屏蔽。
# 这不影响真实报错：我们自己的 try/except 仍按各自逻辑处理并返回兜底值。
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from nicegui import app, ui  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from trader.monitor_data import (  # noqa: E402
    DB_PATH,
    equity_df,
    episode_review_outcome_counts,
    explain_order,
    fills_df,
    heartbeat,
    latest_paper_maturity_status,
    latest_strategy_release_event,
    recent_news,
    live_alpaca_equity,
    live_alpaca_positions,
    latest_reconciliation,
    orders_df,
    plan_risk_events_df,
    render_order_explanation_html,
    signals_df,
)
from trader.operations_observability import (  # noqa: E402
    AsyncButtonActionRunner,
    ActionJobResult,
    BUTTON_ACTIONS,
    ButtonActionAuditStore,
    button_contract_manifest,
    queue_view,
)

if sys.platform == "win32":
    os.environ.setdefault(
        "PYTHONIOENCODING", "utf-8"
    )  # 影响本进程之后 spawn 的子进程（如引擎）
    # 但子进程继承环境变量这件事，对本进程自己已经打开的 stdout/stderr 没用——
    # 控制台默认编码（cp1252/GBK 等）打中文会直接 UnicodeEncodeError 崩溃退出
    # （main.py 之前踩过这个坑），换新机器、用默认 cmd.exe 最容易复现，必须显式 reconfigure。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

_LOG_FILE = _ROOT / "trader_engine.log"
_PID_FILE = _ROOT / ".engine.pid"
_PREFS_PATH = _ROOT / "conf" / "ui_settings.json"
_REFRESH_SEC = 5.0
_AI_DB = str(_ROOT / "ai_states.duckdb")
_DAILY_DB_RAW = Path(os.getenv("DAILY_RESEARCH_DB", "ai_states.duckdb"))
_DAILY_RESEARCH_DB = str(
    _DAILY_DB_RAW if _DAILY_DB_RAW.is_absolute() else _ROOT / _DAILY_DB_RAW
)

from trader.ui_health import (  # noqa: E402
    UI_HEALTH_SCRIPT,
    UIHealthReport,
    format_health_age,
    record_ui_health,
)
from trader.monitor_research_view import live_research_html  # noqa: E402
from trader.research_monitor import (  # noqa: E402
    list_debate_records,
    live_monitor_snapshot,
)
from trader.runtime_status import read_runtime_status  # noqa: E402


@app.get("/healthz")
def _healthz() -> dict:
    return {"status": "ok", "component": "monitor_nice"}


@app.post("/api/ui-health/report")
def _ui_health_report(report: UIHealthReport) -> dict:
    try:
        fingerprint = record_ui_health(report, DB_PATH)
        return {"accepted": True, "fingerprint": fingerprint}
    except Exception as exc:
        logger.warning("UI health report persistence failed: %s", type(exc).__name__)
        return {"accepted": False}


@app.get("/api/ui-actions")
def _ui_actions_contract() -> dict:
    actions = button_contract_manifest()
    return {"action_count": len(actions), "actions": actions}


@app.get("/api/order-explanation/{plan_id}")
def _order_explanation(plan_id: str) -> dict:
    return explain_order(DB_PATH, plan_id)


_ACTION_AUDIT_STORE = ButtonActionAuditStore(DB_PATH)
_ACTIVE_UI_ACTIONS: dict[str, str] = {}
_ASYNC_ACTION_RUNNER = AsyncButtonActionRunner(_ACTION_AUDIT_STORE)


def _audited_callback(action_id: str, callback):
    """Persist one handler dispatch and fail closed on duplicate in-flight calls."""

    async def _wrapped():
        request_id = _ACTIVE_UI_ACTIONS.get(action_id) or str(uuid.uuid4())
        started = _ACTION_AUDIT_STORE.begin(
            action_id,
            request_id,
            now=datetime.now(timezone.utc),
        )
        if action_id in _ACTIVE_UI_ACTIONS:
            return started
        _ACTIVE_UI_ACTIONS[action_id] = request_id
        try:
            result = callback()
            if inspect.isawaitable(result):
                result = await result
            empty_result = isinstance(result, (list, dict, tuple, set)) and not result
            return _ACTION_AUDIT_STORE.finish(
                started["action_run_id"],
                state="EMPTY" if empty_result else "SUCCESS",
                result_ref=f"ui-handler:{action_id}:{request_id}",
                user_message="动作已完成" if result is not None else "动作已接收",
                now=datetime.now(timezone.utc),
            )
        except Exception as exc:
            _ACTION_AUDIT_STORE.finish(
                started["action_run_id"],
                state="ERROR",
                result_ref="",
                user_message=f"操作失败: {type(exc).__name__}",
                now=datetime.now(timezone.utc),
            )
            raise
        finally:
            _ACTIVE_UI_ACTIONS.pop(action_id, None)

    return _wrapped


def _push_result_html(label: str, outcome) -> str:
    """把推送结果渲染成状态条。

    以前只有"✓ 已发送 / ✗ 发送失败"两档，而 send() 在没配 Discord 时返回的
    是 console 打印的结果（True），于是这里稳稳地显示绿色的"已发送"——用户
    完全没法知道消息其实一个字都没发出去。现在三种非成功状态各有各的说法。
    """
    status = getattr(outcome, "status", "SENT" if outcome else "FAILED")
    style = {
        "SENT": ("#3fb950", f"✓ {label}已发送"),
        "DRY_RUN": ("#d29922", f"⚠ {label}未发送：未配置 Discord，仅打印到本地日志"),
        "BLOCKED": ("#d29922", f"⚠ {label}未发送：外发未授权"),
        "FAILED": ("#f85149", f"✗ {label}发送失败"),
    }.get(status, ("#f85149", f"✗ {label}发送失败（{status}）"))
    color, text = style
    detail = getattr(outcome, "detail", "")
    if detail and status == "FAILED":
        text += f"（{detail}）"
    return f'<span style="color:{color}">{text}</span>'


def _audited_job_callback(action_id: str, callback):
    """Start a worker-backed UI action and leave it BUSY until work completes."""
    return lambda: _ASYNC_ACTION_RUNNER.start(action_id, callback)


ui.add_body_html(f"<script>{UI_HEALTH_SCRIPT}</script>")

# ═══════════════════════════════════════════════════════════════════════════
# 持久化用户偏好 (纯 JSON，跨重启)
# ═══════════════════════════════════════════════════════════════════════════


# 当前代码里仍会被 _pref/_set_pref 读写的 key；不在这里的 key（旧引擎配置页、
# 旧图表探索页、旧决策台遗留）不会被保存回 conf/ui_settings.json。
_KNOWN_PREF_KEYS = {
    "sys_sym", "sys_strat", "sys_tf", "sys_int", "sys_min_score",
    "fa_sym", "fa_tf", "fa_fac", "fa_fwd", "fa_nq",
    "r_sym", "r_tf", "r_strat", "r_cap", "r_lev", "r_slip", "r_fill", "r_risk",
    "sel_source", "sel_long_n", "sel_daily_n", "sel_decision_style",
    "sel_scan_n", "sel_scan_keep_n", "sel_include_broad",
    "active_tab",
}


def _load_prefs() -> dict:
    try:
        raw = json.loads(_PREFS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {k: v for k, v in raw.items() if k in _KNOWN_PREF_KEYS}


def _save_prefs() -> None:
    try:
        _PREFS_PATH.parent.mkdir(parents=True, exist_ok=True)
        clean = {k: v for k, v in _PREFS.items() if k in _KNOWN_PREF_KEYS}
        _PREFS_PATH.write_text(
            json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8"
        )
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
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        return str(pid) in out.stdout
    except Exception:
        return False


def _start_engine(
    symbols: str,
    strategies: str,
    tf: str,
    interval,
    auto_trade: bool = False,
    min_ai_score: int = 65,
) -> str:
    if _engine_running():
        return "引擎已在运行"
    syms = ",".join(s.strip().upper() for s in symbols.split(",") if s.strip())
    # Empty strategies is valid and intentional: ConsensusSelector treats an
    # empty list as "use all STRATEGY_OPTIONS" (full 24-strategy consensus),
    # so we must pass "" through explicitly rather than fall back to
    # trader.main's single-strategy CLI default.
    strats = ",".join(s.strip() for s in strategies.split(",") if s.strip())
    if not syms:
        return "❌ 请填写标的（或先在「机会中心」生成决策池）"
    cmd = [
        sys.executable,
        "-m",
        "trader.main",
        "--symbols",
        syms,
        "--strategies",
        strats,
        "--tf",
        tf,
        "--interval",
        str(int(interval)),
    ]
    if auto_trade:
        cmd += ["--auto-trade", "--min-ai-score", str(int(min_ai_score))]
    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        log_fh = open(_LOG_FILE, "w", encoding="utf-8", buffering=1)
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            cwd=str(_ROOT),
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
            subprocess.call(
                ["taskkill", "/F", "/PID", pid],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
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
        return (
            "\n".join(
                _LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()[
                    -n:
                ]
            )
            or "（日志为空）"
        )
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

/* 每日研究/Runtime 监控用的是原生 <table>（不是 NiceGUI ui.table），
   上面那组 .q-table 规则管不到它——一直没有专门的样式，表头和数据
   全靠浏览器默认间距对齐，长短不一就会看着错位。*/
.qa-table{width:100%;border-collapse:collapse;font-size:12.5px;}
.qa-table th,.qa-table td{padding:7px 10px;text-align:left;white-space:nowrap;
  border-bottom:1px solid var(--border);}
.qa-table th{color:var(--fg3);font-size:11px;font-weight:600;
  text-transform:uppercase;letter-spacing:.04em;}
.qa-table td{color:var(--fg);}
.qa-table tbody tr:hover{background:var(--panel2);}
.qa-table tbody tr:last-child td{border-bottom:none;}

/* 每日研究表——标的/AI结论就两三个字，压窄让给说明；说明栏是 AI 写的
   多行简报，要能换行，不能跟别的列一样强制 nowrap。 */
.qa-table-research th:nth-child(1),.qa-table-research td:nth-child(1){width:9%;white-space:nowrap;}
.qa-table-research th:nth-child(2),.qa-table-research td:nth-child(2){width:13%;white-space:nowrap;}
.qa-table-research th:nth-child(3),.qa-table-research td:nth-child(3){width:78%;white-space:normal;line-height:1.55;}

.qa-symbol-link{color:var(--ai);text-decoration:none;cursor:pointer;}
.qa-symbol-link:hover{text-decoration:underline;}

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

/* 权益曲线时间跨度选择器 */
.qa-span-btn{font-size:11px;padding:2px 8px;border-radius:5px;
  border:1px solid var(--border);background:transparent;color:var(--fg3);
  cursor:pointer;line-height:1.8;transition:all .15s;font-family:var(--mono);}
.qa-span-btn:hover{color:var(--fg);border-color:var(--fg2);}
.qa-span-btn.sp-active{background:rgba(88,166,255,.15);color:var(--ai);
  border-color:rgba(88,166,255,.45);font-weight:600;}

/* 决策池详细报告 */
.cp-report-wrap{display:flex;flex-direction:column;gap:8px;width:100%}
.cp-report-sym{background:var(--panel);border:1px solid var(--border);border-radius:10px;overflow:hidden}
.cp-report-sym[open]{border-color:rgba(88,166,255,.4)}
.cp-report-summary{display:flex;align-items:center;gap:10px;padding:13px 16px;
  cursor:pointer;list-style:none;user-select:none}
.cp-report-summary::-webkit-details-marker{display:none}
.cp-report-summary:hover{background:var(--panel2)}
.cp-report-sym-name{font-size:15px;font-weight:800;color:var(--fg);min-width:56px}
.cp-report-verdict-badge{font-size:11px;font-weight:700;padding:2px 9px;border-radius:999px;background:rgba(255,255,255,.06)}
.cp-report-composite{font-size:12px;color:var(--fg3);font-family:var(--mono)}
.cp-report-chevron{margin-left:auto;color:var(--fg3);font-size:18px;transition:transform .2s}
.cp-report-sym[open] .cp-report-chevron{transform:rotate(90deg)}
.cp-report-body{border-top:1px solid var(--border)}
.cp-report-section{padding:12px 16px;border-bottom:1px solid #21262d}
.cp-report-section:last-child{border-bottom:none}
.cp-report-sec-title{font-size:12.5px;font-weight:700;color:var(--fg2);margin-bottom:6px}
.cp-report-score-badge{font-size:11px;font-family:var(--mono);background:rgba(88,166,255,.12);
  color:var(--ai);padding:1px 7px;border-radius:999px;margin-left:4px}
.cp-report-meta{font-size:11px;color:var(--fg3);font-family:var(--mono);line-height:1.6}
.cp-report-reasoning{font-size:12px;color:var(--fg2);margin-top:6px;line-height:1.5;
  background:var(--panel2);padding:8px 10px;border-radius:6px}
.cp-report-key-factor{font-size:12px;color:var(--warn);margin-bottom:4px;font-weight:600}
.cp-report-suggested{font-size:12px;color:var(--fg2);margin-bottom:8px;font-style:italic}
.cp-report-debate-row{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:8px}
.cp-report-bull-box{background:rgba(63,185,80,.06);border:1px solid rgba(63,185,80,.2);border-radius:8px;padding:10px 12px}
.cp-report-bear-box{background:rgba(248,81,73,.06);border:1px solid rgba(248,81,73,.2);border-radius:8px;padding:10px 12px}
.cp-report-debate-label{font-size:11px;font-weight:700;margin-bottom:5px}
.cp-report-thesis{font-size:11.5px;color:var(--fg2);line-height:1.45}
.cp-report-rtag{display:inline-block;font-size:10px;padding:2px 7px;border-radius:999px;
  margin:2px 3px 2px 0;background:rgba(88,166,255,.1);color:var(--ai)}
.cp-report-rtag.pos{background:rgba(63,185,80,.1);color:var(--pos)}
.cp-report-rtag.neg{background:rgba(248,81,73,.1);color:var(--neg)}
.cp-report-rtag.warn{background:rgba(210,153,34,.1);color:var(--warn)}

@media (max-width: 700px) {
  .qa-topbar{gap:8px;padding:0 8px;overflow:hidden;}
  .qa-brand{display:none;}
  .qa-spacer{display:none;}
  .qa-stat{min-width:68px;}
  .qa-stat .l{font-size:9px;}
  .qa-stat .v{font-size:12px;}
  .qa-nav{width:56px;min-width:56px;}
  .qa-nav-group{display:none;}
  .qa-nav-item{justify-content:center;padding:11px 8px;gap:0;}
  .qa-nav-item span:not(.ico){display:none;}
  .qa-nav-item .ico{font-size:17px;width:22px;}
  .qa-content{padding:12px;gap:12px;min-width:0;}
  .qa-card{padding:12px;overflow-x:auto;}
  .qa-kpi{min-width:100%;}
  .qa-content [style*="min-width"]{min-width:0!important;}
}
"""

# ═══════════════════════════════════════════════════════════════════════════
# 页面骨架
# ═══════════════════════════════════════════════════════════════════════════

ui.add_head_html("<style>" + _CSS + "</style>")

_state: dict = {"tab": "overview", "updater": None}
_nav_refs: dict = {}
#: 点"每日研究"表里的标的名 → 跳转到"研究档案"时带过去的筛选条件。
#: _select() 每次都整卡重建，读一次就清空，不然下次进研究档案会莫名带着筛选。
_pending_archive_filter: dict = {}

with ui.element("div").classes("qa-topbar"):
    ui.html(
        '<span class="qa-brand">美股<span class="dot">K线</span>'
        '<span style="font-weight:400;color:var(--fg3);font-size:12px;margin-left:6px">'
        "DuckDB · Alpaca 实时</span></span>"
    )
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
    with ui.element("div").classes("qa-stat"):
        ui.label("AutoTrade").classes("l")
        top_autotrade = ui.label("—").classes("v")
    with ui.element("div").classes("qa-stat").style("cursor:pointer") as top_tasks_stat:
        ui.label("任务队列").classes("l")
        top_tasks = ui.label("—").classes("v")
    top_tasks_stat.on("click", lambda: _select("tasks"))

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

        _nav_group("工作台")
        _nav_item("overview", "📊", "今日总览")
        _nav_item("selection", "🔭", "机会中心")
        _nav_item("activity", "🧾", "交易与持仓")
        _nav_item("research", "🔬", "研究验证")
        _nav_item("operations", "⚙️", "系统运营")

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
    cols = [
        {"name": f, "label": label, "field": f, "align": a} for f, label, a in col_specs
    ]
    return (
        ui.table(columns=cols, rows=[], row_key="__i", pagination=0)
        .props("flat dense")
        .classes("w-full")
    )


def _fill_table(
    table, df: pd.DataFrame, col_specs: list, fmts=None, max_rows: int = 12
):
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
                row[f] = (
                    "—"
                    if v is None or (isinstance(v, float) and pd.isna(v))
                    else str(v)
                )
        rows.append(row)
    table.rows = rows
    table.update()


# ── 基准指数（yfinance，5 min TTL） ───────────────────────────────────────────
# 只用 SPY（大盘基准）和 QQQ（科技/成长对标）
# 这是量化基金、散户组合管理的行业共识，IWM 是小盘基金基准，此处不适用
_BENCHMARKS = ["SPY", "QQQ"]
_BENCH_COLORS = {"SPY": "#c9d1d9", "QQQ": "#d29922"}  # 灰白 / 橙黄
_BENCH_CACHE: dict = {}  # key=(sym, span_key) → (fetched_at, df|None)
_BENCH_CACHE_TTL = 300  # 5 min

# span_key → (equity_db_hours, yf_period, yf_interval)
# 覆盖标准时间跨度：日内 / 近一周 / 近一月 / 近三月 / 年初至今 / 全部
_SPAN_CFG = {
    "1D": (24, "2d", "15m"),
    "1W": (24 * 7, "7d", "60m"),
    "1M": (24 * 30, "1mo", "1d"),
    "3M": (24 * 90, "3mo", "1d"),
    "YTD": (None, "ytd", "1d"),  # hours=None → 按日历算
    "All": (9999, "max", "1wk"),
}


def _ytd_hours() -> int:
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    jan1 = datetime(now.year, 1, 1, tzinfo=timezone.utc)
    return max(1, int((now - jan1).total_seconds() / 3600))


def _fetch_benchmark(sym: str, span_key: str) -> "pd.DataFrame | None":
    """yfinance 收盘价，5 min 缓存，返回 tz-naive 美东（ET）DatetimeIndex——
    整个仪表盘的时间口径统一用美东，跟"研究交易日""收盘报告"这些美东框定
    的内容对得上，不再是内部自洽但跟别处对不上的 UTC。"""
    import time as _time

    key = (sym, span_key)
    now_s = _time.time()
    cached = _BENCH_CACHE.get(key)
    if cached and (now_s - cached[0]) < _BENCH_CACHE_TTL:
        return cached[1]
    try:
        import yfinance as yf

        _, period, interval = _SPAN_CFG[span_key]
        hist = yf.Ticker(sym).history(
            period=period, interval=interval, auto_adjust=True
        )
        if hist.empty:
            _BENCH_CACHE[key] = (now_s, None)
            return None
        if hist.index.tz is not None:
            hist.index = hist.index.tz_convert("America/New_York").tz_localize(None)
        _BENCH_CACHE[key] = (now_s, hist[["Close"]])
        return hist[["Close"]]
    except Exception:
        _BENCH_CACHE[key] = (now_s, None)
        return None


def _norm_pct(series: "pd.Series") -> "pd.Series":
    """期初对齐到 0% 的归一化收益率。"""
    base = float(series.iloc[0])
    return (series.astype(float) - base) / base * 100 if base else series * 0


def _last_pct(series: "pd.Series") -> float:
    """期末 % 收益（期初为 0%）。"""
    return float(_norm_pct(series).iloc[-1]) if len(series) >= 2 else 0.0


def _align_bench(eq: pd.DataFrame, bench: dict) -> dict:
    """
    将基准数据裁剪到与组合权益数据相同的起始时间，确保 0% 基准一致。

    关键细节（现在整条链路都是 tz-naive 美东时间，不是 UTC）：
    - yfinance 日线数据本来就是美东午夜（00:00 ET）打的时间戳
    - 权益快照时间戳是盘中时间（如 10:30 ET）
    - 若用精确时间对齐，当天日线 bar（00:00）< 权益起点（10:30）→ 被排除
    - 修复：日线数据检测为"小时数 < 7 的 bar"→ 用当天零点 ET 对齐
      （这个阈值本来是为了兼容"转 UTC 后 04:00/05:00"的旧逻辑留的余量，
      现在日线本来就是 00:00，< 7 依然成立，不用改）
    """
    if eq is None or eq.empty or "total_equity" not in eq.columns:
        return bench
    first_ts = pd.Timestamp(pd.to_datetime(eq["ts"]).iloc[0])
    aligned: dict = {}
    for sym, hist in bench.items():
        if hist is None or hist.empty:
            continue
        idx = hist.index
        # 检测日线 / 周线：美东午夜附近（0 点）；分钟线 / 小时线：盘中 9-16 点 ET
        sample_hours = [t.hour for t in idx[: min(10, len(idx))]]
        is_daily = max(sample_hours) < 7  # True = 日线或周线

        if is_daily:
            # 日线对齐到当天零点 ET，否则 00:00 < 10:30 → 当天 bar 被误切
            cutoff = first_ts.normalize()  # 2026-05-15 10:32 → 2026-05-15 00:00
        else:
            cutoff = first_ts  # 分钟线用精确时间

        trimmed = hist[idx >= cutoff]
        if not trimmed.empty:
            aligned[sym] = trimmed
        # 若窗口内无数据（如周末查 1D 分钟线），跳过，不显示基准
    return aligned


def _perf_chips_html(eq_pct: float, bench: dict) -> str:
    """
    生成超额收益摘要 HTML。
    目的：一行读出"我赚了多少、市场赚了多少、我跑赢/跑输多少"。
    """

    def chip(label: str, val: float, color: str) -> str:
        sign = "+" if val >= 0 else ""
        return (
            f'<span style="background:rgba(255,255,255,.04);border:1px solid #30363d;'
            f'border-radius:5px;padding:2px 8px;font-family:var(--mono);font-size:11px">'
            f'<span style="color:{color}">{label}</span> '
            f'<b style="color:{"#3fb950" if val >= 0 else "#f85149"}">{sign}{val:.2f}%</b>'
            f"</span>"
        )

    parts = [chip("我的组合", eq_pct, "#58a6ff")]
    spy_pct = None
    for sym, hist in bench.items():
        if hist is None or hist.empty:
            continue
        p = _last_pct(hist["Close"])
        color = _BENCH_COLORS.get(sym, "#8b949e")
        parts.append(chip(sym, p, color))
        if sym == "SPY":
            spy_pct = p

    if spy_pct is not None:
        alpha = eq_pct - spy_pct
        sign = "+" if alpha >= 0 else ""
        alpha_color = "#3fb950" if alpha >= 0 else "#f85149"
        parts.append(
            f'<span style="background:rgba(255,255,255,.04);border:1px solid #30363d;'
            f'border-radius:5px;padding:2px 8px;font-size:11px">'
            f'<span style="color:var(--fg3)">超额 vs SPY</span> '
            f'<b style="color:{alpha_color}">{sign}{alpha:.2f}%</b>'
            f"</span>"
        )
    return (
        '<div style="display:flex;flex-wrap:wrap;gap:6px;padding:6px 0 10px">'
        + "".join(parts)
        + "</div>"
    )


def _equity_fig(eq: pd.DataFrame, bench: dict, uirev: str) -> go.Figure:
    """
    组合收益率 vs 基准（SPY / QQQ），全部归一化到期初 0%。

    设计原则：
    - 唯一 Y 轴 = % 收益率，所有线可直接比高低
    - 组合（蓝色实线）作为主角，填充到 0% 基准线，正收益蓝色调，亏损一样显示
    - SPY（灰色实线）= 主要对标，买大盘 ETF 能赚多少
    - QQQ（橙色虚线）= 科技/成长对标，看组合偏向价值还是成长
    - 0% 水平线 = "躺平不动"基准
    - Y 轴自适应数据范围，不锁定到 0
    """
    fig = go.Figure()
    has_eq = eq is not None and not eq.empty and "total_equity" in eq.columns

    if has_eq:
        ts = pd.to_datetime(eq["ts"])
        eq_pct = _norm_pct(eq["total_equity"])
        # 填充到 0%（"期初平线"）：正区间蓝填充 = 盈利感，负区间填充到零 = 亏损感
        fig.add_trace(
            go.Scatter(
                x=ts,
                y=eq_pct,
                mode="lines",
                line=dict(width=2.5, color="#58a6ff"),
                name="我的组合",
                fill="tozeroy",
                fillcolor="rgba(88,166,255,0.08)",
                hovertemplate="<b>我的组合</b>  %{x|%m-%d %H:%M} ET<br><b>%{y:+.2f}%</b><extra></extra>",
            )
        )

    # SPY：实线灰色，主要视觉参照物
    spy_hist = bench.get("SPY")
    if spy_hist is not None and not spy_hist.empty:
        fig.add_trace(
            go.Scatter(
                x=spy_hist.index,
                y=_norm_pct(spy_hist["Close"]),
                mode="lines",
                line=dict(width=1.5, color="#c9d1d9"),
                name="SPY（S&P500）",
                hovertemplate="<b>SPY</b>  %{x|%m-%d}<br>%{y:+.2f}%<extra></extra>",
            )
        )

    # QQQ：虚线橙色，科技/成长参照
    qqq_hist = bench.get("QQQ")
    if qqq_hist is not None and not qqq_hist.empty:
        fig.add_trace(
            go.Scatter(
                x=qqq_hist.index,
                y=_norm_pct(qqq_hist["Close"]),
                mode="lines",
                line=dict(width=1.2, color="#d29922", dash="dash"),
                name="QQQ（纳指100）",
                hovertemplate="<b>QQQ</b>  %{x|%m-%d}<br>%{y:+.2f}%<extra></extra>",
            )
        )

    # 0% 参考线 = "什么都不做"
    if fig.data:
        fig.add_hline(y=0, line_dash="dot", line_color="#444c56", line_width=1)

    fig.update_layout(
        height=300,
        margin=dict(l=8, r=8, t=4, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8b949e", size=11),
        showlegend=True,
        legend=dict(
            orientation="h",
            x=0,
            y=1.14,
            xanchor="left",
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=10),
            traceorder="normal",
        ),
        uirevision=uirev,
        xaxis=dict(
            gridcolor="#21262d",
            showgrid=True,
            tickfont=dict(size=10),
            title=dict(text="美东时间 (ET)", font=dict(size=10)),
        ),
        yaxis=dict(
            gridcolor="#21262d",
            showgrid=True,
            rangemode="normal",  # Y 轴贴合数据，不从 0 开始
            tickformat="+.2f",
            ticksuffix="%",
            tickfont=dict(size=10),
            zeroline=False,  # 用 add_hline 画 0% 线，不用轴自带
        ),
    )
    if not fig.data:
        fig.update_layout(showlegend=False)
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.add_annotation(
            text="暂无权益记录 — 在「系统」页启动引擎后将自动写入",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"color": "#6e7681", "size": 13},
        )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 各页渲染 —— 实时页返回 update 函数；静态页返回 None
# ═══════════════════════════════════════════════════════════════════════════


def _render_overview():
    _page_head("总览", "账户权益与持仓 · 数据窗口 24 小时", badge="live")

    # ── 市场环境快捷条 ───────────────────────────────────────────────────────

    def _refresh_regime():
        def _work():
            try:
                from trader.teams.market_env import run_market_env

                out = run_market_env()
                regime = out.data.get("regime")
                if regime:
                    vix_str = f"{regime.vix:.1f}" if regime.vix else "N/A"
                    spy_dev = (
                        f"{regime.spy_vs_200ma_pct:+.2f}%"
                        if regime.spy_vs_200ma_pct is not None
                        else "N/A"
                    )
                    regime_bar.set_content(
                        f'<div style="display:flex;align-items:center;gap:14px;font-size:12px">'
                        f'<span style="color:{regime.color};font-weight:700">'
                        f"{'📈' if 'bull' in regime.regime.value else ('📉' if 'bear' in regime.regime.value else '⚡')}"
                        f" {regime.label}</span>"
                        f'<span style="color:var(--fg3)">VIX <span style="color:'
                        f"{'var(--neg)' if (regime.vix or 0) > 25 else 'var(--fg)'};"
                        f'font-family:var(--mono)">{vix_str}</span></span>'
                        f'<span style="color:var(--fg3)">SPY vs 200MA <span style="color:'
                        f"{'var(--pos)' if (regime.spy_vs_200ma_pct or 0) >= 0 else 'var(--neg)'};"
                        f'font-family:var(--mono)">{spy_dev}</span></span>'
                        f'<span style="color:var(--fg3);font-size:11px">'
                        f"{regime.as_of.strftime('%H:%M UTC')}</span>"
                        f"</div>"
                    )
                return bool(regime)
            except Exception as exc:
                regime_bar.set_content(
                    f'<div style="font-size:12px;color:var(--neg)">市场环境获取失败: {exc}</div>'
                )
                raise

        return _work()

    with ui.element("div").style(
        "display:flex;align-items:center;gap:10px;padding:8px 14px;"
        "background:var(--panel);border:1px solid var(--border);"
        "border-radius:8px;margin-bottom:8px"
    ):
        regime_bar = ui.html(
            '<span style="font-size:12px;color:var(--fg3)">🌍 市场环境 — 点击「刷新」获取</span>'
        )
        ui.element("div").style("flex:1")
        ui.button(
            "刷新市场环境",
            on_click=_audited_job_callback("overview.refresh_regime", _refresh_regime),
        ).props("unelevated dense flat outline").style(
            "font-size:11px;color:var(--fg3)"
        )

    with ui.element("div").classes("qa-card"):
        with ui.row().classes("items-center gap-3"):
            ui.label("每日研究与 Runtime 实时监控").classes("qa-card-title")
            research_status = ui.label("就绪").style(
                "font-size:12px;color:var(--qa-text-muted)"
            )
            ui.element("div").style("flex:1")
            run_research_btn = ui.button("运行今日研究").props(
                "unelevated dense color=primary"
            )
        ui.html(
            '<div class="qa-note">每天盘后或盘前只运行一次 TradingAgents；'
            "盘中 Runtime 只更新价格、入场距离、持仓、订单和状态。</div>"
        )
        research_live = ui.html(
            live_research_html(live_monitor_snapshot(_DAILY_RESEARCH_DB))
        ).style("margin-top:10px")
        # ui.html() 内容会被 DOMPurify 清洗，标的名那个 <a> 上不能挂内联
        # onclick（会被剥掉，实测验证过）——改用事件委托：在 document 上挂
        # 一次监听，点击命中 .qa-symbol-link 就读 data-symbol 触发 emitEvent。
        # 挂在 document 上 + 加个标记位防重复绑定，这样研究表内容刷新/整卡
        # 重建都不用重新挂监听。
        try:
            import asyncio as _aio

            _aio.get_running_loop()  # 没有事件循环时（如模块刚加载、ui.run() 还没起服务器）
            ui.run_javascript(
                """
                if (!window.__qaSymbolClickBound) {
                    window.__qaSymbolClickBound = true;
                    document.addEventListener('click', function (ev) {
                        const link = ev.target.closest('.qa-symbol-link');
                        if (link && link.dataset.symbol) {
                            emitEvent('research_symbol_click', link.dataset.symbol);
                        }
                    });
                }
                """
            )  # 直接跳过，否则会创建一个永远没人 await 的协程
        except Exception:
            pass

        def _run_daily_research_now():
            run_research_btn.props("disable")
            research_status.set_text("研究运行中…")

            def _research_work():
                try:
                    from trader.config import settings as _settings
                    from trader.daily_research import (
                        DailyResearchService,
                        DailyResearchStore,
                        TradingAgentsAdapter,
                        research_target_date,
                    )

                    now = datetime.now(timezone.utc)
                    service = DailyResearchService(
                        DailyResearchStore(_settings.daily_research_db),
                        TradingAgentsAdapter(),
                        notifier=None,
                        ensure_bars=True,
                    )
                    # AI 决策池 ∪ 自选（跟"启动引擎"/"全部重建"用的同一个规则）——
                    # 手动研究不该只覆盖自选，否则 AI 选出来的标的反而没有研究依据。
                    try:
                        from trader.selection_pools import (
                            decision_symbols as _decision_symbols,
                        )

                        _pool_symbols = list(_decision_symbols(limit=8))
                    except Exception:
                        _pool_symbols = []
                    _manual_symbols = [
                        symbol.strip().upper()
                        for symbol in str(_pref("sys_sym", "")).split(",")
                        if symbol.strip()
                    ]
                    research_symbols = list(
                        dict.fromkeys([*_pool_symbols, *_manual_symbols])
                    ) or ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]
                    result = service.run(
                        research_symbols,
                        trading_date=research_target_date(now),
                        timeframe=str(_pref("sys_tf", "5m")),
                        screen_limit=_settings.daily_research_screen_limit,
                        deep_limit=_settings.daily_research_deep_limit,
                        strategy_statistics_path=_settings.strategy_statistics_path,
                        force=True,
                        now=now,
                    )
                    research_status.set_text(
                        f"{result.status} · 完成 {result.completed_symbols} · "
                        f"失败 {result.failed_symbols}"
                        + (f" · {result.error_code}" if result.error_code else "")
                    )
                    research_live.set_content(
                        live_research_html(live_monitor_snapshot(_DAILY_RESEARCH_DB))
                    )
                    status = str(result.status).upper()
                    if status in {"FAILED", "ERROR"}:
                        return ActionJobResult(
                            state="ERROR",
                            user_message=f"研究失败: {result.error_code or status}",
                        )
                    if int(result.completed_symbols or 0) == 0:
                        return ActionJobResult(
                            state="EMPTY", user_message="研究未产生可用结果"
                        )
                    return result
                except Exception as exc:
                    research_status.set_text(f"研究失败: {exc}")
                    raise
                finally:
                    run_research_btn.props(remove="disable")

            return _research_work()

        run_research_btn.on_click(
            _audited_job_callback("overview.run_research", _run_daily_research_now)
        )

    with ui.element("div").classes("qa-kpi-row"):
        k_total = _kpi("总资产")
        k_pnl = _kpi("近 24h 盈亏")
        k_cash = _kpi("现金")
        k_unreal = _kpi("浮动盈亏")
        k_pos_n = _kpi("持仓数")
        k_pos_v = _kpi("持仓市值")

    maturity = latest_paper_maturity_status()
    if maturity is not None:
        _tone = "neg" if maturity["has_anomaly"] else "pos"
        _anomaly_note = "· 发现未解释异常" if maturity["has_anomaly"] else ""
        ui.html(
            f'<div class="qa-note" style="margin:-4px 0 8px">'
            f"生产成熟度 · REAL "
            f'<span class="{_tone}">{maturity["real_sessions"]}/{maturity["required_sessions"]}</span>'
            f" · Paper 环境，不代表实盘授权 {_anomaly_note}</div>"
        )

    # ── 权益曲线 ────────────────────────────────────────────────────────────
    _eq_span_key = "1D"
    _SPAN_OPTS = ["1D", "1W", "1M", "3M", "YTD", "All"]

    with ui.element("div").classes("qa-card"):
        # 标题行 + 跨度选择器
        with ui.element("div").style(
            "display:flex;justify-content:space-between;align-items:center;width:100%;margin-bottom:4px"
        ):
            ui.label("组合收益 vs 基准").classes("qa-card-title")

            span_sel = ui.select(
                _SPAN_OPTS,
                value="1D",
            ).props("dense outlined dark").style("width:90px")

            def _on_span():
                nonlocal _eq_span_key
                _eq_span_key = span_sel.value
                update()

            span_sel.on_value_change(_audited_callback("overview.span_change", _on_span))

        # 超额收益摘要（动态更新）
        _perf_bar = ui.html('<div style="height:28px"></div>')

        eq_plot = ui.plotly(_equity_fig(None, {}, "ov-eq")).classes("w-full")
        eq_empty = ui.element("div")

    with ui.element("div").classes("qa-card"):
        ui.label("当前持仓").classes("qa-card-title")
        ui.label("持仓来自 Alpaca 实时 API · 30 秒缓存").classes("qa-card-sub")
        pos_cols = [
            ("symbol", "标的", "left"),
            ("side", "方向", "center"),
            ("qty", "数量", "right"),
            ("avg_entry_price", "均价", "right"),
            ("current_price", "现价", "right"),
            ("market_value", "市值", "right"),
            ("unrealized_pl", "浮盈", "right"),
        ]
        pos_t = _make_table(pos_cols)
        pos_empty = ui.element("div")

    with ui.element("div").classes("qa-card"):
        ui.label("最近成交").classes("qa-card-title")
        ui.label("fills · 最近 24 小时").classes("qa-card-sub")
        fill_cols = [
            ("fill_time", "时间", "left"),
            ("symbol", "标的", "left"),
            ("side", "方向", "center"),
            ("filled_qty", "数量", "right"),
            ("avg_price", "均价", "right"),
        ]
        ft = _make_table(fill_cols)
        ft_empty = ui.element("div")

    with ui.element("div").classes("qa-card"):
        ui.label("最新新闻").classes("qa-card-title")
        ui.label(
            "WSCN 快讯 · SEC 8-K · Finnhub · 本地价格异动 · 不受盘中/盘前/盘后限制"
        ).classes("qa-card-sub")
        news_area = ui.column().style("gap:4px;width:100%;margin-top:6px")

    def _render_news():
        news_area.clear()
        news_df = recent_news(24, limit=20)
        with news_area:
            if news_df.empty:
                _empty("过去 24 小时暂无新闻", "📰")
                return
            for _, row in news_df.iterrows():
                symbol_html = (
                    f'<span style="color:var(--ai);font-family:var(--mono)">{_he(row["symbol"])}</span> '
                    if pd.notna(row.get("symbol"))
                    else ""
                )
                ui.html(
                    '<div class="qa-note" style="display:flex;gap:10px;align-items:baseline;flex-wrap:wrap;font-size:12px">'
                    f'<span style="color:var(--fg3);font-family:var(--mono)">{_fmt_time(row["ts"])}</span>'
                    f"{symbol_html}"
                    f'<b>{_he(row["title"])}</b>'
                    f'<span style="color:var(--fg3);font-size:11px">{_he(row["source"])}</span>'
                    "</div>"
                )

    def update():
        nonlocal _eq_span_key
        research_live.set_content(
            live_research_html(live_monitor_snapshot(_DAILY_RESEARCH_DB))
        )
        live = live_alpaca_equity()

        # ── 确定 DuckDB 查询窗口 ──────────────────────────────────────────
        _cfg_hours, _, _ = _SPAN_CFG.get(_eq_span_key, (24, "2d", "15m"))
        if _cfg_hours is None:  # YTD
            _fetch_h = _ytd_hours()
        elif _cfg_hours >= 9999:  # Max
            _fetch_h = 365 * 24 * 10
        else:
            _fetch_h = _cfg_hours
        eq = equity_df(_fetch_h)
        if not eq.empty and "ts" in eq.columns:
            # DuckDB 的 TIMESTAMPTZ 读出来带时区（如 America/Los_Angeles）；这个
            # 文件里图表相关的时间戳一律用"tz-naive but 美东（ET）"约定（参考
            # _fetch_benchmark 对 yfinance 数据的处理）——全仪表盘统一按美东
            # 显示，不用 UTC，这里统一转成 naive ET，否则下面和 _now_ts /
            # _align_bench 的裸时间戳比较会直接 TypeError。
            eq["ts"] = (
                pd.to_datetime(eq["ts"], utc=True)
                .dt.tz_convert("America/New_York")
                .dt.tz_localize(None)
            )
        has = not eq.empty and "total_equity" in eq.columns

        if live is not None:
            k_total.set_text(_money(live["equity"]))
            k_cash.set_text(_money(live["cash"]))
            if has:
                pnl = live["equity"] - float(eq["total_equity"].iloc[0])
                k_pnl.set_text(f"{pnl:+,.0f}")
                k_pnl.classes(remove="pos neg", add="pos" if pnl >= 0 else "neg")
            else:
                k_pnl.set_text("—")
            unreal = (
                float(eq["unrealized_pnl"].iloc[-1])
                if has and "unrealized_pnl" in eq.columns
                else None
            )
            k_unreal.set_text(f"{unreal:+,.0f}" if unreal is not None else "—")
            if unreal is not None:
                k_unreal.classes(remove="pos neg", add="pos" if unreal >= 0 else "neg")
        elif has:
            total = float(eq["total_equity"].iloc[-1])
            pnl = total - float(eq["total_equity"].iloc[0])
            cash = float(eq["cash"].iloc[-1]) if "cash" in eq.columns else None
            unreal = (
                float(eq["unrealized_pnl"].iloc[-1])
                if "unrealized_pnl" in eq.columns
                else None
            )
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

        # ── 把实时 Alpaca 权益追加为曲线终点（延伸到"现在"）────────────────
        # equity_snapshots 由引擎定期写入，最新点可能几分钟前；
        # live 有当前值，注入后曲线才会延伸到当前时刻。
        if live is not None:
            # eq["ts"] 已在上面归一化成 naive 美东（ET），这里保持同一约定。
            _now_ts = pd.Timestamp.now("America/New_York").tz_localize(None)
            _live_row = pd.DataFrame([{"ts": _now_ts, "total_equity": live["equity"]}])
            if has:
                _last_eq_ts = eq["ts"].iloc[-1]
                if _now_ts > _last_eq_ts:
                    eq = pd.concat(
                        [eq[["ts", "total_equity"]], _live_row], ignore_index=True
                    )
            else:
                eq = _live_row
                has = True

        # ── 基准指数（SPY / QQQ，5 min 缓存）────────────────────────────────
        _raw_bench: dict = {}
        for _bsym in _BENCHMARKS:
            _bh = _fetch_benchmark(_bsym, _eq_span_key)
            if _bh is not None:
                _raw_bench[_bsym] = _bh

        # 关键：对齐到组合数据的实际起始时间
        # YTD → 若账户 3 月开户，基准也从 3 月算，不从 Jan 1 算
        # All → 基准从 DuckDB 第一条记录时间开始，不从 SPY 历史起点算
        bench = _align_bench(eq, _raw_bench)

        # 超额收益摘要（与图表使用相同的对齐数据，数字一致）
        has_history = has and len(eq) >= 2
        if has_history:
            _eq_period_pct = _last_pct(eq["total_equity"])
            _perf_bar.set_content(_perf_chips_html(_eq_period_pct, bench))
        else:
            _perf_bar.set_content('<div style="height:28px"></div>')

        eq_plot.set_visibility(True)
        eq_plot.figure = _equity_fig(
            eq if has_history else None,
            bench if has_history else {},
            "ov-eq",
        )
        eq_plot.update()
        eq_empty.clear()
        # 当前持仓（Alpaca 实时 API）
        positions = live_alpaca_positions()
        k_pos_n.set_text(str(len(positions)) if positions else "0")
        total_mv = sum(p["market_value"] for p in positions) if positions else None
        k_pos_v.set_text(_money(total_mv) if total_mv else "—")
        pos_t.set_visibility(bool(positions))
        pos_empty.clear()
        if not positions:
            with pos_empty:
                _empty("暂无持仓", "💼")
        else:
            pos_df = pd.DataFrame(positions)
            _fill_table(
                pos_t,
                pos_df,
                pos_cols,
                fmts={
                    "side": lambda v: (
                        "▲ long" if str(v).lower() == "long" else "▼ short"
                    ),
                    "qty": lambda v: f"{float(v):,.0f}",
                    "avg_entry_price": lambda v: f"${float(v):,.2f}",
                    "current_price": lambda v: f"${float(v):,.2f}",
                    "market_value": lambda v: f"${float(v):,.0f}",
                    "unrealized_pl": lambda v: (
                        f"+${float(v):,.2f}"
                        if float(v) >= 0
                        else f"-${abs(float(v)):,.2f}"
                    ),
                },
            )

        # 最近成交
        fills = fills_df(24)
        ft.set_visibility(not fills.empty)
        ft_empty.clear()
        if fills.empty:
            with ft_empty:
                _empty("暂无成交记录", "🧾")
        else:
            _fill_table(
                ft,
                fills,
                fill_cols,
                fmts={
                    "fill_time": _fmt_time,
                    "filled_qty": lambda v: f"{float(v):,.0f}",
                    "avg_price": lambda v: f"${float(v):,.2f}",
                },
            )

        _render_news()

    update()
    return update


def _render_activity():
    _page_head(
        "交易与持仓",
        "AI 计划 · 信号 · 风控事件 · 订单 · 数据窗口 24 小时",
        badge="live",
    )

    # ── 策略信号 / 风控事件 / 订单 ──────────────────────────────────────────
    sig_cols = [
        ("signal_time", "时间", "left"),
        ("symbol", "标的", "left"),
        ("strategy", "策略", "left"),
        ("side", "方向", "center"),
        ("exec_price", "执行价", "right"),
    ]
    risk_cols = [
        ("ts", "时间", "left"),
        ("symbol", "标的", "left"),
        ("action", "动作", "center"),
        ("verdict", "裁决", "center"),
        ("reason", "原因", "left"),
    ]
    order_cols = [
        ("created_at", "时间", "left"),
        ("symbol", "标的", "left"),
        ("side", "方向", "center"),
        ("qty", "数量", "right"),
        ("status", "状态", "center"),
        ("plan_id", "计划 ID", "left"),
    ]

    with ui.element("div").classes("qa-card"):
        ui.label("策略信号").classes("qa-card-title")
        ui.label("signals").classes("qa-card-sub")
        sig_t = _make_table(sig_cols)
        sig_e = ui.element("div")
    with ui.element("div").classes("qa-card"):
        ui.label("风控事件").classes("qa-card-title")
        ui.label("plan_risk_events").classes("qa-card-sub")
        risk_t = _make_table(risk_cols)
        risk_e = ui.element("div")
    with ui.element("div").classes("qa-card"):
        ui.label("订单").classes("qa-card-title")
        ui.label("orders").classes("qa-card-sub")
        order_t = _make_table(order_cols)
        order_e = ui.element("div")
    with ui.element("div").classes("qa-card"):
        ui.label("最新订单完整解释").classes("qa-card-title")
        ui.label("来源 · 研究 · 计划版本 · 风险 · OrderIntent · 成交").classes(
            "qa-card-sub"
        )
        order_explanation = ui.html(
            render_order_explanation_html(
                {"status": "EMPTY", "plan_id": "", "missing": ["order"]}
            )
        )

    def _refresh_one(table, empty_box, df, cols, fmts, icon, msg):
        table.set_visibility(not df.empty)
        empty_box.clear()
        if df.empty:
            with empty_box:
                _empty(msg, icon)
        else:
            _fill_table(table, df, cols, fmts=fmts)

    def update():
        recent_orders = orders_df(24)
        _refresh_one(
            sig_t,
            sig_e,
            signals_df(24),
            sig_cols,
            {"signal_time": _fmt_time, "exec_price": lambda v: f"${float(v):,.2f}"},
            "📡",
            "暂无信号",
        )
        _refresh_one(
            risk_t,
            risk_e,
            plan_risk_events_df(24),
            risk_cols,
            {"ts": _fmt_time},
            "🛡️",
            "暂无风控事件",
        )
        _refresh_one(
            order_t,
            order_e,
            recent_orders,
            order_cols,
            {"created_at": _fmt_time, "qty": lambda v: f"{float(v):,.0f}"},
            "📋",
            "暂无订单",
        )
        if recent_orders.empty or "plan_id" not in recent_orders.columns:
            explanation = {
                "status": "EMPTY",
                "plan_id": "",
                "missing": ["order"],
            }
        else:
            plan_id = str(recent_orders.iloc[0].get("plan_id") or "")
            explanation = explain_order(DB_PATH, plan_id)
        order_explanation.set_content(render_order_explanation_html(explanation))

    update()
    return update


def _render_model_schedule_settings():
    from trader.ai.client import _DEFAULT_OLLAMA_URL, _ollama_list_models
    from trader.config import settings
    from trader.config_editor import EnvKeyNotWritableError, write_env_setting

    with ui.element("div").classes("qa-card"):
        ui.label("模型与调度参数").classes("qa-card-title")
        ui.label(
            "写入 .env，不涉及交易权限；重启引擎后生效"
        ).classes("qa-card-sub")

        def _save(key: str, value) -> None:
            if value is None:
                ui.notify(f"{key} 不能为空，未保存", type="warning")
                return
            try:
                write_env_setting(key, value)
                ui.notify(f"已保存 {key}，重启引擎生效", type="positive")
            except EnvKeyNotWritableError:
                ui.notify(f"{key} 不允许通过界面修改", type="negative")

        base_url = os.getenv("OLLAMA_BASE_URL", _DEFAULT_OLLAMA_URL)
        installed = [m["name"] for m in _ollama_list_models(base_url)]
        current_model = os.getenv("OLLAMA_MODEL", "")
        model_options = installed or ([current_model] if current_model else ["（本机 Ollama 不可达）"])
        if current_model and current_model not in model_options:
            model_options = [current_model] + model_options

        with ui.row().classes("items-end gap-3 flex-wrap"):
            model_sel = ui.select(
                model_options,
                value=current_model if current_model in model_options else model_options[0],
                label="本地模型 (OLLAMA_MODEL)",
            ).props("dark dense outlined").style("width:220px")
            model_sel.on_value_change(lambda e: _save("OLLAMA_MODEL", e.value))

            provider_sel = ui.select(
                ["ollama", "anthropic", "stub"],
                value=os.getenv("LLM_PROVIDER", "ollama"),
                label="LLM 提供方",
            ).props("dark dense outlined").style("width:140px")
            provider_sel.on_value_change(lambda e: _save("LLM_PROVIDER", e.value))

        _NUMERIC_ENV_FIELDS = [
            # 这两项不再是"触发时刻"——研究批次只在盘前 5:30-9:15 ET 这一个固定
            # 窗口跑（daily_research.py::in_daily_run_window，未暴露成配置项）。
            # close_hour/minute 现在只用来判断一个批次算今天还是明天的研究
            # （research_target_date 的收盘分界线），标签要如实反映这一点，
            # 不能再叫"触发时"——原来的收盘后触发窗口已经删掉了。
            ("研究收盘分界（美东，时）", "DAILY_RESEARCH_CLOSE_HOUR_ET", settings.daily_research_close_hour_et, 0, 23),
            ("研究收盘分界（美东，分）", "DAILY_RESEARCH_CLOSE_MINUTE_ET", settings.daily_research_close_minute_et, 0, 59),
            # AGENT_CYCLE_INTERVAL_MINUTES 这一项已移除：它是死配置，没有任何
            # 代码读取它，但界面上却标着"Runtime 轮询周期(分钟)"，让人以为在
            # 调引擎的 tick 频率。真正的轮询间隔是「系统」页的「间隔(秒)」
            # （-> trader.main --interval -> TradingConfig.poll_interval_secs）。
            ("AI 采信阈值", "MIN_AI_SCORE", settings.min_ai_score, 0, 100),
            ("AI 证据最大有效期(分钟)", "AI_SCORE_MAX_AGE_MINUTES", settings.ai_score_max_age_minutes, 1, None),
            ("AI 最少贡献者数", "AI_MIN_CONTRIBUTORS", settings.ai_min_contributors, 1, None),
            ("AI 最小权重覆盖率", "AI_MIN_WEIGHT_COVERAGE", settings.ai_min_weight_coverage, 0, 1),
            ("每日筛选数量", "DAILY_RESEARCH_SCREEN_LIMIT", settings.daily_research_screen_limit, 1, None),
            ("每日深度研究数量", "DAILY_RESEARCH_DEEP_LIMIT", settings.daily_research_deep_limit, 1, None),
            ("Universe 最大标的数", "UNIVERSE_MAX_SYMBOLS", settings.universe_max_symbols, 1, None),
            ("Universe 新鲜度上限(分钟)", "UNIVERSE_MAX_AGE_MINUTES", settings.universe_max_age_minutes, 1, None),
        ]
        with ui.row().classes("items-end gap-3 flex-wrap").style("margin-top:10px"):
            for label, key, value, lo, hi in _NUMERIC_ENV_FIELDS:
                field = ui.number(
                    label, value=value, min=lo, max=hi
                ).props("dark dense outlined").style("width:170px")
                field.on_value_change(lambda e, k=key: _save(k, e.value))


def _render_system():
    _page_head("系统运营 · 引擎与通知", "引擎控制、运行健康与外部通知", badge="live")

    _render_model_schedule_settings()

    with ui.element("div").classes("qa-card"):
        ui.label("引擎控制").classes("qa-card-title")
        ui.label(
            "启动 / 停止实时交易引擎 (trader.main · Runtime 管道)。"
            "策略留空 = 全部 24 个策略共识投票。标的框始终叠加在「机会中心」当前决策池"
            "之上——留空只用决策池，填了就是决策池 + 你手动加的标的（不会替换决策池）。"
        ).classes("qa-card-sub")
        with ui.row().classes("items-end gap-3 flex-wrap"):

            def _default_pool_symbols() -> str:
                try:
                    from trader.selection_pools import decision_symbols

                    return ", ".join(decision_symbols(limit=8))
                except Exception:
                    return ""

            sym_in = _persist(
                ui.input(
                    "手动追加标的（叠加到决策池，留空=不加）",
                    value=_pref("sys_sym", ""),
                    placeholder=_default_pool_symbols() or "QQQ",
                )
                .props("dark dense outlined")
                .style("width:200px"),
                "sys_sym",
            )
            strat_in = _persist(
                ui.input(
                    "策略（留空=全部24策略共识）",
                    value=_pref("sys_strat", ""),
                    placeholder="留空 = 全部策略共识",
                )
                .props("dark dense outlined")
                .style("width:220px"),
                "sys_strat",
            )
            tf_in = _persist(
                ui.select(
                    ["5m", "30m", "1h", "1d"],
                    value=_pref("sys_tf", "30m"),
                    label="周期",
                )
                .props("dark dense outlined")
                .style("width:90px"),
                "sys_tf",
            )
            int_in = _persist(
                ui.number("间隔(秒)", value=_pref("sys_int", 30))
                .props("dark dense outlined")
                .style("width:110px"),
                "sys_int",
            )

        # ── AI 自动交易 ────────────────────────────────────────────────────
        with ui.row().classes("items-center gap-4").style("margin-top:12px"):
            # Execution authority is intentionally session-scoped. A saved UI
            # preference must never silently re-enable order submission after
            # an app restart.
            auto_trade_cb = ui.checkbox(
                "AI 自动交易（虚拟盘）",
                value=False,
            ).props("dark color=warning")
            score_in = _persist(
                ui.number(
                    "AI 评分门槛",
                    value=_pref("sys_min_score", 65),
                    min=40,
                    max=95,
                    step=5,
                )
                .props("dark dense outlined")
                .style("width:130px"),
                "sys_min_score",
            )
        ui.html(
            '<div class="qa-note">每次打开平台都默认关闭，需本次会话手动勾选。'
            "不勾选 = DRY-RUN（只记日志，不下单）。自动实盘不受支持。</div>"
        )
        ui.html(
            '<div class="qa-note" style="margin-top:4px">总资产、现金、持仓全部以 Alpaca 账户为准；系统不会在本地覆盖账户权益。</div>'
        )

        with ui.row().classes("gap-3").style("margin-top:14px"):

            def _do_start():
                try:
                    from trader.selection_pools import decision_symbols

                    pool_symbols = list(decision_symbols(limit=8))
                except Exception:
                    pool_symbols = []
                manual_symbols = [
                    s.strip().upper() for s in (sym_in.value or "").split(",") if s.strip()
                ]
                # 手动标的永远叠加在决策池之上，不做替换——留空时行为和过去一致。
                resolved_list = list(dict.fromkeys([*pool_symbols, *manual_symbols]))
                resolved_symbols = ", ".join(resolved_list)
                ui.notify(
                    _start_engine(
                        resolved_symbols or "QQQ",
                        strat_in.value or "",
                        tf_in.value,
                        int_in.value,
                        auto_trade=bool(auto_trade_cb.value),
                        min_ai_score=int(score_in.value or 65),
                    )
                )

            def _do_stop():
                ui.notify(_stop_engine())

            ui.button(
                "▶ 启动引擎",
                on_click=_audited_callback("system.start", _do_start),
                color="positive",
            ).props("unelevated")
            ui.button(
                "■ 停止",
                on_click=_audited_callback("system.stop", _do_stop),
                color="negative",
            ).props("unelevated outline")

    # ── Discord 推送 ──────────────────────────────────────────────────────────
    with ui.element("div").classes("qa-card"):
        with ui.row().classes("items-center gap-3"):
            ui.label("Discord 推送").classes("qa-card-title")
            ui.html(
                '<span style="font-size:11px;color:var(--qa-text-muted)">晨报 · 复盘 · 信号通知</span>'
            )
        ui.html(
            '<div class="qa-note">引擎运行时会在下方设置的美东时间自动发送晨报/复盘。'
            "下方按钮可立即手动触发（用于测试 Discord 配置）。</div>"
        )
        with ui.row().classes("items-end gap-3 flex-wrap").style("margin-top:8px"):
            from trader.config import settings as _settings
            from trader.config_editor import (
                EnvKeyNotWritableError as _EnvKeyNotWritableError,
                write_env_setting as _write_env_setting,
            )

            def _save_schedule(key: str, value) -> None:
                if value is None:
                    ui.notify(f"{key} 不能为空，未保存", type="warning")
                    return
                try:
                    _write_env_setting(key, int(value))
                    ui.notify(f"已保存 {key}，重启引擎生效", type="positive")
                except _EnvKeyNotWritableError:
                    ui.notify(f"{key} 不允许通过界面修改", type="negative")

            brief_hour_in = (
                ui.number(
                    "晨报发送时(美东)",
                    value=_settings.morning_brief_hour_et,
                    min=0,
                    max=23,
                )
                .props("dark dense outlined")
                .style("width:150px")
            )
            brief_hour_in.on_value_change(
                lambda e: _save_schedule("MORNING_BRIEF_HOUR_ET", e.value)
            )
            review_hour_in = (
                ui.number(
                    "复盘发送时(美东)",
                    value=_settings.daily_review_hour_et,
                    min=0,
                    max=23,
                )
                .props("dark dense outlined")
                .style("width:150px")
            )
            review_hour_in.on_value_change(
                lambda e: _save_schedule("DAILY_REVIEW_HOUR_ET", e.value)
            )
        with ui.row().classes("gap-3 items-center flex-wrap").style("margin-top:10px"):
            _push_status = ui.html(
                '<span style="font-size:12px;color:var(--qa-text-muted)">就绪</span>'
            )
            # 「复盘方向」下拉已移除：让使用者在收盘后凭印象挑一个方向再打分，
            # 等于给一个事后编造的判断评分，甚至可以挑一个事后看对的方向让复盘
            # 好看。方向现在只从 BriefCallStore 读当天晨报的真实判断。

            def _system_symbols():
                syms_raw = sym_in.value or "QQQ,SPY,NVDA,AAPL,MSFT"
                return [s.strip().upper() for s in syms_raw.split(",") if s.strip()]

            def _do_send_brief():
                _push_status.set_content(
                    '<span style="color:#d29922">晨报发送中…</span>'
                )
                try:
                    from trader.morning_brief import send_morning_brief

                    ok = send_morning_brief(symbols=_system_symbols())
                    _push_status.set_content(_push_result_html("晨报", ok))
                    return ok
                except Exception as exc:
                    _push_status.set_content(
                        f'<span style="color:#f85149">晨报错误: {exc}</span>'
                    )
                    raise

            def _do_send_intraday():
                _push_status.set_content(
                    '<span style="color:#d29922">盘中跟踪发送中…</span>'
                )
                try:
                    from trader.manual_push import send_intraday_levels_push

                    ok = send_intraday_levels_push(_system_symbols())
                    _push_status.set_content(_push_result_html("盘中 OR/VWAP", ok))
                    return ok
                except Exception as exc:
                    _push_status.set_content(
                        f'<span style="color:#f85149">盘中跟踪错误: {exc}</span>'
                    )
                    raise

            def _do_send_review():
                _push_status.set_content(
                    '<span style="color:#d29922">复盘发送中…</span>'
                )
                try:
                    from trader.discord_report import build_daily_review_message
                    from trader.notify import make_notifier
                    from trader.monitor_data import fills_df

                    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                    pnl, cnt = 0.0, 0
                    try:
                        df = fills_df(24)
                        if not df.empty and "realized_pnl" in df.columns:
                            pnl = float(df["realized_pnl"].sum())
                            cnt = len(df)
                    except Exception:
                        pass

                    msg = build_daily_review_message(
                        today=today,
                        pnl=pnl,
                        trade_count=cnt,
                        symbols=_system_symbols(),
                    )
                    ok = make_notifier().send(msg)
                    _push_status.set_content(_push_result_html("复盘", ok))
                    return ok
                except Exception as exc:
                    _push_status.set_content(
                        f'<span style="color:#f85149">复盘错误: {exc}</span>'
                    )
                    raise

            def _do_send_direction_review():
                _push_status.set_content(
                    '<span style="color:#d29922">方向复盘发送中…</span>'
                )
                try:
                    from trader.manual_push import send_direction_review_push

                    ok = send_direction_review_push(_system_symbols())
                    _push_status.set_content(_push_result_html("方向复盘", ok))
                    return ok
                except Exception as exc:
                    _push_status.set_content(
                        f'<span style="color:#f85149">方向复盘错误: {exc}</span>'
                    )
                    raise

            def _manual_push(label: str, fn):
                """统一的手动补发外壳：所有报告类型共用同一套状态显示。"""

                def _run():
                    _push_status.set_content(
                        f'<span style="color:#d29922">{label}发送中…</span>'
                    )
                    try:
                        ok = fn()
                        _push_status.set_content(_push_result_html(label, ok))
                        return ok
                    except Exception as exc:
                        _push_status.set_content(
                            f'<span style="color:#f85149">{label}错误: {exc}</span>'
                        )
                        raise

                return _run

            def _do_send_open_confirmation():
                from trader.manual_push import send_open_confirmation_push

                return send_open_confirmation_push()

            def _do_send_close_report():
                from trader.manual_push import send_close_report_push

                return send_close_report_push()

            def _do_send_weekly():
                from trader.manual_push import send_weekly_report_push

                return send_weekly_report_push()

            # 每一种自动推送的报告都要能手动补发。补发的周期语义由
            # report_period 统一决定：当前周期已开始就发本周期至今（正文会标注
            # 尚未结束），否则发上一个完整周期——而不是拿"此刻"硬算，那样周日
            # 点一次复盘会得到一份标着当天日期的空报告。
            _DISCORD_REPORTS = {
                "晨报": _do_send_brief,
                "开盘确认": _manual_push("开盘确认", _do_send_open_confirmation),
                "盘中 OR/VWAP": _do_send_intraday,
                "收盘报告（复盘+研究）": _manual_push("收盘报告", _do_send_close_report),
                "每日复盘（仅复盘）": _do_send_review,
                "方向复盘": _do_send_direction_review,
                "周报（策略体检）": _manual_push("周报", _do_send_weekly),
            }
            report_type_sel = ui.select(
                list(_DISCORD_REPORTS),
                value="晨报",
                label="报告类型",
            ).props("dense outlined dark").style("width:210px")

            def _do_send_selected():
                return _DISCORD_REPORTS[str(report_type_sel.value)]()

            ui.button(
                "📨 发送",
                on_click=_audited_job_callback("discord.send", _do_send_selected),
            ).props("unelevated dense color=primary")

        with ui.row().classes("gap-3 items-end flex-wrap").style("margin-top:8px"):
            stock_sym_in = (
                ui.input("标的", placeholder="AAPL")
                .props("dark dense outlined")
                .style("width:120px")
            )

            def _do_send_stock_analysis():
                symbol = str(stock_sym_in.value or "").strip()
                if not symbol:
                    _push_status.set_content(
                        '<span style="color:#f85149">请先填写标的代码</span>'
                    )
                    return False
                _push_status.set_content(
                    '<span style="color:#d29922">个股分析发送中…</span>'
                )
                try:
                    from trader.manual_push import send_stock_analysis_push

                    ok = send_stock_analysis_push(symbol)
                    _push_status.set_content(_push_result_html("个股分析", ok))
                    return ok
                except Exception as exc:
                    _push_status.set_content(
                        f'<span style="color:#f85149">个股分析错误: {exc}</span>'
                    )
                    raise

            ui.button(
                "📤 发送个股分析",
                on_click=_audited_job_callback(
                    "discord.stock_analysis", _do_send_stock_analysis
                ),
            ).props("unelevated dense outline")
        ui.html(
            '<div class="qa-note" style="margin-top:6px">'
            "配置推送目标：在 .env 设置 <code>DISCORD_WEBHOOK_URL</code>"
            "（Webhook）或 <code>DISCORD_BOT_TOKEN + DISCORD_CHANNEL_ID</code>（Bot）</div>"
        )

    with ui.element("div").classes("qa-kpi-row"):
        k_eng = _kpi("引擎进程")
        k_hb = _kpi("最近心跳")
        _kpi("审计库", "trade.duckdb")  # 静态展示，不随 update() 刷新，无需持有引用
        _kpi("数据窗口", "24h")

    reconciliation_banner = ui.html("")

    def _log_html() -> str:
        content = _he(_tail_log(40))
        return (
            f'<div class="qa-code" id="qa-log-box" '
            f'style="max-height:380px;overflow-y:auto;white-space:pre-wrap;word-break:break-all">'
            f"{content}</div>"
        )

    _LOG_SCROLL_JS = (
        "var e=document.getElementById('qa-log-box');if(e)e.scrollTop=e.scrollHeight"
    )

    with ui.element("div").classes("qa-card"):
        ui.label("引擎日志 · 最后 40 行").classes("qa-card-title")
        log_html = ui.html(_log_html())

    def update():
        running = _engine_running()
        k_eng.set_text("运行中" if running else "已停止")
        k_eng.classes(remove="pos neg", add="pos" if running else "neg")
        hb = heartbeat()
        if hb is not None:
            secs = (datetime.now(timezone.utc) - hb).total_seconds()
            k_hb.set_text(
                f"{secs:.0f} 秒前" if secs < 120 else f"{secs / 60:.0f} 分钟前"
            )
        else:
            k_hb.set_text("—")

        report = latest_reconciliation()
        blocked = bool(report) and not bool(report.get("ok"))
        if blocked:
            details = []
            for key, label in (
                ("errors", "连接错误"),
                ("unexplained_orders", "未知订单"),
                ("unexplained_positions", "未知持仓"),
            ):
                raw = report.get(key)
                try:
                    values = (
                        json.loads(raw) if isinstance(raw, str) else list(raw or [])
                    )
                except Exception:
                    values = [str(raw)] if raw else []
                if values:
                    details.append(f"{label}: {', '.join(map(str, values[:5]))}")
            reason = "；".join(details) or "对账未完成"
            checked_at = _fmt_time(report.get("ts", ""))
            reconciliation_banner.set_content(
                '<div style="margin:10px 0;padding:12px 14px;border-radius:8px;'
                "background:rgba(248,81,73,.16);border:1px solid #f85149;"
                'color:#f85149;font-weight:600">🛑 启动对账失败，自动交易已阻断'
                f'<div style="margin-top:5px;font-size:12px;font-weight:400">'
                f"{_he(reason)} · 最近检查 {_he(checked_at)}</div></div>"
            )
            k_eng.set_text("对账阻断")
            k_eng.classes(remove="pos neg", add="neg")
        else:
            reconciliation_banner.set_content("")

        log_html.set_content(_log_html())
        try:
            import asyncio as _aio

            _aio.get_running_loop()  # 没有事件循环时（如模块刚加载、ui.run() 还没起服务器）
            ui.run_javascript(
                _LOG_SCROLL_JS
            )  # 直接跳过，否则会创建一个永远没人 await 的协程
        except Exception:
            pass

    update()
    return update


def _render_research():
    _page_head("研究验证", "因子分析 · 策略回测 · 数据本地优先")

    import asyncio

    from trader.data_cache import list_cached_names

    cached_names = list_cached_names()

    with ui.element("div").classes("qa-card"):
        ui.label("最近复盘与候选状态").classes("qa-card-title")
        ui.label(
            "episode_reviews · strategy_release_events · 只读，发布事件不会自动修改 Runtime 生产参数"
        ).classes("qa-card-sub")
        outcome_df = episode_review_outcome_counts(50)
        if outcome_df.empty:
            _empty("暂无复盘记录", "🧬")
        else:
            _OUTCOME_LABELS = {
                "SUCCESS": "成功",
                "RISK_REJECTED": "风控拒绝",
                "NO_FILL": "未成交",
                "DATA_FAILURE": "数据故障",
                "BROKER_FAILURE": "broker 故障",
            }
            with ui.element("div").classes("qa-kpi-row"):
                for _, row in outcome_df.iterrows():
                    outcome_key = str(row["outcome"])
                    _kpi(
                        _OUTCOME_LABELS.get(outcome_key, outcome_key),
                        str(int(row["count"])),
                        tone="neg" if outcome_key != "SUCCESS" else "pos",
                    )
        release = latest_strategy_release_event()
        if release is None:
            _empty("暂无策略发布/拒绝事件", "🧬")
        else:
            _event_type = str(release.get("event_type", "—"))
            ui.html(
                f'<div class="qa-note" style="margin-top:8px">'
                f"最新发布事件 · <b>{html.escape(str(release.get('strategy_name', '—')))}</b> "
                f"· {html.escape(_event_type)} "
                f"· {html.escape(str(release.get('from_version', '—')))} → "
                f"{html.escape(str(release.get('to_version', '—')))} "
                f"· {html.escape(str(release.get('created_at', '—')))}</div>"
            )

    with ui.element("div").classes("qa-card"):
        ui.label("研究档案").classes("qa-card-title")
        ui.label(
            "历史研究记录 · 完整多空辩论存档 · 从「今日总览」点标的名会自动跳转到这里"
        ).classes("qa-card-sub")

        _pending_symbol = str(_pending_archive_filter.pop("symbol", "") or "")
        with ui.row().classes("items-end gap-3"):
            archive_symbol_in = (
                ui.input("标的代码（留空=全部）", value=_pending_symbol)
                .props("dark dense outlined")
                .style("width:180px")
            )
            archive_days_sel = ui.select(
                {0: "全部", 7: "最近 7 天", 30: "最近 30 天", 90: "最近 90 天"},
                value=30,
                label="时间范围",
            ).props("dark dense outlined").style("width:140px")
            archive_refresh_btn = ui.button("查询").props("unelevated dense")

        archive_columns = [
            {"name": "trading_date", "label": "研究交易日", "field": "trading_date", "align": "left", "sortable": True},
            {"name": "symbol", "label": "标的", "field": "symbol", "align": "left", "sortable": True},
            {"name": "conclusion", "label": "AI 结论", "field": "conclusion", "align": "left"},
        ]
        with ui.row().classes("w-full gap-3").style("align-items:flex-start;margin-top:10px"):
            with ui.element("div").style("flex:1;min-width:420px"):
                archive_table = (
                    ui.table(columns=archive_columns, rows=[], row_key="row_id", pagination=15)
                    .props("flat dense bordered")
                    .classes("w-full")
                )
            with ui.element("div").style("width:420px;min-width:340px;flex:0 0 420px"):
                archive_detail = ui.html(
                    '<div class="qa-note">点左边表格里的一行，看完整多空辩论记录。</div>'
                )

        def _archive_row_id(rec: dict) -> str:
            return f"{rec.get('run_id', '')}:{rec.get('symbol', '')}"

        _archive_cache: dict[str, dict] = {}

        def _refresh_archive() -> None:
            days = int(archive_days_sel.value or 0) or None
            records = list_debate_records(
                _AI_DB,
                symbol=str(archive_symbol_in.value or ""),
                days=days,
                limit=300,
            )
            _archive_cache.clear()
            rows = []
            for rec in records:
                row_id = _archive_row_id(rec)
                _archive_cache[row_id] = rec
                score = rec.get("ai_score")
                conclusion = (
                    f"{rec.get('recommendation', '—')} · {float(score):.1f} 分"
                    if score is not None
                    else str(rec.get("recommendation") or "—")
                )
                rows.append(
                    {
                        "row_id": row_id,
                        "trading_date": rec.get("trading_date", "—"),
                        "symbol": rec.get("symbol", ""),
                        "conclusion": conclusion,
                    }
                )
            archive_table.rows = rows
            archive_table.update()
            if not rows:
                archive_detail.set_content(
                    '<div class="qa-note">没有符合条件的记录——换个标的或者放宽时间范围试试。</div>'
                )

        def _on_archive_row(e) -> None:
            args = getattr(e, "args", None)
            row = {}
            if isinstance(args, dict):
                row = args.get("row") or args
            elif isinstance(args, (list, tuple)):
                for part in args:
                    if isinstance(part, dict) and part.get("row_id"):
                        row = part
                        break
            rec = _archive_cache.get(str(row.get("row_id", "")))
            if rec is None:
                return
            debate = str(rec.get("debate") or "").strip()
            body = (
                html.escape(debate).replace("\n", "<br>")
                if debate
                else "（这条记录没有完整辩论正文，可能是较早版本产生的，只有简报）"
                + (f"<br><br>{html.escape(str(rec.get('thesis') or ''))}" if rec.get("thesis") else "")
            )
            score = rec.get("ai_score")
            conclusion = (
                f"{rec.get('recommendation', '—')} · {float(score):.1f} 分"
                if score is not None
                else str(rec.get("recommendation") or "—")
            )
            archive_detail.set_content(
                f'<div style="display:grid;gap:8px">'
                f'<div><b style="font-size:15px">{html.escape(str(rec.get("symbol", "")))}</b> '
                f'<span style="color:var(--fg3)">· {html.escape(str(rec.get("trading_date", "")))} '
                f'· {html.escape(conclusion)}</span></div>'
                f'<div style="line-height:1.6;font-size:12.5px">{body}</div>'
                f"</div>"
            )

        archive_table.on("rowClick", _on_archive_row)
        archive_refresh_btn.on_click(
            _audited_callback("research.archive_query", _refresh_archive)
        )
        _refresh_archive()

    with ui.row().classes("items-center gap-2").style("margin-bottom:10px"):
        ui.label("研究工具").classes("qa-card-title")
        research_mode_sel = ui.toggle(["因子分析", "回测"], value="因子分析").props("dense no-caps")
    fa_section = ui.column().classes("w-full").style("gap:0")
    bt_section = ui.column().classes("w-full").style("gap:0")
    bt_section.set_visibility(False)

    def _on_research_mode():
        fa_section.set_visibility(research_mode_sel.value == "因子分析")
        bt_section.set_visibility(research_mode_sel.value == "回测")

    research_mode_sel.on_value_change(_on_research_mode)

    with fa_section:
        # ════════════════════════════════════════════════════════════════════════
        # 一、因子分析
        # ════════════════════════════════════════════════════════════════════════
        ui.html(
            '<div style="font-size:15px;font-weight:700;margin-bottom:12px">🔬 因子分析</div>'
        )

        try:
            from trader.factors import FACTOR_REGISTRY
            from trader.data_cache import get_bars as _get_bars
        except Exception as exc:
            ui.html(
                f'<div style="color:var(--neg);font-size:13px">无法加载因子库: {exc}</div>'
            )
        else:
            # 因子库展示
            with ui.element("div").classes("qa-card"):
                ui.label("因子库").classes("qa-card-title")
                ui.label(
                    f"已注册 {len(FACTOR_REGISTRY)} 个因子，交易 Agent 可直接调用"
                ).classes("qa-card-sub")
                cats: dict = {}
                for name, f in FACTOR_REGISTRY.items():
                    cats.setdefault(f.meta.category, []).append((name, f))
                cat_colors = {
                    "momentum": "#58a6ff",
                    "trend": "#3fb950",
                    "volatility": "#d29922",
                    "volume": "#a5d6ff",
                }
                html_parts = []
                for cat, items in cats.items():
                    color = cat_colors.get(cat, "#8b949e")
                    for name, f in items:
                        html_parts.append(
                            f'<div style="display:inline-flex;align-items:center;gap:6px;'
                            f"background:var(--panel2);border:1px solid var(--border);"
                            f'border-radius:6px;padding:5px 10px;margin:3px">'
                            f'<span style="width:8px;height:8px;border-radius:50%;'
                            f'background:{color};flex-shrink:0"></span>'
                            f'<span style="font-weight:600;font-size:11px">{name}</span>'
                            f'<span style="color:var(--fg3);font-size:10px">'
                            f"{f.meta.description[:30]}</span></div>"
                        )
                ui.html(
                    '<div style="display:flex;flex-wrap:wrap;gap:2px">'
                    + "".join(html_parts)
                    + "</div>"
                )

            # 因子分析控制区
            fa_syms = sorted({name.rsplit("_", 1)[0] for name in cached_names}) or ["AAPL"]
            fa_tfs = sorted(
                {name.rsplit("_", 1)[1].replace(".parquet", "") for name in cached_names}
            ) or ["5m"]
            fa_factor_names = list(FACTOR_REGISTRY.keys())

            def _fa_valid(val, opts, default):
                return val if val in opts else default

            with ui.element("div").classes("qa-card"):
                ui.label("因子分析设置").classes("qa-card-title")
                ui.label("选择因子和标的，分析 IC 和分位数收益率").classes("qa-card-sub")
                with ui.row().classes("items-end gap-3 flex-wrap"):
                    fa_sym_sel = _persist(
                        ui.select(
                            fa_syms,
                            value=_fa_valid(
                                _pref("fa_sym", fa_syms[0]), fa_syms, fa_syms[0]
                            ),
                            label="标的",
                        )
                        .props("dark dense outlined")
                        .style("width:120px"),
                        "fa_sym",
                    )
                    fa_tf_sel = _persist(
                        ui.select(
                            fa_tfs,
                            value=_fa_valid(_pref("fa_tf", "5m"), fa_tfs, fa_tfs[0]),
                            label="周期",
                        )
                        .props("dark dense outlined")
                        .style("width:90px"),
                        "fa_tf",
                    )
                    fa_fac_sel = _persist(
                        ui.select(
                            fa_factor_names,
                            value=_fa_valid(
                                _pref("fa_fac", "RSI_14"),
                                fa_factor_names,
                                fa_factor_names[0],
                            ),
                            label="因子",
                        )
                        .props("dark dense outlined")
                        .style("width:160px"),
                        "fa_fac",
                    )
                    fa_fwd_sel = _persist(
                        ui.select(
                            {1: "1 bar", 5: "5 bar", 10: "10 bar", 20: "20 bar"},
                            value=_pref("fa_fwd", 5),
                            label="前瞻期",
                        )
                        .props("dark dense outlined")
                        .style("width:100px"),
                        "fa_fwd",
                    )
                    fa_nq_sel = _persist(
                        ui.select(
                            {3: "三分位", 5: "五分位", 10: "十分位"},
                            value=_pref("fa_nq", 5),
                            label="分位数",
                        )
                        .props("dark dense outlined")
                        .style("width:100px"),
                        "fa_nq",
                    )
                    fa_run_btn = ui.button("▶ 分析", color="primary").props("unelevated")

            fa_result = ui.column().style("gap:12px;width:100%")

            fa_busy = False

            async def _run_fa():
                nonlocal fa_busy
                if fa_busy:
                    return
                fa_busy = True
                fa_run_btn.disable()
                symbol = fa_sym_sel.value
                timeframe = fa_tf_sel.value
                factor = FACTOR_REGISTRY[fa_fac_sel.value]
                fwd = int(fa_fwd_sel.value) if fa_fwd_sel.value else 5
                nq = int(fa_nq_sel.value) if fa_nq_sel.value else 5
                fa_result.clear()
                with fa_result:
                    ui.label("⏳ 计算中...").style("color:var(--fg3)")

                def _calculate_fa():
                    from trader.backtest.factor_analysis import run_factor_analysis

                    df = _get_bars(symbol, timeframe)
                    if df is None or df.empty:
                        return None
                    return run_factor_analysis(
                        df,
                        factor,
                        symbol=symbol,
                        forward_period=fwd,
                        n_quantiles=nq,
                    )

                try:
                    result = await asyncio.to_thread(_calculate_fa)
                    if _state["tab"] != "research":
                        return
                    fa_result.clear()
                    if result is None:
                        with fa_result:
                            _empty(f"无 {symbol} {timeframe} 本地数据", "📭")
                        return
                    with fa_result:
                        ic_color = (
                            "pos"
                            if result.ic_mean > 0.03
                            else "neg"
                            if result.ic_mean < -0.03
                            else ""
                        )
                        with ui.element("div").classes("qa-kpi-row"):
                            _kpi("IC 均值", f"{result.ic_mean:.4f}", tone=ic_color)
                            _kpi("IC 标准差", f"{result.ic_std:.4f}")
                            _kpi(
                                "ICIR",
                                f"{result.icir:.3f}",
                                tone="pos" if abs(result.icir) > 0.5 else "",
                            )
                            _kpi("有效样本", f"{result.n_valid:,}")
                            _kpi("前瞻期", f"{result.forward_period} bar")
                        with ui.element("div").classes("qa-card"):
                            ui.label("滚动 IC (20 bar 窗口)").classes("qa-card-title")
                            ui.label("IC > 0.05 = 因子有正向预测力").classes("qa-card-sub")
                            ic = result.ic_series
                            fig = go.Figure()
                            fig.add_trace(
                                go.Bar(
                                    x=list(range(len(ic))),
                                    y=ic.values.tolist(),
                                    marker_color=[
                                        "#3fb950" if v >= 0 else "#f85149"
                                        for v in ic.values
                                    ],
                                )
                            )
                            fig.add_hline(
                                y=0.05, line=dict(color="#3fb950", width=1, dash="dot")
                            )
                            fig.add_hline(
                                y=-0.05, line=dict(color="#f85149", width=1, dash="dot")
                            )
                            fig.update_layout(
                                height=200,
                                margin=dict(l=8, r=8, t=8, b=8),
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                font=dict(color="#8b949e", size=11),
                                showlegend=False,
                                uirevision="fa-ic",
                                xaxis=dict(gridcolor="#21262d"),
                                yaxis=dict(gridcolor="#21262d"),
                            )
                            ui.plotly(fig).classes("w-full")
                        with ui.element("div").classes("qa-card"):
                            qr = result.quantile_returns
                            ui.label("分位数平均前瞻收益率 (%)").classes("qa-card-title")
                            ui.label(
                                f"Q1=因子最低组，Q{nq}=最高组；单调递增 = 因子正向有效"
                            ).classes("qa-card-sub")
                            fig2 = go.Figure()
                            fig2.add_trace(
                                go.Bar(
                                    x=list(qr.index),
                                    y=[round(v, 4) for v in qr.values],
                                    marker_color=[
                                        "#3fb950" if v >= 0 else "#f85149"
                                        for v in qr.values
                                    ],
                                    text=[f"{v:.3f}%" for v in qr.values],
                                    textposition="outside",
                                )
                            )
                            fig2.update_layout(
                                height=220,
                                margin=dict(l=8, r=8, t=30, b=8),
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                font=dict(color="#8b949e", size=11),
                                showlegend=False,
                                uirevision="fa-qr",
                                xaxis=dict(gridcolor="#21262d"),
                                yaxis=dict(gridcolor="#21262d"),
                            )
                            ui.plotly(fig2).classes("w-full")
                except Exception as exc:
                    if _state["tab"] == "research":
                        fa_result.clear()
                        with fa_result:
                            _empty(f"因子分析失败: {exc}", "⚠️")
                finally:
                    fa_busy = False
                    if _state["tab"] == "research":
                        fa_run_btn.enable()

            fa_run_btn.on_click(_audited_callback("factors.run", _run_fa))

    with bt_section:
        # ════════════════════════════════════════════════════════════════════════
        # 二、策略回测
        # ════════════════════════════════════════════════════════════════════════
        ui.html('<div style="border-top:1px solid var(--border);margin:20px 0 12px"></div>')
        ui.html(
            '<div style="font-size:15px;font-weight:700;margin-bottom:12px">📉 策略回测</div>'
        )

        try:
            from trader.data_cache import get_bars
            from trader.strategy_core import (
                DEFAULT_STRATEGY_PARAMS,
                STRATEGY_OPTIONS,
                STRATEGY_PARAM_SPECS,
                compute_signals,
            )
            from trader.engine import simulate
        except Exception as exc:
            _empty(f"无法加载策略引擎: {exc}", "⚠️")
            return None

        symbols = sorted({name.rsplit("_", 1)[0] for name in cached_names}) or ["QQQ"]
        tfs = sorted(
            {name.rsplit("_", 1)[1].replace(".parquet", "") for name in cached_names}
        ) or ["30m"]
        strategies = list(STRATEGY_OPTIONS)

        def _bt_valid(val, options, default):
            return val if val in options else default

        r_sym = _bt_valid(_pref("r_sym", symbols[0]), symbols, symbols[0])
        r_tf = _bt_valid(_pref("r_tf", tfs[0]), tfs, tfs[0])
        r_strat = _bt_valid(_pref("r_strat", "5/20均线金叉死叉"), strategies, strategies[0])

        with ui.element("div").classes("qa-card"):
            ui.label("回测设置").classes("qa-card-title")
            ui.label("数据严格本地优先 (bars/ Parquet)，不自动联网").classes("qa-card-sub")
            with ui.row().classes("items-end gap-3 flex-wrap"):
                bt_sym_sel = _persist(
                    ui.select(symbols, value=r_sym, label="标的")
                    .props("dark dense outlined")
                    .style("width:130px"),
                    "r_sym",
                )
                bt_tf_sel = _persist(
                    ui.select(tfs, value=r_tf, label="周期")
                    .props("dark dense outlined")
                    .style("width:95px"),
                    "r_tf",
                )
                bt_strat_sel = _persist(
                    ui.select(strategies, value=r_strat, label="策略")
                    .props("dark dense outlined")
                    .style("width:250px"),
                    "r_strat",
                )
                bt_cap_in = _persist(
                    ui.number("本金", value=_pref("r_cap", 10000), format="%.0f")
                    .props("dark dense outlined")
                    .style("width:110px"),
                    "r_cap",
                )
                bt_lev_in = _persist(
                    ui.number("杠杆", value=_pref("r_lev", 1.0), step=0.5, min=1.0)
                    .props("dark dense outlined")
                    .style("width:90px"),
                    "r_lev",
                )
                bt_slip_in = _persist(
                    ui.number(
                        "单边滑点(bps)", value=_pref("r_slip", 5.0), step=1.0, min=0.0
                    )
                    .props("dark dense outlined")
                    .style("width:125px"),
                    "r_slip",
                )
                bt_fill_sel = _persist(
                    ui.select(
                        {"next_open": "下一开盘", "close": "当根收盘"},
                        value=_pref("r_fill", "next_open"),
                        label="成交",
                    )
                    .props("dark dense outlined")
                    .style("width:120px"),
                    "r_fill",
                )
                bt_risk_sw = _persist(
                    ui.switch("风控熔断", value=_pref("r_risk", False)), "r_risk"
                )
                bt_run_btn = ui.button("▶ 运行回测", color="primary").props("unelevated")

            bt_param_row = ui.row().classes("items-end gap-3 flex-wrap").style(
                "margin-top:8px"
            )
            bt_param_widgets: dict[str, object] = {}

            def _rebuild_bt_params():
                bt_param_row.clear()
                bt_param_widgets.clear()
                spec = STRATEGY_PARAM_SPECS.get(bt_strat_sel.value, [])
                if not spec:
                    return
                with bt_param_row:
                    ui.label("策略参数（仅作用于本次回测）:").style(
                        "font-size:12px;color:var(--fg3)"
                    )
                    for label, key, default, kind in spec:
                        if kind == "bool":
                            widget = ui.switch(label, value=bool(default))
                        else:
                            widget = (
                                ui.number(label, value=default, step=kind)
                                .props("dark dense outlined")
                                .style("width:130px")
                            )
                        bt_param_widgets[key] = widget

            bt_strat_sel.on_value_change(_rebuild_bt_params)
            _rebuild_bt_params()

        bt_result = ui.column().style("gap:16px;width:100%")
        with bt_result:
            _empty("选择参数后点击“运行回测”", "▶")

        bt_busy = False

        async def _run():
            nonlocal bt_busy
            if bt_busy:
                return
            bt_busy = True
            bt_run_btn.disable()
            symbol = bt_sym_sel.value
            timeframe = bt_tf_sel.value
            strategy = bt_strat_sel.value
            capital = float(bt_cap_in.value or 10000)
            leverage = float(bt_lev_in.value or 1.0)
            slippage_bps = float(bt_slip_in.value or 0.0)
            fill = bt_fill_sel.value
            risk_halt = bool(bt_risk_sw.value)
            strategy_params = {
                **DEFAULT_STRATEGY_PARAMS,
                **{
                    k: w.value
                    for k, w in bt_param_widgets.items()
                    if w.value is not None
                },
            }
            bt_result.clear()
            with bt_result:
                ui.label("⏳ 正在回测...").style("color:var(--fg3)")

            def _calculate_backtest():
                df = get_bars(symbol, timeframe)
                if df is None or df.empty:
                    return None
                df_sig = compute_signals(df.copy(), strategy, **strategy_params)
                res = simulate(
                    df_sig,
                    capital=capital,
                    leverage=leverage,
                    slippage_bps=slippage_bps,
                    fill=fill,
                    risk_halt=risk_halt,
                )
                return df, df_sig, res

            try:
                computed = await asyncio.to_thread(_calculate_backtest)
                if _state["tab"] != "research":
                    return
                bt_result.clear()
                if computed is None:
                    with bt_result:
                        _empty(f"本地无 {symbol} {timeframe} 数据 — 请先下载", "📭")
                    return
                df, df_sig, res = computed
                with bt_result:
                    tr = res.total_return
                    with ui.element("div").classes("qa-kpi-row"):
                        _kpi("最终权益", _money(res.final_equity))
                        _kpi("总收益", f"{tr:+.2%}", tone=("pos" if tr >= 0 else "neg"))
                        _kpi("平仓次数", str(res.closed_trades))
                        _kpi("胜率", f"{res.win_rate:.1%}" if res.closed_trades else "—")
                        _kpi("数据根数", f"{len(df):,}")

                    with ui.element("div").classes("qa-card"):
                        ui.label("权益曲线").classes("qa-card-title")
                        ui.label(f"{strategy} · {symbol} {timeframe}").classes(
                            "qa-card-sub"
                        )
                        if res.equity_curve is not None and not res.equity_curve.empty:
                            fig = go.Figure()
                            fig.add_trace(
                                go.Scatter(
                                    x=res.equity_curve.index.strftime("%Y-%m-%d %H:%M"),
                                    y=res.equity_curve.values,
                                    mode="lines",
                                    line=dict(width=2, color="#58a6ff"),
                                    name="权益",
                                    fill="tozeroy",
                                    fillcolor="rgba(88,166,255,0.08)",
                                )
                            )
                            fig.add_hline(
                                y=res.initial_capital,
                                line=dict(width=1, dash="dot", color="#6e7681"),
                            )
                            fig.update_layout(
                                height=260,
                                margin=dict(l=8, r=8, t=8, b=8),
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                font=dict(color="#8b949e", size=11),
                                showlegend=False,
                                uirevision="rs-eq",
                                xaxis=dict(gridcolor="#21262d"),
                                yaxis=dict(gridcolor="#21262d"),
                            )
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
                        fig.add_trace(
                            go.Candlestick(
                                x=x,
                                open=d["open"],
                                high=d["high"],
                                low=d["low"],
                                close=d["close"],
                                name="OHLC",
                                increasing_line_color="#3fb950",
                                decreasing_line_color="#f85149",
                            )
                        )
                        buys = d.index[d["strat_signal"] == 1].tolist()
                        sells = d.index[d["strat_signal"] == -1].tolist()
                        if buys:
                            fig.add_trace(
                                go.Scatter(
                                    x=buys,
                                    y=d.loc[buys, "strat_exec_px"],
                                    mode="markers",
                                    name="买入",
                                    marker=dict(
                                        symbol="triangle-up", size=11, color="#3fb950"
                                    ),
                                )
                            )
                        if sells:
                            fig.add_trace(
                                go.Scatter(
                                    x=sells,
                                    y=d.loc[sells, "strat_exec_px"],
                                    mode="markers",
                                    name="卖出",
                                    marker=dict(
                                        symbol="triangle-down", size=11, color="#f85149"
                                    ),
                                )
                            )
                        fig.update_layout(
                            height=380,
                            margin=dict(l=8, r=8, t=8, b=8),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#8b949e", size=11),
                            uirevision="rs-kline",
                            legend=dict(orientation="h", yanchor="bottom", y=1.01),
                            xaxis=dict(
                                gridcolor="#21262d", rangeslider=dict(visible=False)
                            ),
                            yaxis=dict(gridcolor="#21262d"),
                        )
                        ui.plotly(fig).classes("w-full")

                    with ui.element("div").classes("qa-card"):
                        ui.label("深度研究 · Marimo").classes("qa-card-title")
                        ui.label(
                            "需要响应式多单元格探索时，用 Marimo notebook（同一引擎）"
                        ).classes("qa-card-sub")
                        ui.html(
                            '<div class="qa-code">.venv\\Scripts\\marimo.exe edit notebooks/research.py</div>'
                        )

            except Exception as exc:
                if _state["tab"] == "research":
                    bt_result.clear()
                    with bt_result:
                        _empty(f"回测失败: {exc}", "⚠️")
            finally:
                bt_busy = False
                if _state["tab"] == "research":
                    bt_run_btn.enable()

        bt_run_btn.on_click(_audited_callback("backtest.run", _run))
    return None


def _render_selection_pools():
    _page_head(
        "机会中心",
        "低频全市场扫描 · 热点候选 · 动态观察池（现阶段沿用长期池/决策池数据）",
    )

    try:
        from trader.selection_pools import (
            DECISION_STYLE_AGGRESSIVE,
            DECISION_STYLE_STANDARD,
            DAILY_DECISION,
            LONG_TERM,
            build_daily_decision_pool,
            build_long_term_pool,
            load_decision_pool_report,
            load_selection_pool,
            rebuild_selection_pipeline,
            save_selection_pool,
            save_selection_pools,
        )
        from trader.decision_trade_plans import (
            load_decision_trade_plan_report,
        )
        from trader.market_scan import load_market_scan_report, run_market_scan
    except Exception as exc:
        _empty(f"无法加载选股模块: {exc}", "⚠️")
        return None

    progress_state = {
        "running": False,
        "label": "就绪",
        "started_at": "",
        "finished_at": "",
        "error": "",
        "active_stage": "",
        "expected_stages": [],
        "stages": {},
    }
    import threading

    progress_lock = threading.RLock()

    with ui.element("div").classes("qa-card"):
        ui.label("选股控制").classes("qa-card-title")
        with ui.row().classes("items-end gap-3 flex-wrap"):
            source_in = _persist(
                ui.textarea(
                    "候选源（可空）",
                    value=_pref("sel_source", ""),
                    placeholder="AAPL,NVDA,MSFT；留空则读取真实市场来源，来源失败不会用离线名单兜底",
                )
                .props("dark dense outlined")
                .style("width:360px;min-height:72px"),
                "sel_source",
            )
            long_n = _persist(
                ui.number(
                    "长期池数量",
                    value=_pref("sel_long_n", 100),
                    min=20,
                    max=300,
                    step=5,
                )
                .props("dark dense outlined")
                .style("width:120px"),
                "sel_long_n",
            )
            daily_n = _persist(
                ui.number(
                    "决策池上限", value=_pref("sel_daily_n", 7), min=3, max=7, step=1
                )
                .props("dark dense outlined")
                .style("width:120px"),
                "sel_daily_n",
            )
            decision_style = _persist(
                ui.select(
                    ["标准", "小资金进攻"],
                    value=_pref("sel_decision_style", "标准"),
                    label="决策风格",
                )
                .props("dark dense outlined")
                .style("width:140px"),
                "sel_decision_style",
            )
        with ui.row().classes("items-end gap-3 flex-wrap").style("margin-top:12px"):
            scan_n = _persist(
                ui.number(
                    "扫描股票数",
                    value=_pref("sel_scan_n", 10000),
                    min=50,
                    max=15000,
                    step=100,
                )
                .props("dark dense outlined")
                .style("width:145px"),
                "sel_scan_n",
            )
            selected_n = _persist(
                ui.number(
                    "扫描保留数",
                    value=_pref("sel_scan_keep_n", 3000),
                    min=50,
                    max=10000,
                    step=100,
                )
                .props("dark dense outlined")
                .style("width:130px"),
                "sel_scan_keep_n",
            )
            broad_sw = _persist(
                ui.switch("包含普通股全集", value=_pref("sel_include_broad", True)),
                "sel_include_broad",
            )
            ui.label(
                "全市场扫描是重任务，需单独运行；全部重建只使用最近一次扫描结果生成长期关注池与 3-7 个决策池候选。小资金进攻会放宽候选、但仍保留风控提示。"
            ).style("font-size:12px;color:var(--fg3)")
        with ui.row().classes("items-center gap-3 flex-wrap").style("margin-top:12px"):
            build_all_btn = ui.button("全部重建：长期→决策", color="primary").props(
                "unelevated dense"
            )
            status_html = ui.html(
                '<span style="font-size:12px;color:var(--fg3)">就绪</span>'
            )
        with ui.expansion("分步执行（高级）", icon="tune").classes("w-full").style(
            "margin-top:8px"
        ):
            with ui.row().classes("items-center gap-3 flex-wrap").style(
                "margin-top:6px"
            ):
                scan_btn = ui.button("全市场扫描", color="primary").props(
                    "unelevated dense outline"
                )
                build_long_btn = ui.button(
                    "更新长期池：最近扫描→长期", color="secondary"
                ).props("unelevated dense outline")
                build_daily_btn = ui.button(
                    "更新决策池：长期→决策", color="secondary"
                ).props("unelevated dense outline")
        progress_html = ui.html(_selection_progress_html(progress_state)).style(
            "margin-top:10px"
        )

    scan_report = load_market_scan_report()
    with ui.element("div").classes("qa-card"):
        scan_html = ui.html(_market_scan_summary_html(scan_report))
        with ui.row().classes("items-end gap-3 flex-wrap").style("margin-top:12px"):
            scan_search = (
                ui.input(
                    "搜索标的", placeholder="AAPL / NVDA / momentum / missing_bars"
                )
                .props("dark dense outlined clearable")
                .style("width:230px")
            )
            scan_status = (
                ui.select(
                    ["全部", "入选", "没入选"],
                    value="入选",
                    label="状态",
                )
                .props("dark dense outlined")
                .style("width:120px")
            )
            scan_tag = (
                ui.select(
                    ["全部", "指数成分", "热度补充", "普通股全集", "手动输入"],
                    value="全部",
                    label="来源标签",
                )
                .props("dark dense outlined")
                .style("width:135px")
            )
            scan_limit = (
                ui.number(
                    "显示行数",
                    value=max(
                        100, int(getattr(scan_report, "selected_size", 0) or 250)
                    ),
                    min=25,
                    max=10000,
                    step=25,
                )
                .props("dark dense outlined")
                .style("width:110px")
            )
            reset_filter_btn = ui.button("清除筛选", color="secondary").props(
                "unelevated dense outline"
            )

        scan_columns = [
            {
                "name": "rank",
                "label": "#",
                "field": "rank",
                "align": "right",
                "sortable": True,
            },
            {
                "name": "symbol",
                "label": "Symbol",
                "field": "symbol",
                "align": "left",
                "sortable": True,
            },
            {
                "name": "status",
                "label": "状态",
                "field": "status",
                "align": "left",
                "sortable": True,
            },
            {
                "name": "setup",
                "label": "机会",
                "field": "setup",
                "align": "left",
                "sortable": True,
            },
            {
                "name": "score",
                "label": "总分",
                "field": "score",
                "align": "right",
                "sortable": True,
            },
            {
                "name": "liquidity",
                "label": "流动性",
                "field": "liquidity",
                "align": "right",
                "sortable": True,
            },
            {
                "name": "trend",
                "label": "趋势",
                "field": "trend",
                "align": "right",
                "sortable": True,
            },
            {
                "name": "risk",
                "label": "风险",
                "field": "risk",
                "align": "right",
                "sortable": True,
            },
            {
                "name": "price",
                "label": "价格",
                "field": "price",
                "align": "right",
                "sortable": True,
            },
            {
                "name": "adv20_m",
                "label": "ADV20($M)",
                "field": "adv20_m",
                "align": "right",
                "sortable": True,
            },
            {
                "name": "ret60",
                "label": "60日%",
                "field": "ret60",
                "align": "right",
                "sortable": True,
            },
            {
                "name": "vol20",
                "label": "波动%",
                "field": "vol20",
                "align": "right",
                "sortable": True,
            },
            {"name": "tags", "label": "来源", "field": "tags", "align": "left"},
            {
                "name": "reason",
                "label": "理由/淘汰",
                "field": "reason",
                "align": "left",
            },
        ]
        with (
            ui.row()
            .classes("w-full gap-3")
            .style("align-items:flex-start;margin-top:12px")
        ):
            with ui.element("div").style("flex:1;min-width:680px"):
                scan_table = (
                    ui.table(
                        columns=scan_columns,
                        rows=_market_scan_table_rows(
                            scan_report,
                            status="入选",
                            limit=max(
                                100,
                                int(getattr(scan_report, "selected_size", 0) or 250),
                            ),
                        ),
                        row_key="symbol",
                        pagination=25,
                    )
                    .props("flat dense bordered")
                    .classes("w-full")
                )
            with ui.element("div").style("width:360px;min-width:300px;flex:0 0 360px"):
                scan_detail = ui.html(
                    _market_scan_detail_html((scan_report.items or [None])[0])
                )

    pool_htmls = {}
    with ui.row().classes("w-full gap-4").style("align-items:flex-start"):
        for layer, title in (
            (LONG_TERM, "长期关注池"),
            (DAILY_DECISION, "决策池"),
        ):
            with ui.element("div").classes("qa-card").style("flex:1;min-width:280px"):
                pool_htmls[layer] = ui.html(
                    _selection_pool_html(load_selection_pool(layer), title)
                )

    with ui.element("div").classes("qa-card"):
        decision_report_html = ui.html(
            _decision_pool_report_html(load_decision_pool_report())
        )

    with ui.element("div").classes("qa-card"):
        trade_plan_html = ui.html(
            _decision_trade_plan_report_html(load_decision_trade_plan_report())
        )

    buttons = [build_all_btn, scan_btn, build_long_btn, build_daily_btn]

    def _limits() -> tuple[int, int]:
        return (
            max(1, int(long_n.value or 100)),
            max(3, min(7, int(daily_n.value or 7))),
        )

    def _scan_limits() -> tuple[int, int, int]:
        scan_limit_value = max(1, int(scan_n.value or 10000))
        return (
            scan_limit_value,
            scan_limit_value,
            max(1, int(selected_n.value or 3000)),
        )

    def _source_value() -> str:
        return str(source_in.value or "").strip()

    def _manual_symbols_value() -> list[str]:
        # 自选来自"系统运营"引擎控制卡片的"手动追加标的"框（_pref("sys_sym", ...)），
        # 两处共用同一份持久化值，重建决策池时也要把自选算进去，不能只在启动引擎
        # 那一刻才临时拼一次——否则"决策池"这些分析卡片永远看不到自选标的。
        raw = str(_pref("sys_sym", "") or "").strip()
        return [s.strip().upper() for s in raw.split(",") if s.strip()]

    def _decision_style_value() -> str:
        return (
            DECISION_STYLE_AGGRESSIVE
            if str(decision_style.value or "") == "小资金进攻"
            else DECISION_STYLE_STANDARD
        )

    def _refresh_scan_table() -> None:
        rows = _market_scan_table_rows(
            scan_report,
            search=str(scan_search.value or ""),
            status=str(scan_status.value or "入选"),
            tag=str(scan_tag.value or "全部"),
            limit=max(1, int(scan_limit.value or 250)),
        )
        scan_table.rows = rows
        scan_table.update()
        first_symbol = rows[0]["symbol"] if rows else ""
        item = _market_scan_find(scan_report, first_symbol) if first_symbol else None
        scan_detail.set_content(_market_scan_detail_html(item))

    def _refresh_scan() -> None:
        nonlocal scan_report
        scan_report = load_market_scan_report()
        scan_html.set_content(_market_scan_summary_html(scan_report))
        _refresh_scan_table()

    def _on_scan_row(e) -> None:
        row = {}
        args = getattr(e, "args", None)
        if isinstance(args, dict):
            row = args.get("row") or args
        elif isinstance(args, (list, tuple)):
            for part in args:
                if isinstance(part, dict) and part.get("symbol"):
                    row = part
                    break
        symbol = str(row.get("symbol", "") or "")
        scan_detail.set_content(
            _market_scan_detail_html(_market_scan_find(scan_report, symbol))
        )

    def _reset_scan_filters() -> None:
        scan_search.value = ""
        scan_status.value = "入选"
        scan_tag.value = "全部"
        scan_limit.value = max(
            100, int(getattr(scan_report, "selected_size", 0) or 250)
        )
        _refresh_scan_table()

    def _refresh_pools() -> None:
        titles = {
            LONG_TERM: "长期关注池",
            DAILY_DECISION: "决策池",
        }
        for layer, title in titles.items():
            pool_htmls[layer].set_content(
                _selection_pool_html(load_selection_pool(layer), title)
            )
        decision_report_html.set_content(
            _decision_pool_report_html(load_decision_pool_report())
        )
        trade_plan_html.set_content(
            _decision_trade_plan_report_html(load_decision_trade_plan_report())
        )

    def _busy(is_busy: bool, text: str = "") -> None:
        for btn in buttons:
            if is_busy:
                btn.props("disable")
            else:
                btn.props(remove="disable")
        if text:
            status_html.set_content(
                f'<span style="font-size:12px;color:var(--ai)">{_he(text)}</span>'
            )

    def _progress_snapshot() -> dict:
        with progress_lock:
            return {
                **progress_state,
                "expected_stages": list(progress_state.get("expected_stages", [])),
                "stages": {
                    key: dict(value)
                    for key, value in progress_state.get("stages", {}).items()
                },
            }

    def _progress_update(event: dict) -> None:
        stage = str(event.get("stage", "") or "job")
        current = int(event.get("current", 0) or 0)
        total = int(event.get("total", 0) or 0)
        with progress_lock:
            progress_state["active_stage"] = stage
            progress_state["stages"][stage] = {
                "current": current,
                "total": total,
                "message": str(event.get("message", "") or ""),
                "updated_at": str(event.get("updated_at", "") or ""),
            }

    def _start_progress(label: str, expected_stages: list[str] | None = None) -> None:
        with progress_lock:
            progress_state["running"] = True
            progress_state["label"] = label
            progress_state["started_at"] = _now_local_s()
            progress_state["finished_at"] = ""
            progress_state["error"] = ""
            progress_state["active_stage"] = ""
            progress_state["expected_stages"] = list(expected_stages or [])
            progress_state["stages"] = {}
        progress_html.set_content(_selection_progress_html(_progress_snapshot()))

    def _finish_progress(error: str = "") -> None:
        with progress_lock:
            progress_state["running"] = False
            progress_state["finished_at"] = _now_local_s()
            progress_state["error"] = error
        progress_html.set_content(_selection_progress_html(_progress_snapshot()))

    def _run_job(label: str, fn, expected_stages: list[str] | None = None) -> None:
        _busy(True, f"{label}中...")
        _start_progress(label, expected_stages)

        def _work():
            try:
                fn()
                _refresh_scan()
                _refresh_pools()
                status_html.set_content(
                    f'<span style="font-size:12px;color:var(--pos)">✓ {_he(label)}完成</span>'
                )
                ui.notify(f"{label}完成")
                _finish_progress()
            except Exception as exc:
                status_html.set_content(
                    f'<span style="font-size:12px;color:var(--neg)">✗ {_he(label)}失败: {_he(exc)}</span>'
                )
                ui.notify(f"{label}失败", type="negative")
                _finish_progress(str(exc))
                raise
            finally:
                _busy(False)

        return _work()

    def _run_scan() -> None:
        max_symbols, max_downloads, keep_n = _scan_limits()

        def _job():
            run_market_scan(
                source=_source_value() or None,
                refresh_universe=True,
                include_broad_market=bool(broad_sw.value),
                max_symbols=max_symbols,
                max_downloads=max_downloads,
                selected_limit=keep_n,
                save=True,
                progress_callback=_progress_update,
            )

        return _run_job("全市场扫描", _job, ["market_scan"])

    def _build_all() -> None:
        long_limit, daily_limit = _limits()
        if not _source_value() and not load_market_scan_report().items:
            ui.notify("请先运行全市场扫描，或手动输入候选源", type="warning")
            status_html.set_content(
                '<span style="font-size:12px;color:var(--warn)">需要先运行全市场扫描，或手动输入候选源</span>'
            )
            return False

        def _job():
            results = rebuild_selection_pipeline(
                _source_value() or None,
                long_limit=long_limit,
                daily_limit=daily_limit,
                decision_style=_decision_style_value(),
                ai_db_path=_AI_DB,
                save=False,
                download_missing_decision_etfs=True,
                manual_symbols=_manual_symbols_value(),
                progress_callback=_progress_update,
            )
            save_selection_pools(results)

        return _run_job("全部重建", _job, [LONG_TERM, DAILY_DECISION])

    def _build_long() -> None:
        long_limit, _daily_limit = _limits()
        if not _source_value() and not load_market_scan_report().items:
            ui.notify("请先运行全市场扫描，或手动输入候选源", type="warning")
            status_html.set_content(
                '<span style="font-size:12px;color:var(--warn)">需要先运行全市场扫描，或手动输入候选源</span>'
            )
            return False

        def _job():
            result = build_long_term_pool(
                _source_value() or None,
                limit=long_limit,
                ai_db_path=_AI_DB,
                progress_callback=_progress_update,
            )
            save_selection_pool(result)

        return _run_job("更新长期池", _job, [LONG_TERM])

    def _build_daily() -> None:
        _long_limit, daily_limit = _limits()

        def _job():
            result = build_daily_decision_pool(
                _source_value() or None,
                limit=daily_limit,
                decision_style=_decision_style_value(),
                ai_db_path=_AI_DB,
                download_missing_decision_etfs=True,
                manual_symbols=_manual_symbols_value(),
                progress_callback=_progress_update,
            )
            save_selection_pool(result)

        return _run_job("更新决策池", _job, [DAILY_DECISION])

    scan_search.on_value_change(lambda _e: _refresh_scan_table())
    scan_status.on_value_change(lambda _e: _refresh_scan_table())
    scan_tag.on_value_change(lambda _e: _refresh_scan_table())
    scan_limit.on_value_change(lambda _e: _refresh_scan_table())
    reset_filter_btn.on_click(
        _audited_callback("pool.clear_filter", _reset_scan_filters)
    )
    scan_table.on("rowClick", _on_scan_row)
    scan_btn.on_click(_audited_job_callback("pool.full_scan", _run_scan))
    build_all_btn.on_click(_audited_job_callback("pool.rebuild_all", _build_all))
    build_long_btn.on_click(_audited_job_callback("pool.long_term", _build_long))
    build_daily_btn.on_click(_audited_job_callback("pool.decision", _build_daily))
    ui.timer(
        1.0,
        lambda: progress_html.set_content(
            _selection_progress_html(_progress_snapshot())
        ),
    )
    _refresh_scan()
    _refresh_pools()
    return None


def _render_risk():
    _page_head(
        "系统运营 · 风控",
        "回撤熔断 · Kill Switch · 执行开关 · 风控事件",
        badge="live",
    )

    # ── 读取 config（一次，不在 update 里重复） ───────────────────────────────
    try:
        from trader.config import TradingConfig

        _cfg = TradingConfig()
        _r = _cfg.risk
        _dd_limit = _r.daily_drawdown_limit_pct  # e.g. 0.03
        _pos_pct = _r.max_position_pct  # e.g. 0.20
        _trade_risk = _r.max_trade_risk_pct  # e.g. 0.005
        _max_fail = _r.max_consecutive_failures  # e.g. 3
        _allow_sh = _r.allow_short  # False
        _exec_on = _cfg.auto_trade_paper
    except Exception:
        _dd_limit = 0.03
        _pos_pct = 0.20
        _trade_risk = 0.005
        _max_fail = 3
        _allow_sh = False
        _exec_on = False

    # ── KPI 行 ───────────────────────────────────────────────────────────────
    with ui.element("div").classes("qa-kpi-row"):
        k_status = _kpi("系统状态", "检查中…")
        k_dd = _kpi("日内回撤", "—")
        k_events = _kpi("风控事件 24h", "—")
        k_ks = _kpi("Kill Switch", "检查中…")

    # ── 双列：风控参数 + Kill Switch 控制 ────────────────────────────────────
    with ui.row().classes("w-full gap-4").style("align-items:flex-start"):
        # 左：风控参数（动态读 config）
        with ui.element("div").classes("qa-card").style("flex:1;min-width:260px"):
            ui.label("风控参数").classes("qa-card-title")
            _exec_color = "#f85149" if _exec_on else "#3fb950"
            _exec_txt = "⚠ 已开启（可下单）" if _exec_on else "✓ 已关闭（DRY-RUN）"
            ui.html(f"""
            <table style="width:100%;border-collapse:collapse;font-size:13px;margin-top:8px">
              <colgroup><col style="width:55%"><col style="width:45%"></colgroup>
              <tr style="border-bottom:1px solid var(--border)">
                <td style="padding:7px 4px;color:var(--fg2)">日内回撤熔断线</td>
                <td style="padding:7px 4px;text-align:right;font-family:var(--mono);color:#f85149;font-weight:600">
                  -{_dd_limit * 100:.2f}%
                </td>
              </tr>
              <tr style="border-bottom:1px solid var(--border)">
                <td style="padding:7px 4px;color:var(--fg2)">单标的最大仓位</td>
                <td style="padding:7px 4px;text-align:right;font-family:var(--mono)">
                  ≤ {_pos_pct * 100:.0f}%
                </td>
              </tr>
              <tr style="border-bottom:1px solid var(--border)">
                <td style="padding:7px 4px;color:var(--fg2)">单笔最大风险</td>
                <td style="padding:7px 4px;text-align:right;font-family:var(--mono)">
                  {_trade_risk * 100:.2f}% 资金
                </td>
              </tr>
              <tr style="border-bottom:1px solid var(--border)">
                <td style="padding:7px 4px;color:var(--fg2)">连续失败熔断</td>
                <td style="padding:7px 4px;text-align:right;font-family:var(--mono)">
                  {_max_fail} 次
                </td>
              </tr>
              <tr style="border-bottom:1px solid var(--border)">
                <td style="padding:7px 4px;color:var(--fg2)">做空</td>
                <td style="padding:7px 4px;text-align:right;color:{"#3fb950" if _allow_sh else "#f85149"}">
                  {"允许" if _allow_sh else "不允许"}
                </td>
              </tr>
              <tr>
                <td style="padding:7px 4px;color:var(--fg2)">AI 自动模拟盘</td>
                <td style="padding:7px 4px;text-align:right;color:{_exec_color};font-weight:600">
                  {_exec_txt}
                </td>
              </tr>
            </table>
            """)

        # 右：Kill Switch 控制
        with ui.element("div").classes("qa-card").style("flex:1;min-width:260px"):
            ui.label("Kill Switch 控制").classes("qa-card-title")
            ui.html(
                '<div class="qa-card-sub">手动急停 / 解除急停，立即影响引擎下单行为</div>'
            )

            ks_banner = ui.html("")
            dd_bar = ui.html("")

            def _do_engage():
                try:
                    from trader.watchdog import FileKillSwitch

                    FileKillSwitch().engage("UI 手动触发急停")
                    ui.notify(
                        "🛑 Kill Switch 已激活，引擎停止下单",
                        color="negative",
                        timeout=4000,
                    )
                    update()
                except Exception as exc:
                    ui.notify(f"操作失败: {exc}", color="warning")

            def _do_disengage():
                try:
                    from trader.watchdog import FileKillSwitch

                    FileKillSwitch().disengage()
                    ui.notify("✓ Kill Switch 已解除", color="positive", timeout=3000)
                    update()
                except Exception as exc:
                    ui.notify(f"操作失败: {exc}", color="warning")

            with ui.row().classes("gap-3").style("margin-top:14px"):
                ui.button(
                    "🛑 触发急停",
                    on_click=_audited_callback("risk.trigger", _do_engage),
                    color="negative",
                ).props("unelevated")
                ui.button(
                    "✓ 解除急停",
                    on_click=_audited_callback("risk.clear", _do_disengage),
                ).props("unelevated outline")

            ui.html(
                '<div style="font-size:11px;color:var(--fg3);margin-top:8px">'
                "急停后引擎不会自动恢复，需手动解除后重启。</div>"
            )

    # ── 风控事件记录（彩色 HTML 表格） ───────────────────────────────────────
    with ui.element("div").classes("qa-card"):
        with ui.row().classes("items-center gap-3").style("margin-bottom:10px"):
            ui.label("风控事件记录 (24h)").classes("qa-card-title").style("margin:0")
        events_area = ui.html("")

    # ── update ───────────────────────────────────────────────────────────────
    def update():
        # Kill Switch 状态
        _ks_on = False
        try:
            from trader.watchdog import FileKillSwitch

            _ks_on = FileKillSwitch().engaged()
            k_ks.set_text("⚠ 已激活" if _ks_on else "○ 正常")
            k_ks.classes(remove="pos neg", add="neg" if _ks_on else "pos")
            if _ks_on:
                ks_banner.set_content(
                    '<div style="margin-top:12px;padding:10px 14px;border-radius:8px;'
                    "background:rgba(248,81,73,.15);border:1px solid #f85149;"
                    'font-size:13px;color:#f85149;font-weight:600">'
                    "🛑 急停已激活 — 引擎停止所有下单操作</div>"
                )
            else:
                ks_banner.set_content(
                    '<div style="margin-top:12px;padding:10px 14px;border-radius:8px;'
                    "background:rgba(63,185,80,.1);border:1px solid rgba(63,185,80,.4);"
                    'font-size:13px;color:#3fb950">'
                    "✓ 正常运行中</div>"
                )
        except Exception:
            k_ks.set_text("—")

        # 回撤
        _dd_val = None
        try:
            eq = equity_df(24)
            if not eq.empty and "total_equity" in eq.columns:
                vals = eq["total_equity"].values
                if len(vals) > 1:
                    start = float(vals[0])
                    cur = float(vals[-1])
                    if start > 0:
                        _dd_val = (cur - start) / start * 100
                        k_dd.set_text(f"{_dd_val:+.2f}%")
                        k_dd.classes(
                            remove="pos neg", add="neg" if _dd_val < 0 else "pos"
                        )

                        # 进度条
                        bar_pct = min(abs(_dd_val) / (_dd_limit * 100) * 100, 100)
                        bar_color = (
                            "#f85149"
                            if _dd_val <= -_dd_limit * 100 * 0.7
                            else "#d29922"
                            if _dd_val < 0
                            else "#3fb950"
                        )
                        dd_bar.set_content(f"""
                        <div style="margin-top:18px">
                          <div style="display:flex;justify-content:space-between;
                               font-size:12px;color:var(--fg3);margin-bottom:5px">
                            <span>日内回撤进度</span>
                            <span style="font-family:var(--mono)">{_dd_val:+.2f}% / -{_dd_limit * 100:.2f}%</span>
                          </div>
                          <div style="background:var(--border);border-radius:4px;height:8px;overflow:hidden">
                            <div style="width:{bar_pct:.1f}%;background:{bar_color};
                                 height:100%;transition:width .4s;border-radius:4px"></div>
                          </div>
                        </div>""")
        except Exception:
            pass

        # 风控事件（拒单）+ 整体状态 —— 只统计 verdict=BLOCKED 的裁决，
        # 通过的评估不算"事件"
        try:
            re = plan_risk_events_df(24, blocked_only=True)
            cnt = 0 if re.empty else len(re)
            k_events.set_text(str(cnt))

            if _ks_on:
                k_status.set_text("🛑 急停")
                k_status.classes(remove="pos neg", add="neg")
            elif cnt > 0:
                k_status.set_text("⚠ 有事件")
                k_status.classes(remove="pos neg", add="neg")
            else:
                k_status.set_text("✓ 正常")
                k_status.classes(remove="pos neg", add="pos")

            if re.empty or cnt == 0:
                events_area.set_content(
                    '<div style="font-size:13px;color:var(--fg3);padding:8px 0">'
                    "✓ 过去 24h 无风控事件</div>"
                )
            else:
                rows_html = ""
                for _, row in re.iterrows():
                    ts = _fmt_time(row.get("ts", ""))
                    sym = str(row.get("symbol", ""))
                    act = f"{row.get('side', '')} {row.get('action', '')}".strip()
                    reason = str(row.get("reason", ""))[:80]
                    rows_html += (
                        '<tr style="background:rgba(248,81,73,.12);color:#f85149;'
                        'border-bottom:1px solid var(--border)">'
                        f'<td style="padding:6px 8px;font-family:var(--mono);font-size:11px">{ts}</td>'
                        f'<td style="padding:6px 8px;font-weight:600;font-size:11px">{sym}</td>'
                        f'<td style="padding:6px 8px;font-size:12px">{act}</td>'
                        f'<td style="padding:6px 8px;font-size:12px">{reason}</td>'
                        f"</tr>"
                    )
                events_area.set_content(f"""
                <table style="width:100%;border-collapse:collapse">
                  <thead>
                    <tr style="color:var(--fg3);font-size:11px;text-transform:uppercase;
                               letter-spacing:.04em;border-bottom:1px solid var(--border)">
                      <th style="text-align:left;padding:5px 8px">时间</th>
                      <th style="text-align:left;padding:5px 8px">标的</th>
                      <th style="text-align:left;padding:5px 8px">动作</th>
                      <th style="text-align:left;padding:5px 8px">原因</th>
                    </tr>
                  </thead>
                  <tbody>{rows_html}</tbody>
                </table>""")
        except Exception:
            k_events.set_text("—")

    update()
    return update


# ═══════════════════════════════════════════════════════════════════════════
# 维护
# ═══════════════════════════════════════════════════════════════════════════


def _render_maintenance():
    _page_head(
        "系统运营 · 诊断",
        "绩效复盘 · 异常检测 · 整改建议 · Discord 日报",
    )

    ui.html(
        '<div class="qa-note">读取 DuckDB 审计数据，生成绩效报告和参数校准建议；建议仅供参考，不会自动修改策略。'
        "发送日报需 Discord Webhook 配置正确。</div>"
    )

    with ui.row().classes("items-center gap-3"):
        period_sel = (
            ui.select(
                {24: "过去 24h", 48: "过去 48h", 168: "过去 7 天"},
                value=24,
                label="分析窗口",
            )
            .props("dark dense outlined")
            .style("width:130px")
        )
        discord_sw = ui.switch("发送 Discord", value=False)
        run_btn_m = ui.button("▶ 运行维护分析", color="primary").props("unelevated")

    result_area_m = ui.column().style("gap:12px;width:100%")

    def _do_maintenance():
        result_area_m.clear()
        with result_area_m:
            ui.label("⏳ 分析中...").style("color:var(--fg3)")

        def _work():
            try:
                from trader.teams.maintenance import run_maintenance

                out = run_maintenance(
                    period_hours=int(period_sel.value),
                    send_discord=bool(discord_sw.value),
                )
                stats = out.data.get("stats", {})
                anomalies = out.data.get("anomalies", [])
                suggestions = out.data.get("suggestions", [])

                result_area_m.clear()
                with result_area_m:
                    pnl = stats.get("total_pnl", 0)
                    dd = stats.get("max_drawdown_pct", 0)
                    with ui.element("div").classes("qa-kpi-row"):
                        _kpi("成交次数", str(stats.get("trade_count", "—")))
                        _kpi(
                            "区间盈亏",
                            f"${pnl:+,.2f}" if pnl else "—",
                            tone="pos" if pnl > 0 else "neg",
                        )
                        _kpi("最大回撤", f"{dd:.2f}%", tone="neg" if dd < -1 else "")
                        _kpi(
                            "异常数",
                            str(len(anomalies)),
                            tone="neg" if anomalies else "pos",
                        )
                        _kpi("整改建议", str(len(suggestions)))

                    with ui.element("div").classes("qa-card"):
                        ui.label("异常检测结果").classes("qa-card-title")
                        if anomalies:
                            sev_color = {
                                "critical": "#f85149",
                                "high": "#d29922",
                                "medium": "#58a6ff",
                            }
                            for a in anomalies:
                                color = sev_color.get(a.get("severity", ""), "#8b949e")
                                ui.html(
                                    f'<div style="padding:8px 10px;border-left:3px solid {color};'
                                    f"background:var(--panel2);border-radius:0 6px 6px 0;"
                                    f'margin-bottom:6px;font-size:12px">'
                                    f'<span style="color:{color};font-weight:700;font-size:10px">'
                                    f"[{a.get('severity', '').upper()}]</span> {a.get('message', '')}"
                                    f"</div>"
                                )
                        else:
                            ui.html(
                                '<div style="color:var(--pos);padding:8px">✓ 未检测到异常</div>'
                            )

                    with ui.element("div").classes("qa-card"):
                        ui.label("参数校准建议").classes("qa-card-title")
                        ui.label("以下内容是复盘建议，不会自动修改策略参数").classes(
                            "qa-card-sub"
                        )
                        if suggestions:
                            pri_color = {
                                "critical": "#f85149",
                                "high": "#d29922",
                                "medium": "#58a6ff",
                            }
                            for s in suggestions:
                                color = pri_color.get(s.get("priority", ""), "#8b949e")
                                ui.html(
                                    f'<div style="padding:10px;border:1px solid {color}33;'
                                    f"border-left:3px solid {color};border-radius:0 6px 6px 0;"
                                    f'background:var(--panel2);margin-bottom:6px;font-size:12px">'
                                    f'<div style="color:{color};font-size:10px;font-weight:700">'
                                    f"[{s.get('priority', '').upper()}] → {s.get('target_team', '')} "
                                    f"· {s.get('target_param', '')}</div>"
                                    f'<div style="margin-top:4px">{s.get("suggestion", "")}</div>'
                                    f"</div>"
                                )
                        else:
                            ui.html(
                                '<div style="color:var(--fg3);padding:8px">暂无整改建议</div>'
                            )

                    if out.errors:
                        with ui.element("div").classes("qa-card"):
                            ui.label("分析错误").classes("qa-card-title")
                            for e in out.errors:
                                ui.html(
                                    f'<div style="color:var(--neg);font-size:12px">{e}</div>'
                                )

                if out.errors:
                    return ActionJobResult(
                        state="ERROR", user_message="维护分析存在错误"
                    )
            except Exception as exc:
                result_area_m.clear()
                with result_area_m:
                    _empty(f"维护分析失败: {exc}", "⚠️")

                raise

        return _work()

    run_btn_m.on_click(_audited_job_callback("maintenance.run", _do_maintenance))
    return None


def _he(s) -> str:
    """HTML 转义。"""
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _rtags(items: list, cls: str = "") -> str:
    c = f"cp-report-rtag {cls}".strip()
    return "".join(f'<span class="{c}">{_he(i)}</span>' for i in items[:4])


def _now_local_s() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _selection_progress_html(state: dict) -> str:
    stages = dict((state or {}).get("stages", {}) or {})
    running = bool((state or {}).get("running"))
    label = str((state or {}).get("label", "就绪") or "就绪")
    started = str((state or {}).get("started_at", "") or "")
    finished = str((state or {}).get("finished_at", "") or "")
    error = str((state or {}).get("error", "") or "")
    active_stage = str((state or {}).get("active_stage", "") or "")
    expected_stages = list((state or {}).get("expected_stages", []) or [])
    stage_titles = {
        "market_scan": "全市场扫描",
        "long_term": "长期关注池",
        "weekly_focus": "周级重点池",
        "daily_decision": "决策池",
    }
    status_text = "运行中" if running else ("失败" if error else "就绪")
    status_color = "var(--ai)" if running else ("var(--neg)" if error else "var(--fg3)")
    stage_keys = expected_stages or list(stages.keys())
    active = stages.get(active_stage, {}) if active_stage else {}
    current = int(active.get("current", 0) or 0)
    total = int(active.get("total", 0) or 0)
    active_pct = current / total if total else 0.0
    if running and stage_keys and active_stage in stage_keys:
        stage_idx = stage_keys.index(active_stage)
        pct = int(max(0.0, min(1.0, (stage_idx + active_pct) / len(stage_keys))) * 100)
    elif running and total:
        pct = int(max(0.0, min(1.0, active_pct)) * 100)
    elif finished and not error:
        pct = 100
    else:
        pct = 0
    message = str(
        active.get("message", "") or ("等待任务开始" if running else "暂无运行任务")
    )
    stage_text = stage_titles.get(active_stage, active_stage) if active_stage else ""
    count_text = f"{current}/{total}" if total else "—"
    time_text = (
        f"开始 {started}"
        if running and started
        else (f"完成 {finished}" if finished else "")
    )
    err_html = (
        f'<div style="font-size:11px;color:var(--neg);margin-top:4px">错误：{_he(error)}</div>'
        if error
        else ""
    )
    return (
        '<div style="width:100%;max-width:760px">'
        '<div style="display:flex;align-items:center;justify-content:space-between;gap:12px;font-size:11px">'
        f'<span style="color:{status_color}">{_he(label)} · {_he(status_text)}'
        f"{(' · ' + _he(time_text)) if time_text else ''}</span>"
        f'<span style="font-family:var(--mono);color:var(--fg2)">{pct}%</span>'
        "</div>"
        '<div style="height:7px;background:rgba(255,255,255,.06);border-radius:99px;overflow:hidden;margin-top:5px">'
        f'<div style="height:7px;width:{pct}%;background:{"var(--neg)" if error else ("var(--pos)" if pct >= 100 else "var(--ai)")};transition:width .2s ease"></div>'
        "</div>"
        '<div style="display:flex;justify-content:space-between;gap:12px;margin-top:4px;font-size:10.5px;color:var(--fg3)">'
        f"<span>{_he(stage_text + (' · ' if stage_text else '') + message)}</span>"
        f'<span style="font-family:var(--mono)">{_he(count_text)}</span>'
        "</div>"
        f"{err_html}"
        "</div>"
    )


_SCAN_TAG_LABEL = {
    "core_index": "指数成分",
    "hot": "热度补充",
    "broad_market": "普通股全集",
    "manual": "手动输入",
}
_SCAN_TAG_VALUE = {v: k for k, v in _SCAN_TAG_LABEL.items()}


def _market_scan_summary_html(report) -> str:
    items = list(getattr(report, "items", []) or [])
    updated = str(getattr(report, "updated_at", "") or "")
    universe_size = int(getattr(report, "universe_size", 0) or 0)
    scanned_size = int(getattr(report, "scanned_size", 0) or 0)
    selected_size = int(getattr(report, "selected_size", 0) or 0)
    rejected_size = int(getattr(report, "rejected_size", 0) or 0)
    source_status = list(getattr(report, "source_status", []) or [])
    reject_summary = dict(getattr(report, "reject_summary", {}) or {})

    if not updated and not items:
        return (
            '<div style="display:flex;align-items:center;justify-content:space-between;gap:10px">'
            '<div><div class="qa-card-title">全市场扫描</div>'
            '<div class="qa-card-sub">尚未生成。先点击“全市场扫描”，再用“全部重建”生成三层池。</div></div></div>'
            '<div style="color:var(--fg3);font-size:12px;padding:14px 0">'
            "扫描会优先覆盖 Dow / S&P 500 / Nasdaq-100 成分股，并补充散户热股与普通股全集。</div>"
        )

    ok_count = sum(1 for row in source_status if getattr(row, "ok", False))
    fail_count = sum(1 for row in source_status if not getattr(row, "ok", False))
    source_rows = []
    for row in source_status[:12]:
        ok = bool(getattr(row, "ok", False))
        color = "var(--pos)" if ok else "var(--warn)"
        label = "OK" if ok else "不可用"
        msg = str(getattr(row, "message", "") or "")
        count = int(getattr(row, "count", 0) or 0)
        source_rows.append(
            f'<span class="cp-report-rtag" style="color:{color};background:rgba(255,255,255,.04)">'
            f"{_he(getattr(row, 'source', ''))}: {_he(label)}"
            f"{f' / {count}' if count else ''}"
            f"{f' · {_he(msg[:50])}' if msg else ''}</span>"
        )
    source_html = (
        "".join(source_rows)
        or '<span style="font-size:12px;color:var(--fg3)">无来源状态</span>'
    )

    reject_html = (
        "".join(
            f'<span class="cp-report-rtag warn">{_he(reason)} {int(count)}</span>'
            for reason, count in list(reject_summary.items())[:8]
        )
        or '<span style="font-size:12px;color:var(--fg3)">暂无淘汰统计</span>'
    )

    return (
        '<div style="display:flex;align-items:flex-start;justify-content:space-between;gap:12px;flex-wrap:wrap">'
        '<div><div class="qa-card-title">全市场扫描</div>'
        f'<div class="qa-card-sub">更新时间：{_he(updated or "未知")} · 来源 {ok_count} OK / {fail_count} 不可用</div></div>'
        f'<div style="font-size:12px;color:var(--fg2)">Universe {universe_size} · 已评分 {scanned_size} · '
        f"保留 {selected_size} · 淘汰 {rejected_size} · 表内 {len(items)}</div>"
        "</div>"
        f'<div style="margin-top:10px;display:flex;gap:6px;flex-wrap:wrap">{source_html}</div>'
        f'<div style="margin-top:8px;display:flex;gap:6px;flex-wrap:wrap">{reject_html}</div>'
    )


def _market_scan_table_rows(
    report,
    *,
    search: str = "",
    status: str = "全部",
    tag: str = "全部",
    limit: int = 250,
) -> list[dict]:
    rows = []
    needle = str(search or "").strip().upper()
    tag_value = _SCAN_TAG_VALUE.get(str(tag or ""), "")
    for item in list(getattr(report, "items", []) or []):
        symbol = str(getattr(item, "symbol", "") or "")
        item_status = str(getattr(item, "status", "") or "")
        selected = item_status != "REJECT"
        display_status = _market_scan_status_label(item_status)
        tags = list(getattr(item, "tags", []) or [])
        reasons = list(getattr(item, "reasons", []) or [])
        rejects = list(getattr(item, "reject_reasons", []) or [])
        metrics = getattr(item, "metrics", {}) or {}
        parts = getattr(item, "component_scores", {}) or {}
        haystack = " ".join(
            [symbol, item_status, display_status, *tags, *reasons, *rejects]
        ).upper()
        if needle and needle not in haystack:
            continue
        if status == "入选" and not selected:
            continue
        if status == "没入选" and selected:
            continue
        if tag_value and tag_value not in tags:
            continue
        rows.append(
            {
                "rank": int(getattr(item, "rank", 0) or 0),
                "symbol": symbol,
                "status": display_status,
                "setup": _setup_label(getattr(item, "setup", "")),
                "score": _fmt_num(getattr(item, "score", None), 1),
                "liquidity": _fmt_num(parts.get("liquidity"), 0),
                "trend": _fmt_num(parts.get("trend"), 0),
                "risk": _fmt_num(parts.get("risk"), 0),
                "price": _fmt_num(metrics.get("price"), 2),
                "adv20_m": _fmt_num((metrics.get("adv20") or 0) / 1_000_000, 0)
                if metrics.get("adv20")
                else "—",
                "ret60": _fmt_num(metrics.get("ret60_pct"), 1),
                "vol20": _fmt_num(metrics.get("vol20_pct"), 1),
                "tags": " / ".join(
                    _SCAN_TAG_LABEL.get(str(t), str(t)) for t in tags[:3]
                ),
                "reason": "；".join((reasons or rejects)[:2]) or "—",
            }
        )
        if len(rows) >= int(limit):
            break
    return rows


def _market_scan_find(report, symbol: str):
    symbol = str(symbol or "").upper()
    for item in list(getattr(report, "items", []) or []):
        if str(getattr(item, "symbol", "") or "").upper() == symbol:
            return item
    return None


def _setup_label(setup: str) -> str:
    return {
        "LONG_TREND": "多头趋势",
        "SHORT_TREND": "空头趋势",
        "REVERSAL": "超跌反弹",
        "NO_DATA": "无行情",
    }.get(str(setup or ""), str(setup or "—"))


def _market_scan_status_label(status: str) -> str:
    return "没入选" if str(status or "") == "REJECT" else "入选"


def _market_scan_detail_html(item) -> str:
    if item is None:
        return (
            '<div style="border-top:1px solid #21262d;margin-top:10px;padding-top:10px">'
            '<div class="qa-card-title">标的详情</div>'
            '<div style="font-size:12px;color:var(--fg3);padding:10px 0">点击表格任意一行查看评分拆解。</div>'
            "</div>"
        )
    symbol = _he(getattr(item, "symbol", ""))
    raw_status = str(getattr(item, "status", "") or "")
    status = _market_scan_status_label(raw_status)
    setup = _setup_label(getattr(item, "setup", ""))
    score = _fmt_num(getattr(item, "score", None), 1)
    parts = getattr(item, "component_scores", {}) or {}
    metrics = getattr(item, "metrics", {}) or {}
    tags = [
        _SCAN_TAG_LABEL.get(str(t), str(t))
        for t in list(getattr(item, "tags", []) or [])
    ]
    reasons = list(getattr(item, "reasons", []) or [])
    rejects = list(getattr(item, "reject_reasons", []) or [])
    status_color = {"入选": "var(--pos)", "没入选": "var(--neg)"}
    metric_bits = [
        ("价格", _fmt_num(metrics.get("price"), 2)),
        (
            "ADV20",
            f"${_fmt_num((metrics.get('adv20') or 0) / 1_000_000, 0)}M"
            if metrics.get("adv20")
            else "—",
        ),
        ("20日", f"{_fmt_num(metrics.get('ret20_pct'), 1)}%"),
        ("60日", f"{_fmt_num(metrics.get('ret60_pct'), 1)}%"),
        ("回撤", f"{_fmt_num(metrics.get('drawdown_pct'), 1)}%"),
        ("波动", f"{_fmt_num(metrics.get('vol20_pct'), 1)}%"),
    ]
    score_bits = [
        ("流动性", _fmt_num(parts.get("liquidity"), 1)),
        ("趋势", _fmt_num(parts.get("trend"), 1)),
        ("风险", _fmt_num(parts.get("risk"), 1)),
        ("热度", _fmt_num(parts.get("hot_bonus"), 1)),
        ("指数", _fmt_num(parts.get("index_bonus"), 1)),
    ]
    metric_html = "".join(
        f'<span class="cp-report-rtag">{_he(name)} {_he(value)}</span>'
        for name, value in metric_bits
    )
    score_html = "".join(
        f'<span class="cp-report-rtag">{_he(name)} {_he(value)}</span>'
        for name, value in score_bits
    )
    tag_html = _rtags(tags)
    reason_html = "".join(
        f"<li>{_he(text)}</li>" for text in (reasons[:5] or ["暂无明确入选理由"])
    )
    reject_html = "".join(f"<li>{_he(text)}</li>" for text in rejects[:5])
    reject_block = (
        f'<div style="margin-top:8px;color:var(--warn);font-size:12px"><b>淘汰/风险：</b><ul>{reject_html}</ul></div>'
        if rejects
        else ""
    )
    return (
        '<div style="border-top:1px solid #21262d;margin-top:10px;padding-top:10px">'
        '<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">'
        f'<div class="qa-card-title" style="margin:0">{symbol}</div>'
        f'<span class="cp-report-rtag" style="color:{status_color.get(status, "var(--fg3)")};background:rgba(255,255,255,.04)">'
        f"{_he(status)}</span>"
        f'<span class="cp-report-rtag">{_he(setup)}</span>'
        f'<span style="font-family:var(--mono);font-size:12px;color:var(--fg2)">score {score}</span>'
        f"{tag_html}"
        "</div>"
        f'<div style="margin-top:8px;display:flex;gap:5px;flex-wrap:wrap">{metric_html}</div>'
        f'<div style="margin-top:6px;display:flex;gap:5px;flex-wrap:wrap">{score_html}</div>'
        f'<div style="margin-top:8px;color:var(--fg2);font-size:12px"><b>入选依据：</b><ul>{reason_html}</ul></div>'
        f"{reject_block}"
        "</div>"
    )


def _fmt_num(value, digits: int = 1) -> str:
    try:
        if value is None:
            return "—"
        return f"{float(value):.{digits}f}"
    except Exception:
        return "—"


def _decision_pool_report_html(report) -> str:
    updated = str(getattr(report, "updated_at", "") or "")
    style = str(getattr(report, "decision_style", "standard") or "standard")
    current = list(getattr(report, "current_symbols", []) or [])
    added = list(getattr(report, "added", []) or [])
    removed = list(getattr(report, "removed", []) or [])
    kept = list(getattr(report, "kept", []) or [])
    slot_usage = dict(getattr(report, "slot_usage", {}) or {})
    warnings = list(getattr(report, "warnings", []) or [])
    if not updated and not current:
        return (
            '<div class="qa-card-title">决策池变动报告</div>'
            '<div class="qa-card-sub">尚未生成。更新决策池后会显示新增、移除和保留原因。</div>'
        )

    usage_html = (
        "".join(
            f'<span class="cp-report-rtag">{_he(k)} {int(v)}</span>'
            for k, v in sorted(slot_usage.items())
        )
        or '<span style="font-size:12px;color:var(--fg3)">暂无类型占用</span>'
    )
    current_html = (
        " ".join(
            f'<span class="cp-report-rtag pos">{_he(symbol)}</span>'
            for symbol in current
        )
        or '<span style="font-size:12px;color:var(--fg3)">空</span>'
    )
    warning_html = "".join(
        f'<div style="font-size:11px;color:var(--warn);margin-top:4px">{_he(w)}</div>'
        for w in warnings[:3]
    )

    def _section(title: str, rows: list, color: str) -> str:
        if not rows:
            return (
                f'<div style="border-top:1px solid #21262d;padding:8px 0">'
                f'<div style="font-size:12px;color:var(--fg3)">{_he(title)}：无</div>'
                f"</div>"
            )
        out = [
            f'<div style="border-top:1px solid #21262d;padding:8px 0"><div class="qa-card-title">{_he(title)}</div>'
        ]
        for row in rows[:10]:
            symbol = _he(getattr(row, "symbol", ""))
            score = _fmt_num(getattr(row, "score", None), 1)
            kind = _he(getattr(row, "decision_type", ""))
            direction = _he(getattr(row, "direction", ""))
            reasons = (
                "；".join(list(getattr(row, "reasons", []) or [])[:3]) or "无明确原因"
            )
            risks = "；".join(list(getattr(row, "risk_flags", []) or [])[:2])
            risk_html = (
                f'<div style="font-size:11px;color:var(--warn);margin-top:2px">风险：{_he(risks)}</div>'
                if risks
                else ""
            )
            out.append(
                '<div style="padding:6px 0">'
                f'<span style="font-family:var(--mono);font-weight:700;color:var(--fg)">{symbol}</span> '
                f'<span style="font-family:var(--mono);color:{color}">{score}</span> '
                f'<span class="cp-report-rtag">{kind}</span>'
                f'<span class="cp-report-rtag">{direction}</span>'
                f'<div style="font-size:11.5px;color:var(--fg2);margin-top:3px">{_he(reasons)}</div>'
                f"{risk_html}"
                "</div>"
            )
        out.append("</div>")
        return "".join(out)

    return (
        '<div style="display:flex;align-items:flex-start;justify-content:space-between;gap:12px;flex-wrap:wrap">'
        '<div><div class="qa-card-title">决策池变动报告</div>'
        f'<div class="qa-card-sub">更新时间 {_he(updated or "—")} · 风格 {_he(_decision_style_label(style))}</div></div>'
        "</div>"
        f'<div style="margin-top:6px"><span style="font-size:12px;color:var(--fg3)">当前：</span>{current_html}</div>'
        f'<div style="margin-top:6px"><span style="font-size:12px;color:var(--fg3)">类型：</span>{usage_html}</div>'
        f"{warning_html}"
        f"{_section('新增', added, 'var(--pos)')}"
        f"{_section('移除', removed, 'var(--neg)')}"
        f"{_section('保留', kept, 'var(--ai)')}"
    )


def _decision_style_label(style: str) -> str:
    return "小资金进攻" if str(style or "").lower() == "aggressive" else "标准"


def _decision_trade_plan_report_html(report) -> str:
    updated = str(getattr(report, "updated_at", "") or "")
    style = str(getattr(report, "decision_style", "standard") or "standard")
    plans = list(getattr(report, "plans", []) or [])
    warnings = list(getattr(report, "warnings", []) or [])
    ready_count = int(getattr(report, "ready_count", 0) or 0)
    wait_count = int(getattr(report, "wait_count", 0) or 0)
    blocked_count = int(getattr(report, "blocked_count", 0) or 0)
    if not updated and not plans:
        return (
            '<div class="qa-card-title">决策交易计划</div>'
            '<div class="qa-card-sub">尚未生成。更新决策池后会自动生成触发价、止损、止盈、仓位和执行状态。</div>'
        )

    warning_html = "".join(
        f'<div style="font-size:11px;color:var(--warn);margin-top:4px">{_he(w)}</div>'
        for w in warnings[:3]
    )
    action_label = {
        "TRADE_READY": "可交易",
        "WAIT_TRIGGER": "等触发",
        "HEDGE_ONLY": "对冲参考",
        "REVIEW_ONLY": "仅复核",
        "BLOCKED": "阻断",
    }
    action_color = {
        "TRADE_READY": "var(--pos)",
        "WAIT_TRIGGER": "var(--warn)",
        "HEDGE_ONLY": "var(--ai)",
        "REVIEW_ONLY": "var(--fg3)",
        "BLOCKED": "var(--neg)",
    }
    rows = []
    for plan in plans[:7]:
        action = str(getattr(plan, "action", "") or "")
        color = action_color.get(action, "var(--fg3)")
        label = action_label.get(action, action or "未知")
        symbol = _he(getattr(plan, "symbol", ""))
        rank = int(getattr(plan, "rank", 0) or 0)
        direction = _he(getattr(plan, "direction", ""))
        kind = _he(getattr(plan, "decision_type", ""))
        score = _fmt_num(getattr(plan, "score", None), 1)
        price = _fmt_num(getattr(plan, "latest_price", None), 2)
        atr_pct = _fmt_num(getattr(plan, "atr_pct", None), 2)
        stop = _fmt_num(getattr(plan, "stop_loss", None), 2)
        target = _fmt_num(getattr(plan, "take_profit", None), 2)
        weight = _fmt_num(getattr(plan, "suggested_weight_pct", None), 1)
        risk = _fmt_num(getattr(plan, "risk_per_trade_pct", None), 2)
        max_pos = _fmt_num(getattr(plan, "max_position_pct", None), 1)
        source_status = _he(getattr(plan, "source_status", ""))
        trigger = _he(getattr(plan, "entry_trigger", "") or "等待盘中确认")
        invalidation = _he(getattr(plan, "invalidation", "") or "触发失败后重新评估")
        blocked = _he(getattr(plan, "blocked_reason", "") or "")
        risks = list(getattr(plan, "risk_flags", []) or [])[:3]
        reasons = list(getattr(plan, "reasons", []) or [])[:3]
        reason_html = (
            "".join(f"<li>{_he(text)}</li>" for text in reasons)
            or "<li>暂无明确入池理由</li>"
        )
        risk_html = (
            "".join(f"<li>{_he(text)}</li>" for text in risks)
            or "<li>暂无显著风险标记</li>"
        )
        block_html = (
            f'<div style="font-size:11.5px;color:var(--neg);margin-top:5px">阻断原因：{blocked}</div>'
            if blocked
            else ""
        )
        rows.append(
            '<div style="border-top:1px solid #21262d;padding:12px 0">'
            '<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">'
            f'<span style="font-family:var(--mono);color:var(--fg3);font-size:11px">#{rank}</span>'
            f'<b style="color:var(--fg);font-size:15px">{symbol}</b>'
            f'<span style="font-family:var(--mono);color:{color};font-size:13px">{score}</span>'
            f'<span class="cp-report-rtag" style="color:{color};background:rgba(255,255,255,.04)">{_he(label)}</span>'
            f'<span class="cp-report-rtag">{kind}</span>'
            f'<span class="cp-report-rtag">{direction}</span>'
            f'<span class="cp-report-rtag">数据 {_he(source_status)}</span>'
            "</div>"
            '<div style="margin-top:8px;display:grid;grid-template-columns:repeat(6,minmax(82px,1fr));gap:6px">'
            f"{_plan_metric('价格', price)}"
            f"{_plan_metric('ATR%', atr_pct)}"
            f"{_plan_metric('止损', stop)}"
            f"{_plan_metric('止盈', target)}"
            f"{_plan_metric('建议仓位%', weight)}"
            f"{_plan_metric('单笔风险%', risk)}"
            "</div>"
            f'<div style="margin-top:6px;font-size:11px;color:var(--fg3)">最大仓位 {max_pos}% · 仅通过 AI 安全门与确定性风控后自动执行</div>'
            f'<div style="margin-top:8px;padding:8px 10px;background:rgba(255,255,255,.025);border:1px solid #21262d;border-radius:6px">'
            f'<div style="font-size:11.5px;color:var(--fg2);line-height:1.45">触发：{trigger}</div>'
            f'<div style="font-size:11.5px;color:var(--fg2);line-height:1.45;margin-top:3px">失效：{invalidation}</div>'
            f"{block_html}"
            "</div>"
            '<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:8px">'
            f'<div style="font-size:11.5px;color:var(--fg2);line-height:1.45"><b style="color:var(--fg3)">依据</b><ul style="margin:4px 0 0 18px;padding:0">{reason_html}</ul></div>'
            f'<div style="font-size:11.5px;color:var(--warn);line-height:1.45"><b>风险</b><ul style="margin:4px 0 0 18px;padding:0">{risk_html}</ul></div>'
            "</div>"
            "</div>"
        )

    return (
        '<div style="display:flex;align-items:flex-start;justify-content:space-between;gap:12px;flex-wrap:wrap">'
        '<div><div class="qa-card-title">决策交易计划</div>'
        f'<div class="qa-card-sub">更新时间 {_he(updated or "—")} · 风格 {_he(_decision_style_label(style))} · 可交易 {ready_count} · 等触发 {wait_count} · 阻断 {blocked_count}</div></div>'
        "</div>"
        f"{warning_html}"
        f'<div style="margin-top:8px">{"".join(rows)}</div>'
    )


def _plan_metric(label: str, value: str) -> str:
    return (
        '<div style="background:rgba(255,255,255,.025);border:1px solid #21262d;border-radius:6px;padding:6px 8px">'
        f'<div style="font-size:10px;color:var(--fg3)">{_he(label)}</div>'
        f'<div style="font-family:var(--mono);font-size:12px;color:var(--fg)">{_he(value)}</div>'
        "</div>"
    )


def _selection_pool_html(result, title: str) -> str:
    items = list(getattr(result, "items", []) or [])
    layer = getattr(result, "layer", "")
    updated = str(getattr(result, "updated_at", "") or "")
    source_size = int(getattr(result, "source_size", 0) or 0)
    selected_size = int(getattr(result, "selected_size", len(items)) or 0)
    warnings = list(getattr(result, "warnings", []) or [])

    if not items:
        return (
            f'<div style="display:flex;align-items:center;justify-content:space-between;gap:10px">'
            f'<div><div class="qa-card-title">{_he(title)}</div>'
            f'<div class="qa-card-sub">尚未生成</div></div></div>'
            f'<div style="color:var(--fg3);font-size:12px;padding:18px 0">暂无股票</div>'
        )

    status_label = {
        "CORE": "核心关注",
        "RESEARCH": "研究",
        "FOCUS_READY": "重点",
        "SETUP": "形态",
        "COOL_DOWN": "降温",
        "ENTRY_READY": "可交易",
        "WAIT_TRIGGER": "等触发",
        "WAIT_BREAKOUT": "等突破",
        "MARKET_ANCHOR": "锚点",
        "AVOID_NOW": "暂避",
        "WATCH": "观察",
        "BENCH": "备用",
        "AVOID": "暂避",
    }
    status_color = {
        "CORE": "var(--pos)",
        "FOCUS_READY": "var(--pos)",
        "ENTRY_READY": "var(--pos)",
        "WAIT_TRIGGER": "var(--warn)",
        "WATCH": "var(--ai)",
        "SETUP": "var(--warn)",
        "WAIT_BREAKOUT": "var(--warn)",
        "RESEARCH": "var(--warn)",
        "BENCH": "var(--fg3)",
        "MARKET_ANCHOR": "var(--fg3)",
        "COOL_DOWN": "var(--neg)",
        "AVOID": "var(--neg)",
        "AVOID_NOW": "var(--neg)",
    }
    layer_hint = {
        "long_term": "月/季度",
        "weekly_focus": "周级",
        "daily_decision": "3-7 个",
    }.get(layer, "")
    warning_html = "".join(
        f'<div style="font-size:11px;color:var(--warn);margin-top:3px">{_he(w)}</div>'
        for w in warnings[:2]
    )
    if layer == "daily_decision":
        return _decision_pool_detail_html(
            title=title,
            items=items,
            updated=updated,
            source_size=source_size,
            selected_size=selected_size,
            warning_html=warning_html,
            status_label=status_label,
            status_color=status_color,
        )

    rows = []
    for item in items[:18]:
        symbol = _he(getattr(item, "symbol", ""))
        rank = int(getattr(item, "rank", 0) or 0)
        score = float(getattr(item, "score", 0) or 0)
        status = str(getattr(item, "status", ""))
        confidence = _he(getattr(item, "data_confidence", "低"))
        color = status_color.get(status, "var(--fg3)")
        label = _he(status_label.get(status, status))
        reasons = (
            "；".join(list(getattr(item, "reasons", []) or [])[:2]) or "无明确理由"
        )
        risks = "；".join(list(getattr(item, "risk_flags", []) or [])[:2])
        parts = getattr(item, "component_scores", {}) or {}
        part_html = " ".join(
            f'<span style="color:var(--fg3);font-size:10.5px">{_he(k)}='
            f"{float(v):.0f}</span>"
            for k, v in parts.items()
            if v is not None
        )
        risk_html = (
            f'<div style="color:var(--warn);font-size:11px;margin-top:3px">风险：{_he(risks)}</div>'
            if risks
            else ""
        )
        rows.append(
            f'<div style="border-top:1px solid #21262d;padding:8px 0">'
            f'<div style="display:flex;align-items:center;gap:7px;flex-wrap:wrap">'
            f'<span style="font-family:var(--mono);color:var(--fg3);font-size:11px">#{rank}</span>'
            f'<b style="color:var(--fg);font-size:13px">{symbol}</b>'
            f'<span style="font-family:var(--mono);color:{color};font-size:12px">{score:.1f}</span>'
            f'<span class="cp-report-rtag" style="color:{color};background:rgba(255,255,255,.04)">'
            f"{label}</span>"
            f'<span style="color:var(--fg3);font-size:11px">置信 {confidence}</span>'
            f"</div>"
            f'<div style="margin-top:4px">{part_html}</div>'
            f'<div style="color:var(--fg2);font-size:11.5px;margin-top:4px;line-height:1.45">'
            f"{_he(reasons)}</div>"
            f"{risk_html}"
            f"</div>"
        )

    return (
        f'<div style="display:flex;align-items:flex-start;justify-content:space-between;gap:10px">'
        f'<div><div class="qa-card-title">{_he(title)}</div>'
        f'<div class="qa-card-sub">{_he(layer_hint)} · 来源 {source_size} · 入池 {selected_size}</div></div>'
        f'<span style="font-size:10.5px;color:var(--fg3);font-family:var(--mono)">'
        f"{_he(updated[:16])}</span>"
        f"</div>"
        f"{warning_html}"
        f'<div style="margin-top:8px">{"".join(rows)}</div>'
    )


def _decision_pool_detail_html(
    *,
    title: str,
    items: list,
    updated: str,
    source_size: int,
    selected_size: int,
    warning_html: str,
    status_label: dict,
    status_color: dict,
) -> str:
    rows = []
    for item in items[:7]:
        symbol = _he(getattr(item, "symbol", ""))
        rank = int(getattr(item, "rank", 0) or 0)
        score = float(getattr(item, "score", 0) or 0)
        status = str(getattr(item, "status", ""))
        confidence = _he(getattr(item, "data_confidence", "低"))
        color = status_color.get(status, "var(--fg3)")
        label = _he(status_label.get(status, status))
        reasons_all = list(getattr(item, "reasons", []) or [])
        risks_all = list(getattr(item, "risk_flags", []) or [])
        parts = getattr(item, "component_scores", {}) or {}
        kind = _reason_value(reasons_all, "类型") or "—"
        direction = _reason_value(reasons_all, "方向") or "—"
        style = _decision_style_label(_reason_value(reasons_all, "风格") or "")
        trigger = next((r for r in reasons_all if r.startswith("触发参考")), "")
        invalid = next((r for r in reasons_all if r.startswith("失效参考")), "")
        sizing = next((r for r in reasons_all if r.startswith("仓位提示")), "")
        main_reasons = [
            r
            for r in reasons_all
            if not r.startswith(
                ("类型 ", "方向 ", "风格 ", "触发参考", "失效参考", "仓位提示")
            )
        ][:5]
        intro = _decision_item_intro(
            kind, direction, status, score, main_reasons, risks_all
        )
        thesis = _decision_item_thesis(kind, direction)
        score_bits = [
            ("长期", parts.get("long_pool")),
            ("形态", parts.get("setup")),
            ("质量", parts.get("quality")),
            ("AI", parts.get("ai")),
        ]
        score_html = "".join(
            f'<span class="cp-report-rtag">{_he(name)} {_fmt_num(value, 1)}</span>'
            for name, value in score_bits
            if value is not None
        )
        reason_html = "".join(
            f"<li>{_he(text)}</li>" for text in (main_reasons or ["暂无更多理由"])
        )
        risk_html = (
            "".join(f"<li>{_he(text)}</li>" for text in risks_all[:4])
            or "<li>暂无显著风险标记</li>"
        )
        notes_html = "".join(
            f'<div style="font-size:11.5px;color:var(--fg2);line-height:1.45">{_he(text)}</div>'
            for text in [trigger, invalid, sizing]
            if text
        )
        rows.append(
            '<div style="border-top:1px solid #21262d;padding:12px 0">'
            '<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">'
            f'<span style="font-family:var(--mono);color:var(--fg3);font-size:11px">#{rank}</span>'
            f'<b style="color:var(--fg);font-size:15px">{symbol}</b>'
            f'<span style="font-family:var(--mono);color:{color};font-size:13px">{score:.1f}</span>'
            f'<span class="cp-report-rtag" style="color:{color};background:rgba(255,255,255,.04)">{label}</span>'
            f'<span class="cp-report-rtag">{_he(kind)}</span>'
            f'<span class="cp-report-rtag">{_he(direction)}</span>'
            f'<span style="color:var(--fg3);font-size:11px">置信 {confidence}</span>'
            f'<span style="color:var(--fg3);font-size:11px">风格 {_he(style)}</span>'
            "</div>"
            f'<div style="margin-top:7px;font-size:12px;color:var(--fg2);line-height:1.5">{_he(intro)}</div>'
            f'<div style="margin-top:4px;font-size:11.5px;color:var(--fg3);line-height:1.45">{_he(thesis)}</div>'
            f'<div style="margin-top:7px;display:flex;gap:5px;flex-wrap:wrap">{score_html}</div>'
            f'<div style="margin-top:8px;padding:8px 10px;background:rgba(255,255,255,.025);border:1px solid #21262d;border-radius:6px">{notes_html or '<span style="font-size:11.5px;color:var(--fg3)">暂无触发/失效参考</span>'}</div>'
            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:8px">'
            f'<div style="font-size:11.5px;color:var(--fg2);line-height:1.45"><b style="color:var(--fg3)">理由</b><ul style="margin:4px 0 0 18px;padding:0">{reason_html}</ul></div>'
            f'<div style="font-size:11.5px;color:var(--warn);line-height:1.45"><b>风险</b><ul style="margin:4px 0 0 18px;padding:0">{risk_html}</ul></div>'
            "</div>"
            "</div>"
        )

    return (
        f'<div style="display:flex;align-items:flex-start;justify-content:space-between;gap:10px">'
        f'<div><div class="qa-card-title">{_he(title)}</div>'
        f'<div class="qa-card-sub">自动交易候选 · 来源 {source_size} · 入池 {selected_size}</div></div>'
        f'<span style="font-size:10.5px;color:var(--fg3);font-family:var(--mono)">'
        f"{_he(updated[:16])}</span>"
        f"</div>"
        f"{warning_html}"
        f'<div style="margin-top:8px">{"".join(rows)}</div>'
    )


def _reason_value(reasons: list[str], prefix: str) -> str:
    head = f"{prefix} "
    for reason in reasons:
        if str(reason).startswith(head):
            return str(reason).replace(head, "", 1)
    return ""


def _decision_item_intro(
    kind: str,
    direction: str,
    status: str,
    score: float,
    reasons: list[str],
    risks: list[str],
) -> str:
    status_text = {
        "ENTRY_READY": "当前条件较完整，可进入自动交易重点观察",
        "WAIT_TRIGGER": "当前更适合等待触发，不建议无条件追入",
        "WATCH": "处于观察状态，需要盘中确认",
        "BENCH": "作为备用候选，优先级低于可交易标的",
    }.get(status, "需要结合盘中确认")
    kind_text = {
        "LONG_TREND": "趋势延续候选",
        "AGGRESSIVE_MOMENTUM": "小资金进攻动量候选",
        "ETF_MACRO": "ETF/宏观表达候选",
        "REVERSAL": "反转修复候选",
        "SHORT_TREND": "弱势做空候选",
    }.get(kind, kind or "观察候选")
    direction_text = {
        "LONG": "偏多",
        "SHORT": "偏空",
        "HEDGE": "对冲/防守",
        "WATCH": "观察",
    }.get(direction, direction or "观察")
    reason_text = "；".join(reasons[:2]) if reasons else "入池依据较少"
    risk_text = f"；主要风险：{'；'.join(risks[:2])}" if risks else ""
    return f"{kind_text}，方向 {direction_text}，综合分 {score:.1f}。{status_text}。核心依据：{reason_text}{risk_text}。"


def _decision_item_thesis(kind: str, direction: str) -> str:
    if kind == "AGGRESSIVE_MOMENTUM":
        return "入池逻辑：它不是稳健白马筛选，而是给小资金模式捕捉更强的短线弹性；适合小仓、快进快出、严格止损。"
    if kind == "LONG_TREND":
        return "入池逻辑：趋势结构仍占优，适合等待突破延续或回踩确认，自动交易不应追第一根过热K线。"
    if kind == "ETF_MACRO":
        return "入池逻辑：用于表达指数、行业或宏观方向，也可作为个股信号不清时的更干净交易载体。"
    if kind == "REVERSAL":
        return (
            "入池逻辑：来自超跌后的修复机会，必须等待止跌确认；它的仓位应小于趋势候选。"
        )
    if kind == "SHORT_TREND" or direction == "SHORT":
        return "入池逻辑：弱势结构用于做空或回避多头暴露，必须确认市场环境允许做空。"
    return "入池逻辑：当前分数进入候选范围，但还需要盘中触发条件确认。"


_TASK_STATE_LABELS = {
    "BUSY": ("处理中", ""),
    "SUCCESS": ("成功", "pos"),
    "EMPTY": ("空结果", ""),
    "ERROR": ("失败", "neg"),
}


def _render_tasks():
    _page_head("任务中心", "AI 任务队列 · 全部动作活动日志", badge="live")

    with ui.element("div").classes("qa-card"):
        ui.label("AI 任务队列").classes("qa-card-title")
        ui.label(
            "后台任务严格按到达顺序依次执行，不会并发抢占同一份数据；"
            "预计时间基于该动作最近历史耗时估算。"
        ).classes("qa-card-sub")
        queue_area = ui.column().style("gap:8px;width:100%;margin-top:8px")

    with ui.element("div").classes("qa-card").style("margin-top:12px"):
        ui.label("活动日志").classes("qa-card-title")
        ui.label("全部按钮动作的执行历史，含成功/空结果/失败与耗时。").classes(
            "qa-card-sub"
        )
        with ui.row().classes("items-end gap-3 flex-wrap").style("margin-top:8px"):
            categories = ["全部"] + sorted({a.category for a in BUTTON_ACTIONS})
            cat_sel = (
                ui.select(categories, value="全部", label="分类")
                .props("dark dense outlined")
                .style("width:140px")
            )
            search_in = (
                ui.input("搜索动作名称", placeholder="例如：个股分析")
                .props("dark dense outlined clearable")
                .style("width:220px")
            )
        log_area = ui.column().style("gap:2px;width:100%;margin-top:10px")

    def _render_queue():
        queue_area.clear()
        view = queue_view(_ASYNC_ACTION_RUNNER, _ACTION_AUDIT_STORE)
        with queue_area:
            if not view["items"]:
                _empty("当前没有排队或执行中的任务", "🕒")
                return
            ui.html(
                '<div class="qa-note">队列全部完成预计还需 '
                f'<b>{format_health_age(view["total_seconds"])}</b></div>'
            )
            for position, item in enumerate(view["items"]):
                if item["status"] == "RUNNING":
                    status_html = (
                        '<span class="qa-badge live">▶ 执行中</span>'
                        f' 已耗时 {format_health_age(item["elapsed_seconds"])}'
                        f' · 预计还需 {format_health_age(item["eta_seconds"])}'
                    )
                else:
                    status_html = (
                        f'<span style="color:var(--fg3)">⏳ 排队中（第 {position} 位）</span>'
                        f' · 预计 {format_health_age(item["eta_seconds"])} 后完成'
                    )
                ui.html(
                    '<div class="qa-note" style="display:flex;gap:10px;align-items:center;flex-wrap:wrap">'
                    f'<b>{_he(item["label"])}</b>'
                    f'<span style="color:var(--fg3);font-size:11px">{_he(item["category"])}</span>'
                    f"{status_html}"
                    "</div>"
                )

    def _render_log():
        log_area.clear()
        category = None if cat_sel.value == "全部" else cat_sel.value
        rows = _ACTION_AUDIT_STORE.recent(limit=200, category=category)
        query = (search_in.value or "").strip()
        if query:
            rows = [
                r
                for r in rows
                if query in r["label"] or query in r["action_id"]
            ]
        with log_area:
            if not rows:
                _empty("暂无匹配的活动记录", "📋")
                return
            for row in rows[:100]:
                state_label, tone = _TASK_STATE_LABELS.get(row["state"], (row["state"], ""))
                duration = (
                    format_health_age(row["duration_seconds"])
                    if row["duration_seconds"] is not None
                    else "—"
                )
                ui.html(
                    '<div class="qa-note" style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;font-size:12px">'
                    f'<span style="color:var(--fg3);font-family:var(--mono)">{_fmt_time(row["started_at"])}</span>'
                    f'<b>{_he(row["label"])}</b>'
                    f'<span style="color:var(--fg3)">{_he(row["category"])}</span>'
                    f'<span class="{tone}">{state_label}</span>'
                    f'<span style="color:var(--fg3)">耗时 {duration}</span>'
                    f'<span style="color:var(--fg3);flex:1;min-width:120px">{_he(row["user_message"])}</span>'
                    "</div>"
                )

    cat_sel.on_value_change(lambda e: _render_log())
    search_in.on_value_change(lambda e: _render_log())

    def update():
        _render_queue()
        _render_log()

    update()
    return update


def _render_operations():
    """Thin operations hub; production controls remain in their bounded views."""
    _page_head(
        "系统运营",
        "集中管理 Runtime、风险、通知、诊断与恢复",
        badge="live",
    )
    with (
        ui.element("div")
        .classes("qa-grid")
        .style("grid-template-columns:repeat(auto-fit,minmax(240px,1fr))")
    ):
        with ui.element("div").classes("qa-card"):
            ui.label("引擎与通知").classes("qa-card-title")
            ui.label(
                "启动/停止 Runtime，检查心跳、启动对账、Discord 授权与发送状态。"
            ).classes("qa-card-sub")
            system_link = ui.element("div").classes("qa-nav-item")
            with system_link:
                ui.label("进入引擎与通知")
            system_link.on("click", lambda: _select("system"))
        with ui.element("div").classes("qa-card"):
            ui.label("风控与急停").classes("qa-card-title")
            ui.label(
                "查看回撤、风险事件和 Kill Switch；危险操作与普通监控分开。"
            ).classes("qa-card-sub")
            risk_link = ui.element("div").classes("qa-nav-item")
            with risk_link:
                ui.label("进入风控与急停")
            risk_link.on("click", lambda: _select("risk"))
        with ui.element("div").classes("qa-card"):
            ui.label("诊断与复盘").classes("qa-card-title")
            ui.label("运行异常检测、查看绩效复盘和只读整改建议。").classes(
                "qa-card-sub"
            )
            maintenance_link = ui.element("div").classes("qa-nav-item")
            with maintenance_link:
                ui.label("进入诊断与复盘")
            maintenance_link.on("click", lambda: _select("maintenance"))
        with ui.element("div").classes("qa-card"):
            ui.label("任务中心").classes("qa-card-title")
            ui.label(
                "AI 任务队列（排队/执行中/预计时间）与全部动作活动日志，按类型筛选。"
            ).classes("qa-card-sub")
            tasks_link = ui.element("div").classes("qa-nav-item")
            with tasks_link:
                ui.label("进入任务中心")
            tasks_link.on("click", lambda: _select("tasks"))
    with ui.element("div").classes("qa-card").style("margin-top:12px"):
        ui.label("平台边界").classes("qa-card-title")
        ui.label(
            "NiceGUI 负责监控、研究和运营；Runtime 仍是唯一订单提交者。"
            "旧版自定义 Agent 已从主导航移除，不进入生产交易链。"
        ).classes("qa-card-sub")

    def update():
        return None

    return update


_RENDERERS = {
    "overview": _render_overview,
    "activity": _render_activity,
    "selection": _render_selection_pools,
    "research": _render_research,
    "operations": _render_operations,
    "risk": _render_risk,
    "maintenance": _render_maintenance,
    "system": _render_system,
    "tasks": _render_tasks,
}

# 旧 tab 名称 → 新名称（保留用户偏好跨版本兼容）
_TAB_MIGRATION = {
    "market_env": "overview",
    "t1_selection": "research",
    "t2_factor": "research",
    "t3_trading": "activity",
    "t4_risk": "operations",
    "t5_maintenance": "operations",
    "universe": "selection",
    "models": "overview",
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


def _on_research_symbol_click(e) -> None:
    symbol = str(getattr(e, "args", "") or "").strip().upper()
    if not symbol:
        return
    _pending_archive_filter["symbol"] = symbol
    _select("research")


ui.on("research_symbol_click", _on_research_symbol_click)


# ═══════════════════════════════════════════════════════════════════════════
# 定时器：顶栏 + 当前页增量更新（绝不 clear+rebuild，所以不闪）
# ═══════════════════════════════════════════════════════════════════════════


def _update_topbar():
    running = _engine_running()
    top_engine.set_text("● 运行中" if running else "○ 停止")
    top_engine.classes(remove="pos neg", add="pos" if running else "neg")
    # AutoTrade 状态必须读引擎自己写出来的 runtime_status.json，不能读
    # Dashboard 自己的 _cfg——Dashboard 和引擎是两个独立进程，_cfg 只反映
    # Dashboard 上次用来启动引擎时传的参数，引擎也可能是手动起的，两者会
    # 对不上。
    if not running:
        top_autotrade.set_text("—")
        top_autotrade.classes(remove="pos neg ai")
    else:
        status = read_runtime_status()
        if status is None:
            top_autotrade.set_text("未知")
            top_autotrade.classes(remove="pos neg ai")
        elif status.get("auto_trade_paper"):
            top_autotrade.set_text("● 已开启")
            top_autotrade.classes(remove="pos neg", add="ai")
        else:
            top_autotrade.set_text("○ 关闭")
            top_autotrade.classes(remove="pos neg ai")
    snapshot = _ASYNC_ACTION_RUNNER.queue_snapshot()
    pending_count = len(snapshot["queued"]) + (1 if snapshot["current"] else 0)
    top_tasks.set_text(f"{pending_count} 个" if pending_count else "空闲")
    top_tasks.classes(remove="pos neg ai", add="ai" if pending_count else "")
    hb = heartbeat()
    if hb is not None:
        secs = (datetime.now(timezone.utc) - hb).total_seconds()
        top_hb.set_text(format_health_age(secs))
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


_initial_tab = _pref("active_tab", "overview")
_select(
    _TAB_MIGRATION.get(
        _initial_tab, _initial_tab if _initial_tab in _RENDERERS else "overview"
    )
)
_update_topbar()
ui.timer(_REFRESH_SEC, _tick)

if __name__ in {"__main__", "__mp_main__"}:
    _web = os.getenv("QUANT_WEB") == "1" or "--web" in sys.argv
    _port_arg = next(
        (a.split("=", 1)[1] for a in sys.argv if a.startswith("--port=")), None
    )
    _port = int(_port_arg or os.getenv("QUANT_PORT", "8080"))
    _host = os.getenv("QUANT_HOST", "127.0.0.1")
    try:
        ui.run(
            title="美股K线 · DuckDB + Alpaca 实时",
            reload=False,
            native=not _web,
            host=_host if _web else None,
            port=_port if _web else None,
            show=_web,
            window_size=(1600, 1000) if not _web else None,
            dark=True,
        )
    except KeyboardInterrupt:
        logger.info("NiceGUI stopped")
