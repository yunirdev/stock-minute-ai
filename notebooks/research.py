"""
notebooks/research.py
Marimo reactive research notebook for the trading platform.

Why Marimo (not Streamlit) for research:
  - Reactive dataflow: changing one control re-runs only the cells that depend on
    it, not the whole script — no flicker, no st.session_state juggling.
  - No hidden state; the file is plain Python and diffs cleanly in git.

Run it:
    marimo edit notebooks/research.py     # interactive editing
    marimo run  notebooks/research.py     # app mode

It reuses the SAME engine the UI and live path use (trader.engine.simulate), so
research results match production. Any "signal source" that writes strat_signal /
strat_exec_px columns plugs in here — TA strategies today, an AI model later.
"""
import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    _root = Path(__file__).resolve().parents[1]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    import marimo as mo
    import plotly.graph_objects as go
    return go, mo


@app.cell
def _():
    from trader.data_cache import get_bars, list_cached_files
    from trader.engine import simulate
    from trader.strategy_core import (
        STRATEGY_OPTIONS, DEFAULT_STRATEGY_PARAMS, compute_signals,
    )
    return (
        DEFAULT_STRATEGY_PARAMS, STRATEGY_OPTIONS, compute_signals,
        get_bars, list_cached_files, simulate,
    )


@app.cell
def _(list_cached_files, mo):
    _files = list_cached_files()
    symbols = sorted({f["文件"].rsplit("_", 1)[0] for f in _files}) or ["QQQ"]
    timeframes = sorted({f["文件"].rsplit("_", 1)[1].replace(".parquet", "") for f in _files}) or ["30m"]
    mo.md(f"# 📈 研究台\n本地缓存：**{len(_files)}** 个文件 · {len(symbols)} 个标的")
    return symbols, timeframes


@app.cell
def _(STRATEGY_OPTIONS, mo, symbols, timeframes):
    chartable = [s for s in STRATEGY_OPTIONS if s != "分期定投(20次均匀)"]
    sym = mo.ui.dropdown(options=symbols, value=symbols[0], label="标的")
    tf = mo.ui.dropdown(options=timeframes, value=timeframes[0], label="周期")
    strat = mo.ui.dropdown(
        options=chartable,
        value="5/20均线金叉死叉" if "5/20均线金叉死叉" in chartable else chartable[0],
        label="策略",
    )
    capital = mo.ui.number(start=1_000, stop=10_000_000, value=10_000, step=1_000, label="本金")
    fill = mo.ui.dropdown(options=["next_open", "close"], value="next_open", label="成交")
    lev = mo.ui.slider(start=1.0, stop=5.0, value=1.0, step=0.5, label="杠杆", show_value=True)
    risk = mo.ui.switch(value=False, label="风控熔断")
    controls = mo.hstack([sym, tf, strat, capital, fill, lev, risk], justify="start", wrap=True)
    controls
    return capital, fill, lev, risk, strat, sym, tf


@app.cell
def _(
    DEFAULT_STRATEGY_PARAMS, capital, compute_signals, fill, get_bars,
    lev, mo, risk, simulate, strat, sym, tf,
):
    df = get_bars(sym.value, tf.value)
    if df is None or df.empty:
        result = None
        summary = mo.md(f"⚠️ 本地无 **{sym.value} {tf.value}** 数据，请先在主程序下载。")
    else:
        df_sig = compute_signals(df.copy(), strat.value, **DEFAULT_STRATEGY_PARAMS)
        result = simulate(
            df_sig, capital=float(capital.value), leverage=float(lev.value),
            fill=fill.value, risk_halt=bool(risk.value),
        )
        _wr = f"{result.win_rate:.1%}" if result.closed_trades else "—"
        summary = mo.md(
            f"""
| 最终权益 | 总收益 | 平仓次数 | 胜率 | 数据 |
|---|---|---|---|---|
| ${result.final_equity:,.0f} | {result.total_return:+.2%} | {result.closed_trades} | {_wr} | {len(df)} 根 |
"""
        )
    summary
    return (result,)


@app.cell
def _(go, mo, result):
    if result is not None and not result.equity_curve.empty:
        _fig = go.Figure()
        _fig.add_trace(go.Scatter(
            x=result.equity_curve.index, y=result.equity_curve.values,
            mode="lines", name="权益", line=dict(width=2, color="#00b4d8"),
        ))
        _fig.add_hline(y=result.initial_capital, line=dict(width=1, dash="dot"), opacity=0.4)
        _fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10),
                           title="权益曲线", showlegend=False)
        chart = mo.ui.plotly(_fig)
    else:
        chart = mo.md("")
    chart
    return


if __name__ == "__main__":
    app.run()
