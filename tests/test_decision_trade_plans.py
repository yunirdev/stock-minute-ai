from __future__ import annotations

from pathlib import Path

import pandas as pd


def _bars(rows: int = 80, start: float = 100.0, step: float = 0.8) -> pd.DataFrame:
    close = [start + i * step for i in range(rows)]
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=rows, freq="D", tz="UTC"),
        "open": [value - 0.2 for value in close],
        "high": [value + 1.0 for value in close],
        "low": [value - 1.0 for value in close],
        "close": close,
        "volume": [1_000_000 + i * 1_000 for i in range(rows)],
    })


def _item(symbol: str = "NVDA", *, status: str = "ENTRY_READY"):
    import trader.selection_pools as sp

    return sp.PoolItem(
        symbol=symbol,
        rank=1,
        score=78.0,
        status=status,
        data_confidence="高",
        layer=sp.DAILY_DECISION,
        reasons=["类型 LONG_TREND", "方向 LONG", "20/50 日结构偏多"],
        risk_flags=[],
    )


def test_decision_trade_plan_has_prices(monkeypatch):
    import trader.decision_trade_plans as dtp

    monkeypatch.setattr(dtp, "get_bars", lambda _symbol, _tf: _bars())

    report = dtp.build_decision_trade_plan_report([_item()], save=False)
    plan = report.plans[0]

    assert report.ready_count == 1
    assert plan.action == "TRADE_READY"
    assert plan.stop_loss < plan.latest_price < plan.take_profit
    assert plan.suggested_weight_pct > 0
    assert "站稳" in plan.entry_trigger


def test_decision_trade_plan_blocks_missing_bars(monkeypatch):
    import trader.decision_trade_plans as dtp

    monkeypatch.setattr(dtp, "get_bars", lambda _symbol, _tf: pd.DataFrame())

    report = dtp.build_decision_trade_plan_report([_item()], save=False)
    plan = report.plans[0]

    assert report.blocked_count == 1
    assert plan.action == "BLOCKED"
    assert plan.source_status == "NO_BARS"
    assert "缺少" in plan.blocked_reason


def test_executable_symbols_reads_saved_non_blocked_plans(monkeypatch, tmp_path: Path):
    import trader.decision_trade_plans as dtp

    path = tmp_path / "decision_trade_plans.json"
    monkeypatch.setattr(dtp, "get_bars", lambda _symbol, _tf: _bars())

    dtp.build_decision_trade_plan_report(
        [_item("NVDA"), _item("MSFT", status="WAIT_TRIGGER")],
        save=True,
        path=path,
    )

    assert dtp.executable_symbols(limit=7, path=path) == ["NVDA", "MSFT"]
