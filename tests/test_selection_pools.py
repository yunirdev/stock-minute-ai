from __future__ import annotations

from pathlib import Path

import pandas as pd


def _bars(start: float = 100.0, step: float = 0.4, rows: int = 240) -> pd.DataFrame:
    close = [start + i * step for i in range(rows)]
    return pd.DataFrame({
        "close": close,
        "volume": [1_000_000 + i * 500 for i in range(rows)],
    })


def test_long_term_pool_scores_and_ranks(monkeypatch):
    import trader.selection_pools as sp

    def fake_bars(symbol: str, timeframe: str) -> pd.DataFrame:
        if symbol == "NVDA":
            return _bars(step=0.8)
        return _bars(step=-0.05)

    monkeypatch.setattr(sp, "_load_bars", fake_bars)
    monkeypatch.setattr(sp, "_load_ai_scores", lambda _path: {"NVDA": 82.0, "AAPL": 55.0})

    result = sp.build_long_term_pool(["AAPL", "NVDA"], limit=2, ai_db_path=":memory:")

    assert result.layer == sp.LONG_TERM
    assert [item.symbol for item in result.items] == ["NVDA", "AAPL"]
    assert result.items[0].status in {"CORE", "WATCH"}
    assert result.items[0].component_scores["trend"] is not None


def test_selection_pipeline_builds_long_and_decision_layers(monkeypatch):
    import trader.selection_pools as sp

    monkeypatch.setattr(sp, "_load_bars", lambda _symbol, _tf: _bars(step=0.55))
    monkeypatch.setattr(sp, "_load_ai_scores", lambda _path: {"NVDA": 80.0, "AAPL": 62.0})
    monkeypatch.setattr(sp, "save_daily_candidates", lambda _rows: None)
    monkeypatch.setattr(sp, "save_decision_pool_report", lambda _report: None)
    monkeypatch.setattr(
        sp,
        "load_selection_pool",
        lambda layer, path=sp._STORE: sp.PoolResult(layer=layer, updated_at="", source_size=0, selected_size=0, items=[]),
    )

    results = sp.rebuild_selection_pipeline(
        ["AAPL", "NVDA", "MSFT"],
        long_limit=3,
        daily_limit=3,
        ai_db_path=":memory:",
        save=False,
    )

    assert set(results) == {sp.LONG_TERM, sp.DAILY_DECISION}
    assert results[sp.LONG_TERM].selected_size == 3
    assert 1 <= results[sp.DAILY_DECISION].selected_size <= 3
    assert results[sp.DAILY_DECISION].items[0].status in {"ENTRY_READY", "WAIT_TRIGGER", "WATCH", "BENCH"}


def test_decision_pool_writes_change_report(monkeypatch, tmp_path: Path):
    import trader.selection_pools as sp

    report_path = tmp_path / "decision_report.json"
    pool_path = tmp_path / "selection_pools.json"
    monkeypatch.setattr(sp, "_load_bars", lambda _symbol, _tf: _bars(step=0.6))
    monkeypatch.setattr(sp, "_load_ai_scores", lambda _path: {"NVDA": 82.0, "AAPL": 65.0, "MSFT": 64.0})
    monkeypatch.setattr(sp, "save_daily_candidates", lambda _rows: None)
    previous = sp.PoolResult(
        layer=sp.DAILY_DECISION,
        updated_at="old",
        source_size=1,
        selected_size=1,
        items=[
            sp.PoolItem("OLD", 1, 70.0, "ENTRY_READY", "高", sp.DAILY_DECISION),
        ],
    )
    sp.save_selection_pools({sp.DAILY_DECISION: previous}, pool_path)

    result = sp.build_daily_decision_pool(
        ["AAPL", "NVDA", "MSFT"],
        limit=3,
        ai_db_path=":memory:",
        sync_daily_store=True,
        progress_callback=None,
    )
    report = sp._build_decision_report(["OLD"], result.items, result.items, "now")
    sp.save_decision_pool_report(report, report_path)
    loaded = sp.load_decision_pool_report(report_path)

    assert 1 <= result.selected_size <= 3
    assert loaded.current_symbols
    assert loaded.added
    assert loaded.removed[0].symbol == "OLD"


def test_aggressive_decision_style_expands_candidate_count(monkeypatch):
    import trader.selection_pools as sp

    symbols = ["AAPL", "MSFT", "NVDA", "AMAT", "LRCX", "PANW", "QCOM"]
    monkeypatch.setattr(sp, "_load_bars", lambda _symbol, _tf: _bars(step=0.7))
    monkeypatch.setattr(sp, "_load_ai_scores", lambda _path: {symbol: 70.0 for symbol in symbols})
    monkeypatch.setattr(sp, "save_daily_candidates", lambda _rows: None)
    monkeypatch.setattr(sp, "save_decision_pool_report", lambda _report: None)
    monkeypatch.setattr(
        sp,
        "load_selection_pool",
        lambda layer, path=sp._STORE: sp.PoolResult(layer=layer, updated_at="", source_size=0, selected_size=0, items=[]),
    )

    result = sp.build_daily_decision_pool(
        symbols,
        limit=7,
        decision_style=sp.DECISION_STYLE_AGGRESSIVE,
        ai_db_path=":memory:",
    )

    assert result.selected_size >= 5
    assert any("风格 aggressive" in item.reasons for item in result.items)


def test_source_universe_prefers_latest_market_scan(monkeypatch):
    import trader.market_scan as ms
    import trader.selection_pools as sp

    monkeypatch.setattr(ms, "market_scan_symbols", lambda **_kwargs: ["NVDA", "AAPL"])

    assert sp.build_source_universe(None, max_symbols=10) == ["NVDA", "AAPL"]
    assert sp.build_source_universe("MSFT, TSLA", max_symbols=10) == ["MSFT", "TSLA"]


def test_selection_pools_save_load_and_decision_symbols(tmp_path: Path):
    import trader.selection_pools as sp

    path = tmp_path / "selection_pools.json"
    result = sp.PoolResult(
        layer=sp.DAILY_DECISION,
        updated_at="now",
        source_size=2,
        selected_size=2,
        items=[
            sp.PoolItem(
                symbol="NVDA",
                rank=1,
                score=78.0,
                status="ENTRY_READY",
                data_confidence="高",
                layer=sp.DAILY_DECISION,
            ),
            sp.PoolItem(
                symbol="SPY",
                rank=2,
                score=65.0,
                status="MARKET_ANCHOR",
                data_confidence="中",
                layer=sp.DAILY_DECISION,
            ),
        ],
    )

    sp.save_selection_pools({sp.DAILY_DECISION: result}, path)

    loaded = sp.load_selection_pool(sp.DAILY_DECISION, path)
    assert loaded.items[0].symbol == "NVDA"
    assert sp.decision_symbols(path=path) == ["NVDA"]
