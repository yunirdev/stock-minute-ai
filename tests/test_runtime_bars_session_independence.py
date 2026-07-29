"""_fetch_and_cache_bars must run for monitoring/display purposes regardless
of market session — only the trading-decision pipeline (selection onward)
is gated to session == "open". See runtime.py _tick() step 7/8 ordering.
"""
import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from trader.runtime import Runtime

RUNTIME_SRC = Path(__file__).resolve().parents[1] / "trader" / "runtime.py"


def _raw_bar(ts: datetime, close: float = 100.0):
    return SimpleNamespace(
        timestamp=ts, open=close, high=close, low=close, close=close, volume=1000.0
    )


def _runtime(symbols, feed, *, timeframe="5m", bars_lookback=60) -> Runtime:
    runtime = Runtime.__new__(Runtime)
    runtime._cfg = SimpleNamespace(
        symbols=symbols, timeframe=timeframe, bars_lookback=bars_lookback
    )
    runtime._feed = feed
    return runtime


def test_fetch_and_cache_bars_ignores_market_session(monkeypatch):
    """The method itself takes no session argument — it always fetches."""
    now = datetime(2026, 7, 28, 9, 0, tzinfo=timezone.utc)  # pre-market ET
    raw_bars = [_raw_bar(now - timedelta(minutes=5 * i)) for i in range(35)][::-1]
    feed = SimpleNamespace(fetch_bars=lambda symbol, n_bars: raw_bars)

    upserts = []
    monkeypatch.setattr(
        "trader.runtime._dc_upsert",
        lambda symbol, timeframe, df: upserts.append((symbol, timeframe, len(df))),
    )

    runtime = _runtime(["AAPL"], feed)
    raw_map, model_bars = runtime._fetch_and_cache_bars(now)

    assert "AAPL" in raw_map
    assert "AAPL" in model_bars
    assert upserts == [("AAPL", "5m", 35)]


def test_fetch_and_cache_bars_skips_symbol_with_too_few_bars(monkeypatch):
    now = datetime(2026, 7, 28, 9, 0, tzinfo=timezone.utc)
    feed = SimpleNamespace(fetch_bars=lambda symbol, n_bars: [_raw_bar(now)])
    monkeypatch.setattr("trader.runtime._dc_upsert", lambda *a, **k: None)

    runtime = _runtime(["AAPL"], feed)
    raw_map, model_bars = runtime._fetch_and_cache_bars(now)

    assert raw_map == {}
    assert model_bars == {}


def test_fetch_and_cache_bars_skips_stale_latest_bar(monkeypatch):
    now = datetime(2026, 7, 28, 9, 0, tzinfo=timezone.utc)
    stale_ts = now - timedelta(hours=6)
    raw_bars = [_raw_bar(stale_ts - timedelta(minutes=5 * i)) for i in range(35)][::-1]
    feed = SimpleNamespace(fetch_bars=lambda symbol, n_bars: raw_bars)
    monkeypatch.setattr("trader.runtime._dc_upsert", lambda *a, **k: None)

    runtime = _runtime(["AAPL"], feed)
    raw_map, model_bars = runtime._fetch_and_cache_bars(now)

    assert raw_map == {}
    assert model_bars == {}


def test_fetch_and_cache_bars_tolerates_one_symbol_failing(monkeypatch):
    now = datetime(2026, 7, 28, 9, 0, tzinfo=timezone.utc)
    good_bars = [_raw_bar(now - timedelta(minutes=5 * i)) for i in range(35)][::-1]

    def fetch_bars(symbol, n_bars):
        if symbol == "BROKEN":
            raise RuntimeError("feed unavailable")
        return good_bars

    feed = SimpleNamespace(fetch_bars=fetch_bars)
    monkeypatch.setattr("trader.runtime._dc_upsert", lambda *a, **k: None)

    runtime = _runtime(["BROKEN", "AAPL"], feed)
    raw_map, model_bars = runtime._fetch_and_cache_bars(now)

    assert set(raw_map) == {"AAPL"}
    assert set(model_bars) == {"AAPL"}


def test_tick_fetches_bars_before_checking_session_is_open():
    """Source-level guard: the open-only gate must come after the bar fetch,
    not before it, so pre/post-market ticks still refresh the local cache.
    """
    source = RUNTIME_SRC.read_text(encoding="utf-8")
    tree = ast.parse(source)
    tick_fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_tick"
    )
    call_order = []
    for node in ast.walk(tick_fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "_fetch_and_cache_bars":
                call_order.append(("fetch_bars", node.lineno))
        if isinstance(node, ast.Compare):
            for comparator in node.comparators:
                if (
                    isinstance(comparator, ast.Constant)
                    and comparator.value == "open"
                ):
                    call_order.append(("open_check", node.lineno))
    call_order.sort(key=lambda item: item[1])
    kinds = [kind for kind, _ in call_order]
    assert "fetch_bars" in kinds
    assert "open_check" in kinds
    assert kinds.index("fetch_bars") < kinds.index("open_check")
