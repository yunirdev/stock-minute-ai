from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _bars(start: float = 100.0, step: float = 0.5, rows: int = 180, volume: int = 2_000_000) -> pd.DataFrame:
    close = [start + i * step for i in range(rows)]
    return pd.DataFrame({"close": close, "volume": [volume] * rows})


def test_symbol_master_parses_free_listing_sources(tmp_path: Path):
    import trader.symbol_master as sm

    nasdaq_text = "\n".join([
        "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares",
        "AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N",
        "QQQ|Invesco QQQ Trust, Series 1|G|N|N|100|Y|N",
        "TST|Test Company Common Stock|S|Y|N|100|N|N",
        "ABCDW|ABCD Warrants|G|N|N|100|N|N",
        "File Creation Time: 06182026",
    ])
    other_text = "\n".join([
        "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol",
        "BRK.B|Berkshire Hathaway Inc. Class B Common Stock|N|BRK.B|N|100|N|BRK/B",
        "SPY|SPDR S&P 500 ETF Trust|P|SPY|Y|100|N|SPY",
        "File Creation Time: 06182026",
    ])
    sec_text = json.dumps({
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        "1": {"cik_str": 1067983, "ticker": "BRK-B", "title": "Berkshire Hathaway Inc."},
    })
    payloads = {
        sm.NASDAQ_LISTED_URL: nasdaq_text,
        sm.OTHER_LISTED_URL: other_text,
        sm.SEC_COMPANY_TICKERS_URL: sec_text,
    }

    master = sm.update_symbol_master(fetcher=lambda url: payloads[url], path=tmp_path / "symbols.json")
    symbols = sm.common_equity_symbols(master)

    assert {"AAPL", "BRK.B"}.issubset(symbols)
    assert "QQQ" not in symbols
    assert "SPY" not in symbols
    assert "TST" not in symbols
    assert "ABCDW" not in symbols
    assert all(status.ok for status in master.source_status)


def test_symbol_master_does_not_fallback_when_listing_sources_fail(tmp_path: Path):
    import trader.symbol_master as sm

    master = sm.update_symbol_master(
        fetcher=lambda _url: (_ for _ in ()).throw(OSError("network blocked")),
        path=tmp_path / "symbols.json",
    )

    assert master.symbols == []
    assert all(not status.ok for status in master.source_status)
    assert not any("fallback" in status.source or "cache" in status.source for status in master.source_status)


def test_index_universe_fetches_core_components(tmp_path: Path):
    import trader.index_universe as iu

    nasdaq_rows = "".join(f"<tr><td>N{i:02d}</td></tr>" for i in range(60))
    payloads = {
        iu.SP500_URL: "<table><tr><th>Symbol</th></tr><tr><td>AAPL</td></tr><tr><td>MSFT</td></tr></table>",
        iu.DOW_URL: "<table><tr><th>Symbol</th></tr>"
        + "".join(f"<tr><td>D{i:02d}</td></tr>" for i in range(30))
        + "</table>",
        iu.NASDAQ100_URL: f"<table><tr><th>Ticker</th></tr>{nasdaq_rows}</table>",
    }

    snapshot = iu.update_index_universe(fetcher=lambda url: payloads[url], path=tmp_path / "indexes.json")

    assert "AAPL" in snapshot.sp500
    assert "D00" in snapshot.dow
    assert "N00" in snapshot.nasdaq100
    assert {"AAPL", "D00", "N00"}.issubset(snapshot.core_symbols)
    assert all(status.ok for status in snapshot.source_status)


def test_index_universe_does_not_fallback_when_sources_fail(tmp_path: Path):
    import trader.index_universe as iu

    snapshot = iu.update_index_universe(
        fetcher=lambda _url: (_ for _ in ()).throw(OSError("network blocked")),
        path=tmp_path / "indexes.json",
    )

    assert snapshot.core_symbols == []
    assert all(not status.ok for status in snapshot.source_status)
    assert not any("fallback" in status.source or "cache" in status.source for status in snapshot.source_status)


def test_first_round_universe_merges_indexes_hot_and_broad(monkeypatch):
    import trader.market_scan as ms
    from trader.hot_universe import HotSymbol, HotUniverse
    from trader.index_universe import IndexUniverse
    from trader.symbol_master import SourceStatus, SymbolMaster, SymbolRecord

    master = SymbolMaster(
        updated_at="now",
        symbols=[
            SymbolRecord(symbol="AAPL", name="Apple Common Stock"),
            SymbolRecord(symbol="MSFT", name="Microsoft Common Stock"),
            SymbolRecord(symbol="LOWQ", name="Low Quality Common Stock"),
        ],
        source_status=[SourceStatus("symbols", True, 3, updated_at="now")],
    )
    indexes = IndexUniverse(
        updated_at="now",
        dow=["AAPL"],
        sp500=["MSFT"],
        nasdaq100=["NVDA"],
        source_status=[SourceStatus("indexes", True, 3, updated_at="now")],
    )
    hot = HotUniverse(
        updated_at="now",
        symbols=[HotSymbol("PLTR", 40.0, sources=["retail_seed"])],
        source_status=[SourceStatus("hot", True, 1, updated_at="now")],
    )

    monkeypatch.setattr(ms, "load_symbol_master", lambda: master)
    monkeypatch.setattr(ms, "load_index_universe", lambda: indexes)
    monkeypatch.setattr(ms, "load_hot_universe", lambda: hot)

    symbols, statuses, tags = ms.build_first_round_universe(max_symbols=20)

    assert symbols[:3] == ["AAPL", "MSFT", "NVDA"]
    assert "PLTR" in symbols
    assert "LOWQ" in symbols
    assert tags["AAPL"] == ["core_index", "broad_market"]
    assert tags["PLTR"] == ["hot"]
    assert [status.source for status in statuses] == ["symbols", "indexes", "hot"]


def test_hot_universe_has_no_hardcoded_retail_seed(monkeypatch):
    import trader.hot_universe as hu
    from trader.symbol_master import SourceStatus

    def fake_rss(_urls, _allowed, statuses):
        statuses.append(SourceStatus("rss_news", True, 0, updated_at="now"))
        return {}

    monkeypatch.setattr(hu, "_rss_hot_symbols", fake_rss)

    snapshot = hu.build_hot_universe(base_symbols=["AAPL", "NVDA"], save=False)

    assert snapshot.symbols == []
    assert not any(status.source == "retail_seed" for status in snapshot.source_status)


def test_market_scan_scores_and_keeps_auditable_rejects(monkeypatch, tmp_path: Path):
    import trader.market_scan as ms
    from trader.symbol_master import SourceStatus

    statuses = [SourceStatus(source="test_source", ok=True, count=4, updated_at="now")]
    tags = {
        "AAPL": ["core_index"],
        "HOTX": ["hot"],
        "LOWP": ["broad_market"],
        "MISS": ["broad_market"],
    }

    monkeypatch.setattr(
        ms,
        "build_first_round_universe",
        lambda **_kwargs: (["AAPL", "HOTX", "LOWP", "MISS"], statuses, tags),
    )
    monkeypatch.setattr(
        ms,
        "_load_bars",
        lambda symbol, _tf: {
            "AAPL": _bars(start=100, step=0.7, volume=3_000_000),
            "HOTX": _bars(start=30, step=0.4, volume=2_500_000),
            "LOWP": _bars(start=1, step=0.01, volume=1_000_000),
        }.get(symbol, pd.DataFrame()),
    )

    path = tmp_path / "scan.json"
    report = ms.run_market_scan(
        max_symbols=10,
        max_downloads=0,
        selected_limit=10,
        require_fresh_bars=False,
        save=True,
        path=path,
    )
    by_symbol = {item.symbol: item for item in report.items}

    assert report.scanned_size == 4
    assert report.rejected_size == 2
    assert by_symbol["AAPL"].status in {ms.KEEP, ms.WATCH}
    assert by_symbol["HOTX"].status in {ms.KEEP, ms.HOT}
    assert report.reject_summary["price_below_min"] == 1
    assert report.reject_summary["missing_bars"] == 1
    assert ms.load_market_scan_report(path).items[0].symbol == report.items[0].symbol
