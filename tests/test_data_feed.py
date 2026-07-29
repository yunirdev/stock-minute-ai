from __future__ import annotations


def test_alpaca_fetch_bars_handles_null_bars(monkeypatch):
    from trader.config import TradingConfig
    from trader.data_feed import AlpacaDataFeed

    class Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"bars": None}

    monkeypatch.setattr("trader.data_feed.requests.get", lambda *args, **kwargs: Resp())

    cfg = TradingConfig(
        symbols=["AAPL"],
        timeframe="5m",
        alpaca_api_key="key",
        alpaca_secret_key="secret",
        alpaca_feed="iex",
    )

    assert AlpacaDataFeed(cfg).fetch_bars("AAPL", n_bars=40) == []


def test_list_cached_names_does_not_read_parquet(monkeypatch, tmp_path):
    from trader import data_cache

    (tmp_path / "QQQ_5m.parquet").write_bytes(b"not parquet")
    (tmp_path / "AAPL_1d.parquet").write_bytes(b"not parquet")
    monkeypatch.setattr(data_cache, "_BARS_DIR", tmp_path)
    monkeypatch.setattr(
        data_cache,
        "_read_parquet",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("read")),
    )

    assert data_cache.list_cached_names() == ["AAPL_1d.parquet", "QQQ_5m.parquet"]
