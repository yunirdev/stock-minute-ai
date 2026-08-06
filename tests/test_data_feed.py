from __future__ import annotations


def test_bars_per_trading_day_matches_old_hardcoded_5m_constant():
    from trader.data_feed import _bars_per_trading_day

    # 原来的 lookback_days 是 n_bars // 78 算的，78 = 390(RTH分钟)/5 —— 换成
    # 动态计算后，5m 这个最常用的档位算出来的常数必须原样保留，不能悄悄变。
    assert _bars_per_trading_day("5m") == 78


def test_bars_per_trading_day_scales_with_longer_timeframes():
    from trader.data_feed import _bars_per_trading_day

    assert _bars_per_trading_day("30m") == 13
    assert _bars_per_trading_day("1h") == 6
    # 未收录的周期（比如 "1d"）退化成 1 根/天，回看天数约等于要求的根数，
    # 不会被错误地按 5 分钟档位的密度去算。
    assert _bars_per_trading_day("1d") == 1
    assert _bars_per_trading_day("unknown") == 1


def test_fetch_bars_requests_wider_window_for_longer_timeframes(monkeypatch):
    from trader.config import TradingConfig
    from trader.data_feed import AlpacaDataFeed

    captured = {}

    class Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"bars": []}

    def fake_get(url, headers=None, params=None, timeout=None):
        captured["start"] = params["start"]
        captured["end"] = params["end"]
        return Resp()

    monkeypatch.setattr("trader.data_feed.requests.get", fake_get)

    cfg = TradingConfig(
        symbols=["XLF"], timeframe="30m",
        alpaca_api_key="key", alpaca_secret_key="secret",
    )
    AlpacaDataFeed(cfg).fetch_bars("XLF", n_bars=120)

    from datetime import datetime, timezone
    start = datetime.strptime(captured["start"], "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )
    end = datetime.strptime(captured["end"], "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )
    # 120 // 13 + 3 = 12 天——比旧的按 5 分钟档位算出来的 4 天宽得多，30 分钟
    # 周期下才有机会真正凑够 120 根 bar。
    assert (end - start).days >= 12


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
