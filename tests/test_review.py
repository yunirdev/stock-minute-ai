"""SimpleReviewer 要把"账本文件不存在"和"账本文件存在但打不开/损坏/被锁"
区分开——前者是合法的"还没有数据"，后者才是需要如实报出来的真故障。"""
from datetime import datetime, timezone

import duckdb
import pytest

from trader.review import SimpleReviewer


def test_missing_db_file_returns_an_honest_empty_report_not_an_exception(tmp_path):
    db_path = tmp_path / "never-written.duckdb"
    assert not db_path.exists()

    report = SimpleReviewer(db_path=str(db_path)).review(
        period="daily", as_of=datetime(2026, 8, 5, tzinfo=timezone.utc)
    )

    assert report.portfolio_pnl == 0.0
    assert report.trades == []
    assert report.attribution["trade_count"] == 0
    # 摘要要能看出"还没有数据"，不能长得跟一个验证过的零成交平淡日一样。
    assert "尚无" in report.market_summary or "无账本" in report.market_summary


def test_existing_but_unreadable_db_still_raises(tmp_path):
    # 文件确实存在，但内容不是合法的 DuckDB 文件——这种要原样抛出去，不能
    # 被当成"还没有数据"静默吞掉，否则一个真正的损坏/故障会被伪装成平淡日。
    db_path = tmp_path / "corrupt.duckdb"
    db_path.write_bytes(b"not a real duckdb file")

    with pytest.raises(duckdb.Error):
        SimpleReviewer(db_path=str(db_path)).review(
            period="daily", as_of=datetime(2026, 8, 5, tzinfo=timezone.utc)
        )


def test_existing_db_with_real_data_is_unaffected(tmp_path):
    db_path = tmp_path / "real.duckdb"
    conn = duckdb.connect(str(db_path))
    conn.execute(
        "CREATE TABLE fills (symbol TEXT, side TEXT, filled_qty DOUBLE, "
        "avg_price DOUBLE, fill_time TIMESTAMPTZ)"
    )
    conn.execute(
        "CREATE TABLE equity_snapshots (total_equity DOUBLE, ts TIMESTAMPTZ)"
    )
    as_of = datetime(2026, 8, 5, 20, 0, tzinfo=timezone.utc)
    conn.execute(
        "INSERT INTO fills VALUES ('AAPL','BUY',10,100.0,?)",
        [datetime(2026, 8, 5, 15, 0, tzinfo=timezone.utc)],
    )
    conn.execute(
        "INSERT INTO equity_snapshots VALUES (10000.0, ?), (10050.0, ?)",
        [
            datetime(2026, 8, 5, 14, 0, tzinfo=timezone.utc),
            datetime(2026, 8, 5, 19, 0, tzinfo=timezone.utc),
        ],
    )
    conn.close()

    report = SimpleReviewer(db_path=str(db_path)).review(
        period="daily", as_of=as_of
    )

    assert report.attribution["trade_count"] == 1
    assert report.portfolio_pnl == 50.0
