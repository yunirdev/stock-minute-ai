from __future__ import annotations


def test_account_risk_module_surfaces_concentration_and_cash_buffer():
    from trader.account_risk import build_account_risk_lines

    lines = build_account_risk_lines(
        positions=[
            {"symbol": "AAPL", "market_value": 60000},
            {"symbol": "NVDA", "market_value": 15000},
        ],
        equity=100000,
        cash_pct=18.0,
        movers=[{"symbol": "AAPL", "pct": -1.2}],
    )
    text = "\n".join(lines)

    assert "最大暴露：AAPL 60.0%" in text
    assert "账户动作" in text
    assert "盘前异动持仓：AAPL -1.20%" in text
    assert "现金缓冲 18.0% 偏低" in text
