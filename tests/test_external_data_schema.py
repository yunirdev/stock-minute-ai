"""Schema contract tests for third-party data sources.

The most damaging bug class found in this codebase is NOT a swallowed
exception — it is `dict.get("wrong_key", default)` / a missing DataFrame
column, which raises nothing and silently yields a default value. Four
separate agents were feeding hardcoded zeros into the scoring pipeline this
way, for an unknown length of time, with no error anywhere.

These tests pin the exact upstream field names the code depends on, using
offline fixtures shaped like the real payloads. If a provider renames a
field (or someone "cleans up" a key name), a test fails instead of the
score quietly collapsing to a constant.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import patch

import pandas as pd
import pytest


# ═══════════════════════════════════════════════════════════════════════════
# yfinance: institutional holders
# ═══════════════════════════════════════════════════════════════════════════


def _institutional_frame() -> pd.DataFrame:
    """Mirrors yfinance 1.2.x `Ticker.institutional_holders`.

    Note the column is `pctHeld` — NOT `% Out`, which is what the code used
    to read, silently producing 0.0% for every holder.
    """
    return pd.DataFrame([
        {"Date Reported": datetime(2026, 3, 31), "Holder": "Blackrock Inc.",
         "pctHeld": 0.0779, "Shares": 1_100_000_000, "Value": 300_000_000_000,
         "pctChange": 0.01},
        {"Date Reported": datetime(2026, 3, 31), "Holder": "Vanguard Group Inc",
         "pctHeld": 0.0649, "Shares": 900_000_000, "Value": 250_000_000_000,
         "pctChange": -0.01},
    ])


def test_institutional_holders_reads_pctheld_not_percent_out():
    from trader.ai.agents.elite_holdings import _yf_institutional

    with patch("yfinance.Ticker") as ticker:
        ticker.return_value.institutional_holders = _institutional_frame()
        signal, detail = _yf_institutional("AAPL")

    pcts = [h["pct_out"] for h in detail["top_holders"]]
    assert pcts == [7.79, 6.49]      # real percentages, not zeros
    assert signal > 0                # and they actually move the score


def test_institutional_holders_missing_pctheld_column_is_reported_not_zeroed():
    """If yfinance renames the column again, we must surface that rather than
    quietly scoring every holder at 0%."""
    from trader.ai.agents.elite_holdings import _yf_institutional

    frame = _institutional_frame().rename(columns={"pctHeld": "somethingElse"})
    with patch("yfinance.Ticker") as ticker:
        ticker.return_value.institutional_holders = frame
        signal, detail = _yf_institutional("AAPL")

    assert signal == 0.0
    assert detail == {}              # explicitly empty, not fabricated zeros


# ═══════════════════════════════════════════════════════════════════════════
# yfinance: insider transactions
# ═══════════════════════════════════════════════════════════════════════════


def _insider_frame() -> pd.DataFrame:
    """Mirrors yfinance 1.2.x `Ticker.insider_transactions`.

    The `Transaction` column is present but entirely blank; the direction
    actually lives in `Text`. Reading only `Transaction` made every symbol
    look like it had zero insider buying AND zero insider selling.
    """
    recent = datetime.now() - timedelta(days=10)
    return pd.DataFrame([
        {"Start Date": recent, "Transaction": "", "Shares": 200_000,
         "Text": "Sale at price 295.14 per share.", "Insider": "A", "Position": "CEO"},
        {"Start Date": recent, "Transaction": "", "Shares": 101_390,
         "Text": "Sale at price 290.00 per share.", "Insider": "B", "Position": "CFO"},
        {"Start Date": recent, "Transaction": "", "Shares": 50_000,
         "Text": "Purchase at price 280.00 per share.", "Insider": "C", "Position": "Dir"},
    ])


def test_insider_direction_is_parsed_from_text_when_transaction_is_blank():
    from trader.ai.agents.elite_holdings import _yf_insider

    with patch("yfinance.Ticker") as ticker:
        ticker.return_value.insider_transactions = _insider_frame()
        _signal, detail = _yf_insider("AAPL")

    assert detail["sell_val"] == 301_390          # 200,000 + 101,390
    assert detail["buy_val"] == 50_000
    assert detail["net_direction"] == "sell"


def test_insider_with_no_usable_direction_column_reports_rather_than_neutral():
    from trader.ai.agents.elite_holdings import _yf_insider

    frame = _insider_frame().drop(columns=["Text"])
    with patch("yfinance.Ticker") as ticker:
        ticker.return_value.insider_transactions = frame
        signal, detail = _yf_insider("AAPL")

    assert signal == 0.0
    assert "buy_val" not in detail   # no fabricated 0/0 "neutral" reading


# ═══════════════════════════════════════════════════════════════════════════
# arkfunds.io: holdings weight
# ═══════════════════════════════════════════════════════════════════════════


def test_ark_holding_weight_field_is_named_weight():
    """arkfunds.io v2 returns `weight`. The code previously looked for
    `weight_pct` / `market_value_weight`, neither of which exists, so every
    tracked ticker had weight 0.0."""
    import json
    from io import BytesIO

    from trader.ai.agents.elite_holdings import _fetch_all_ark

    payload = {
        "holdings": [
            {"ticker": "TSLA", "company": "Tesla", "shares": 100,
             "market_value": 1000.0, "weight": 9.41, "weight_rank": 1},
        ]
    }

    def _fake_urlopen(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        body = json.dumps(payload if "holdings" in url else {"trades": []}).encode()
        resp = BytesIO(body)
        resp.__enter__ = lambda s=resp: s
        resp.__exit__ = lambda s, *a: None
        return resp

    with patch("urllib.request.urlopen", _fake_urlopen):
        result = _fetch_all_ark()

    assert result["TSLA"]["weight"] == pytest.approx(9.41)


# ═══════════════════════════════════════════════════════════════════════════
# quant: enough history for the 12-1 month momentum factor
# ═══════════════════════════════════════════════════════════════════════════


def test_quant_requests_enough_history_for_12m_momentum():
    """The 12-1 month factor carries 30% weight but needs >= 252 daily bars.
    yfinance's "1y" period returns ~251 — one short — so the factor never
    fired. The request period must exceed one year."""
    from trader.ai.agents.quant import QuantAgent
    from trader.models import AgentContext, Candidate

    requested = []

    def _spy(symbol, period="1y"):
        requested.append(period)
        return None

    ctx = AgentContext(
        candidates=[Candidate(symbol="AAPL", score=70, rank=1, reasons={})],
        plans=[], news=[], positions={}, equity=100_000.0,
        as_of=datetime.now(), extra={},
    )

    with patch("trader.ai.agents.quant._fetch_daily", _spy):
        QuantAgent().run(ctx)

    assert requested, "quant should request daily history"
    assert all(p != "1y" for p in requested), (
        f"'1y' yields ~251 bars, one short of the 252 the factor needs; got {requested}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# calendar_events: source-failure disclosure must actually be reachable
# ═══════════════════════════════════════════════════════════════════════════


def test_partial_source_failure_names_the_failing_source():
    """`has_partial_source_issue` is only true when economic_available is
    true, but warning_text used to return "" in exactly that case — so the
    pushed report could only ever say "未知原因"."""
    from trader.calendar_events import CalendarEventsResult, EventSourceStatus

    result = CalendarEventsResult(events=[], sources=[
        EventSourceStatus("nasdaq_economic", "ok"),
        EventSourceStatus("official_bls_calendar", "error", "HTTPError 403"),
    ])

    assert result.has_partial_source_issue is True
    assert "official_bls_calendar" in result.warning_text
    assert "403" in result.warning_text


def test_healthy_sources_produce_no_warning_text():
    from trader.calendar_events import CalendarEventsResult, EventSourceStatus

    result = CalendarEventsResult(events=[], sources=[
        EventSourceStatus("nasdaq_economic", "ok"),
        EventSourceStatus("official_bls_calendar", "ok"),
    ])

    assert result.warning_text == ""
    assert result.has_partial_source_issue is False


# ═══════════════════════════════════════════════════════════════════════════
# morning brief: maintained ratings must not render as upgrades
# ═══════════════════════════════════════════════════════════════════════════


def test_maintained_rating_is_not_shown_as_an_upgrade():
    """yfinance Action values are up/down/main/reit/init. `reit` (reiterate)
    and `init` (initiate coverage) were both drawn with an ⬆️ upgrade arrow,
    so "Buy → Buy" appeared under a "评级变动" heading as if it were news."""
    from trader import morning_brief as mb

    # GradeDate 是 DatetimeIndex，不是普通列 —— 函数按 df.index 过滤时间窗
    now = pd.Timestamp.utcnow()
    frame = pd.DataFrame(
        [
            {"Firm": "UBS", "ToGrade": "Buy", "FromGrade": "Buy", "Action": "reit"},
            {"Firm": "Citi", "ToGrade": "Buy", "FromGrade": "Hold", "Action": "up"},
        ],
        index=pd.DatetimeIndex([now, now], name="GradeDate"),
    )

    with patch("yfinance.Ticker") as ticker:
        ticker.return_value.upgrades_downgrades = frame
        text = mb._build_analyst_changes(["AAPL"])

    assert text, "两条评级记录都在 48h 窗口内，应该有输出"
    reit_line = next(line for line in text.splitlines() if "重申" in line)
    up_line = next(line for line in text.splitlines() if "上调" in line)
    assert "⬆️" not in reit_line     # 维持/重申不能画成升级箭头
    assert "⬆️" in up_line           # 真正的上调仍然是升级箭头
