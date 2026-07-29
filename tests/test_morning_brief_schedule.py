from datetime import datetime, timezone

from trader import config as config_module
from trader.morning_brief import should_send_brief


def test_should_send_brief_respects_configured_et_hour(monkeypatch):
    monkeypatch.setattr(config_module.settings, "morning_brief_hour_et", 9)
    # 9am ET in winter (EST, UTC-5) is 14:00 UTC.
    at_hour = datetime(2026, 1, 15, 14, 5, tzinfo=timezone.utc)
    before_hour = datetime(2026, 1, 15, 13, 5, tzinfo=timezone.utc)
    assert should_send_brief(at_hour, last_date=None) is True
    assert should_send_brief(before_hour, last_date=None) is False


def test_should_send_brief_follows_updated_configured_hour(monkeypatch):
    monkeypatch.setattr(config_module.settings, "morning_brief_hour_et", 7)
    # 7am ET in winter (EST, UTC-5) is 12:00 UTC.
    at_hour = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
    assert should_send_brief(at_hour, last_date=None) is True


def test_should_send_brief_does_not_repeat_same_day():
    at_hour = datetime(2026, 1, 15, 14, 5, tzinfo=timezone.utc)
    assert should_send_brief(at_hour, last_date="2026-01-15") is False
