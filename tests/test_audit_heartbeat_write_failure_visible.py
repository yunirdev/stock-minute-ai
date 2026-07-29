"""log_heartbeat()'s DuckDB write is best-effort (another process may hold
the lock), but a failure there defeats HeartbeatWatchdog's entire purpose:
the watchdog reads this table to detect a hung engine, so a silently
swallowed write failure looks identical to a genuinely stale heartbeat. It
used to log at DEBUG, which is off by default and let this fail every tick
for a full day, undetected, while the watchdog fired CRITICAL staleness
alerts nobody could explain. This test pins the failure to at least WARNING.
"""
import logging

from trader.audit import AuditLog
from trader.config import TradingConfig


def test_heartbeat_duckdb_write_failure_logs_at_warning(tmp_path, caplog, monkeypatch):
    config = TradingConfig(db_path=str(tmp_path / "trade.duckdb"))
    audit = AuditLog(config)

    def _boom():
        raise RuntimeError("simulated lock contention")

    monkeypatch.setattr(audit, "_connect", _boom)

    with caplog.at_level(logging.WARNING, logger="trader.audit"):
        audit.log_heartbeat(tick_count=1, equity=100_000.0)

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("heartbeat" in r.message.lower() for r in warnings)


def test_heartbeat_json_sidecar_still_written_when_duckdb_write_fails(tmp_path, monkeypatch):
    config = TradingConfig(db_path=str(tmp_path / "trade.duckdb"))
    audit = AuditLog(config)

    def _boom():
        raise RuntimeError("simulated lock contention")

    monkeypatch.setattr(audit, "_connect", _boom)

    from trader import audit as audit_module

    sidecar = tmp_path / "heartbeat.json"
    monkeypatch.setattr(audit_module, "_HEARTBEAT_FILE", sidecar)

    audit.log_heartbeat(tick_count=7, equity=42.0)

    assert sidecar.exists()
    assert '"tick_count": 7' in sidecar.read_text(encoding="utf-8")
