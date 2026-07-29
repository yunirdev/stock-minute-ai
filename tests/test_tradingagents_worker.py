from pathlib import Path

from trader.tradingagents_worker import (
    _emit,
    _exception_diagnostic,
    _prepare_runtime_paths,
)


def test_worker_result_transport_is_ascii_safe(capsys):
    _emit({"ok": True, "decision": "中文"})

    output = capsys.readouterr().out

    assert "\\u4e2d\\u6587" in output
    assert "中文" not in output


def test_worker_prepares_tradingagents_and_yfinance_paths(
    monkeypatch,
    tmp_path,
):
    configured = {}

    class FakeYFinance:
        @staticmethod
        def set_tz_cache_location(path):
            configured["yfinance"] = path

    monkeypatch.setitem(__import__("sys").modules, "yfinance", FakeYFinance)
    config = {
        "data_cache_dir": str(tmp_path / "cache"),
        "results_dir": str(tmp_path / "results"),
        "memory_log_path": str(tmp_path / "memory" / "log.md"),
    }

    _prepare_runtime_paths(config)

    assert Path(config["data_cache_dir"]).is_dir()
    assert Path(config["results_dir"]).is_dir()
    assert Path(config["memory_log_path"]).parent.is_dir()
    assert configured["yfinance"] == str(
        (tmp_path / "cache" / "yfinance").resolve()
    )


def test_worker_failure_diagnostics_are_stable_and_sanitized():
    assert (
        _exception_diagnostic(
            ConnectionError("secret endpoint could not connect")
        )
        == "LLM_API_CONNECTION_UNAVAILABLE"
    )
    assert (
        _exception_diagnostic(
            OSError("unable to open database file")
        )
        == "CACHE_DATABASE_UNAVAILABLE"
    )
    assert (
        _exception_diagnostic(
            ImportError(
                "DLL load failed: Application Control blocked user path"
            )
        )
        == "DEPENDENCY_LOAD_BLOCKED"
    )
    assert (
        _exception_diagnostic(
            LookupError(
                "Error code: 404 - model 'missing:tag' not found; "
                "type=not_found_error"
            )
        )
        == "MODEL_NOT_FOUND"
    )
