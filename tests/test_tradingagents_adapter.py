import pytest

from trader.daily_research import TradingAgentsAdapter, _confidence, _recommendation


def test_tradingagents_adapter_fails_explicitly_when_module_is_missing(monkeypatch):
    import trader.daily_research as module

    monkeypatch.delenv("TRADINGAGENTS_PYTHON", raising=False)
    monkeypatch.delenv("TRADINGAGENTS_PROJECT_DIR", raising=False)
    real_import = module.importlib.import_module

    def missing(name: str):
        if name.startswith("tradingagents"):
            raise ModuleNotFoundError(name)
        return real_import(name)

    monkeypatch.setattr(module.importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match="TRADINGAGENTS_MODULE_UNAVAILABLE"):
        TradingAgentsAdapter().analyze("AAPL", "2026-07-27")


def test_recommendation_prefers_explicit_or_leading_final_action():
    assert _recommendation("BUY. Sell only if the stop breaks.") == "BUY"
    assert _recommendation("Risks mention BUY. FINAL: SELL") == "SELL"
    assert _recommendation({"recommendation": "HOLD"}) == "HOLD"
    assert _recommendation("Overweight") == "BUY"
    assert _recommendation("Underweight") == "SELL"
    assert _confidence("Overweight") == pytest.approx(0.7)
    assert _confidence("Buy") == pytest.approx(0.85)


def test_tradingagents_adapter_uses_dedicated_python(monkeypatch, tmp_path):
    import trader.daily_research as module

    python_path = tmp_path / "python.exe"
    python_path.touch()
    captured = {}

    def run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        stdout = (
            "third-party progress\n"
            '__TRADINGAGENTS_RESULT__={"ok":true,"decision":'
            '{"recommendation":"BUY","confidence":0.8},'
            '"state":{"market_report":"positive"}}\n'
        )
        return module.subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr(module.subprocess, "run", run)
    adapter = TradingAgentsAdapter(
        python_executable=str(python_path),
        llm_provider="ollama",
        deep_model="qwen2.5:14b",
        quick_model="qwen2.5:14b",
        backend_url="http://127.0.0.1:11434/v1",
        cache_dir=str(tmp_path / "cache"),
        results_dir=str(tmp_path / "results"),
        memory_log_path=str(tmp_path / "memory" / "log.md"),
    )

    result = adapter.analyze("AAPL", "2026-07-27")

    assert result.recommendation == "BUY"
    assert result.confidence == pytest.approx(0.8)
    assert result.score == pytest.approx(90.0)
    assert captured["command"][0] == str(python_path.resolve())
    payload = module.json.loads(captured["kwargs"]["input"])
    assert payload["llm_provider"] == "ollama"
    assert payload["backend_url"].endswith("/v1")
    assert payload["cache_dir"] == str(tmp_path / "cache")
    assert payload["results_dir"] == str(tmp_path / "results")
    assert payload["memory_log_path"] == str(tmp_path / "memory" / "log.md")


def test_tradingagents_adapter_reports_worker_failure(monkeypatch, tmp_path):
    import trader.daily_research as module

    python_path = tmp_path / "python.exe"
    python_path.touch()

    def run(command, **kwargs):
        stdout = (
            '__TRADINGAGENTS_RESULT__={"ok":false,'
            '"error":"ConnectionError",'
            '"diagnostic_code":"LLM_API_CONNECTION_UNAVAILABLE",'
            '"message":"offline"}\n'
        )
        return module.subprocess.CompletedProcess(command, 1, stdout, "")

    monkeypatch.setattr(module.subprocess, "run", run)
    adapter = TradingAgentsAdapter(python_executable=str(python_path))

    with pytest.raises(
        RuntimeError,
        match=(
            "TRADINGAGENTS_WORKER_FAILED:ConnectionError:"
            "LLM_API_CONNECTION_UNAVAILABLE"
        ),
    ):
        adapter.analyze("AAPL", "2026-07-27")
