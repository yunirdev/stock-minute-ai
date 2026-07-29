import pytest

from trader import config_editor


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text("EXISTING_KEY=keep_me\n", encoding="utf-8")
    monkeypatch.setattr(config_editor, "_ENV_PATH", path)
    return path


def test_writable_key_is_persisted_without_disturbing_other_lines(env_file):
    config_editor.write_env_setting("OLLAMA_MODEL", "qwen2.5:14b")
    content = env_file.read_text(encoding="utf-8")
    assert "EXISTING_KEY=keep_me" in content
    assert "OLLAMA_MODEL='qwen2.5:14b'" in content


def test_writable_key_is_case_insensitive_and_normalized(env_file):
    config_editor.write_env_setting("ollama_model", "llama3.2")
    content = env_file.read_text(encoding="utf-8")
    assert "OLLAMA_MODEL" in content


@pytest.mark.parametrize(
    "forbidden_key",
    [
        "BROKER_TYPE",
        "AUTO_TRADE_PAPER",
        "ALPACA_API_KEY",
        "ALPACA_API_SECRET",
        "MAX_POSITION_PCT",
        "not_in_whitelist",
    ],
)
def test_non_whitelisted_keys_are_rejected(env_file, forbidden_key):
    with pytest.raises(config_editor.EnvKeyNotWritableError):
        config_editor.write_env_setting(forbidden_key, "x")
    assert env_file.read_text(encoding="utf-8") == "EXISTING_KEY=keep_me\n"


@pytest.mark.parametrize("empty_value", [None, "", "   "])
def test_none_or_empty_value_is_rejected_not_written_as_literal_none(env_file, empty_value):
    # A cleared NiceGUI number/select field surfaces as None; without this
    # guard it would previously write the literal string "None" into .env
    # and break pydantic-settings parsing on next app start.
    with pytest.raises(config_editor.EnvValueRequiredError):
        config_editor.write_env_setting("MIN_AI_SCORE", empty_value)
    content = env_file.read_text(encoding="utf-8")
    assert "None" not in content
    assert content == "EXISTING_KEY=keep_me\n"
