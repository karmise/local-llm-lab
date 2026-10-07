from pathlib import Path

import pytest

from llm_testkit.config import Settings
from test_support.fixtures.unit_config import (
    clean_settings_environment as clean_settings_environment,
)

pytestmark = pytest.mark.unit


def test_defaults_work_without_credentials() -> None:
    assert Settings.from_env() == Settings()


def test_environment_key_has_priority_and_is_not_in_repr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ANYTHINGLLM_API_KEY", "  secret-value  ")
    monkeypatch.setenv("ANYTHINGLLM_API_KEY_FILE", str(tmp_path / "missing"))
    settings = Settings.from_env()
    assert settings.api_key == "secret-value"
    assert "secret-value" not in repr(settings)


def test_explicit_key_file_overrides_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    default = tmp_path / "default-key"
    explicit = tmp_path / "explicit-key"
    default.write_text("default", encoding="utf-8")
    explicit.write_text("file-key\n", encoding="utf-8")
    monkeypatch.setenv("ANYTHINGLLM_API_KEY_FILE", str(explicit))
    assert Settings.from_env(default).api_key == "file-key"


def test_explicit_missing_key_file_fails_early(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ANYTHINGLLM_API_KEY_FILE", str(tmp_path / "missing"))
    with pytest.raises(ValueError, match="ANYTHINGLLM_API_KEY_FILE"):
        Settings.from_env()


@pytest.mark.parametrize("variable", ["ANYTHINGLLM_BASE_URL", "OLLAMA_BASE_URL"])
@pytest.mark.parametrize(
    "value",
    [
        "file:///tmp",
        "http://",
        "http://user:secret@localhost",
        "http://localhost?x=1",
        "http://localhost#fragment",
        "http://localhost:bad",
        "http://localhost:65536",
        "http://localhost:0",
        "http://local host",
        "http://[broken",
    ],
)
def test_invalid_urls_fail_with_setting_name(
    monkeypatch: pytest.MonkeyPatch, variable: str, value: str
) -> None:
    monkeypatch.setenv(variable, value)
    with pytest.raises(ValueError, match=variable):
        Settings.from_env()


@pytest.mark.parametrize(
    "variable",
    [
        "ANYTHINGLLM_HTTP_TIMEOUT",
        "ANYTHINGLLM_DOCUMENT_TIMEOUT",
        "ANYTHINGLLM_LLM_TIMEOUT",
    ],
)
@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_timeouts_must_be_finite_and_positive(
    monkeypatch: pytest.MonkeyPatch, variable: str, value: str
) -> None:
    monkeypatch.setenv(variable, value)
    with pytest.raises(ValueError, match=variable):
        Settings.from_env()
