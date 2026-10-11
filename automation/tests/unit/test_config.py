"""Settings: environment configuration with safe defaults, validated URLs and timeouts, and a private API key."""

import os
import re

import pytest

from llm_testkit.config import Settings
from llm_testkit.reporting.steps import title

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    """Each test starts without application settings in the environment."""
    for key in os.environ:
        if key.startswith(("ANYTHINGLLM_", "OLLAMA_")):
            monkeypatch.delenv(key)


@title("Default configuration targets the local lab and works without credentials")
def test_defaults():
    settings = Settings.from_env()

    assert settings == Settings(
            base_url="http://127.0.0.1:3001", ollama_base_url="http://127.0.0.1:11434", http_timeout=5.0,
            document_timeout=180.0, llm_timeout=300.0, api_key=None, workspace_slug="company-policy-lab")


@title("Every setting is read from its environment variable, and trailing slashes are removed from URLs")
def test_settings_from_environment(monkeypatch):
    for name, value in {"ANYTHINGLLM_BASE_URL": "https://lab.example:8443/", "OLLAMA_BASE_URL": "http://ollama:11434/",
            "ANYTHINGLLM_HTTP_TIMEOUT": "2.5", "ANYTHINGLLM_DOCUMENT_TIMEOUT": "60", "ANYTHINGLLM_LLM_TIMEOUT": "90",
            "ANYTHINGLLM_WORKSPACE_SLUG": "other-workspace"}.items():
        monkeypatch.setenv(name, value)

    settings = Settings.from_env()

    assert settings == Settings(
            base_url="https://lab.example:8443", ollama_base_url="http://ollama:11434", http_timeout=2.5,
            document_timeout=60.0, llm_timeout=90.0, workspace_slug="other-workspace")


@title("The environment API key has priority, is stripped and never appears in the representation")
def test_environment_key_has_priority(monkeypatch, tmp_path):
    monkeypatch.setenv("ANYTHINGLLM_API_KEY", "  secret-value  ")
    monkeypatch.setenv("ANYTHINGLLM_API_KEY_FILE", str(tmp_path / "missing"))

    settings = Settings.from_env(tmp_path / "default-key")

    assert settings.api_key == "secret-value"
    assert "secret-value" not in repr(settings)


@title("A blank environment API key falls back to the key file")
def test_blank_environment_key_uses_file(monkeypatch, tmp_path):
    monkeypatch.setenv("ANYTHINGLLM_API_KEY", "  ")
    (tmp_path / "key").write_text("file-key\n", encoding="utf-8")

    assert Settings.from_env(tmp_path / "key").api_key == "file-key"


@title("An explicit API key file overrides the default key file")
def test_explicit_key_file_overrides_default(monkeypatch, tmp_path):
    (tmp_path / "default-key").write_text("default", encoding="utf-8")
    (tmp_path / "explicit-key").write_text("file-key\n", encoding="utf-8")
    monkeypatch.setenv("ANYTHINGLLM_API_KEY_FILE", str(tmp_path / "explicit-key"))

    assert Settings.from_env(tmp_path / "default-key").api_key == "file-key"


@title("The default key file is used when no key is configured")
def test_default_key_file(tmp_path):
    (tmp_path / "default-key").write_text(" default-key \n", encoding="utf-8")

    assert Settings.from_env(tmp_path / "default-key").api_key == "default-key"


@title("An empty key file means no key")
def test_empty_key_file_means_no_key(tmp_path):
    (tmp_path / "default-key").write_text(" \n", encoding="utf-8")

    assert Settings.from_env(tmp_path / "default-key").api_key is None


@title("A missing default key file means no key")
def test_missing_default_key_file_means_no_key(tmp_path):
    assert Settings.from_env(tmp_path / "absent").api_key is None


@title("A key file is read as UTF-8")
def test_key_file_is_utf8(tmp_path):
    (tmp_path / "default-key").write_bytes("ключ".encode())

    assert Settings.from_env(tmp_path / "default-key").api_key == "ключ"


@pytest.mark.parametrize("target", [pytest.param("missing", id="missing"), pytest.param("directory", id="directory")])
@title("An explicit key file that is not an existing file fails before any request [{param_id}]")
def test_explicit_key_file_must_exist(monkeypatch, tmp_path, target):
    (tmp_path / "directory").mkdir()
    monkeypatch.setenv("ANYTHINGLLM_API_KEY_FILE", str(tmp_path / target))

    with pytest.raises(ValueError, match="^ANYTHINGLLM_API_KEY_FILE does not point to an existing file$"):
        Settings.from_env(tmp_path / "default-key")


@pytest.mark.parametrize(("variable", "value", "message"), [
        pytest.param("ANYTHINGLLM_BASE_URL", "file:///tmp", "must be an HTTP(S) URL", id="unsupported-scheme"),
        pytest.param("OLLAMA_BASE_URL", "http://", "must be an HTTP(S) URL", id="missing-host"),
        pytest.param("OLLAMA_BASE_URL", "localhost:11434", "must be an HTTP(S) URL", id="no-scheme"),
        pytest.param(
        "ANYTHINGLLM_BASE_URL", "http://localhost:bad", "must be an HTTP(S) URL with a valid port",
        id="nonnumeric-port"),
        pytest.param(
        "ANYTHINGLLM_BASE_URL", "http://localhost:65536", "must be an HTTP(S) URL with a valid port",
        id="port-overflow"),
        pytest.param(
        "OLLAMA_BASE_URL", "http://[broken", "must be an HTTP(S) URL with a valid port", id="malformed-ipv6"),
        pytest.param(
        "OLLAMA_BASE_URL", "http://localhost:0", "must not contain whitespace or an invalid port", id="zero-port"),
        pytest.param(
        "ANYTHINGLLM_BASE_URL", "http://local host", "must not contain whitespace or an invalid port", id="whitespace"),
        pytest.param(
        "ANYTHINGLLM_BASE_URL", "http://user:secret@localhost", "must not contain credentials, query or fragment",
        id="credentials"),
        pytest.param(
        "ANYTHINGLLM_BASE_URL", "http://user@localhost", "must not contain credentials, query or fragment",
        id="user-only"),
        pytest.param(
        "OLLAMA_BASE_URL", "http://localhost?x=1", "must not contain credentials, query or fragment", id="query"),
        pytest.param(
        "ANYTHINGLLM_BASE_URL", "http://localhost#fragment", "must not contain credentials, query or fragment",
        id="fragment")])
@title("An invalid URL names the setting and the rule it breaks [{param_id}]")
def test_invalid_urls(monkeypatch, variable, value, message):
    monkeypatch.setenv(variable, value)

    with pytest.raises(ValueError, match="^" + re.escape(f"{variable} {message}") + "$"):
        Settings.from_env()


@pytest.mark.parametrize(
        "address", [
        pytest.param("https://lab.example", id="https"),
        pytest.param("http://[::1]:3001", id="ipv6"),
        pytest.param("http://localhost:65535", id="highest-port"),
        pytest.param("http://localhost/api", id="path")])
@title("Valid HTTP(S) URLs are accepted [{param_id}]")
def test_valid_urls(address):
    assert Settings(base_url=address, ollama_base_url=address).base_url == address


@pytest.mark.parametrize(
        "variable", ["ANYTHINGLLM_HTTP_TIMEOUT", "ANYTHINGLLM_DOCUMENT_TIMEOUT", "ANYTHINGLLM_LLM_TIMEOUT"])
@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
@title("Every timeout must be a finite positive number [{variable}={value}]")
def test_invalid_timeouts(monkeypatch, variable, value):
    monkeypatch.setenv(variable, value)

    with pytest.raises(ValueError, match=f"^{variable} must be a finite positive number$"):
        Settings.from_env()


@pytest.mark.parametrize("field", ["http_timeout", "document_timeout", "llm_timeout"])
@title("A very small positive timeout is accepted [{field}]")
def test_small_positive_timeout(field):
    assert getattr(Settings(**{field: 0.001}), field) == 0.001


@title("A timeout that is not a number fails when the environment is read")
def test_non_numeric_timeout(monkeypatch):
    monkeypatch.setenv("ANYTHINGLLM_LLM_TIMEOUT", "slow")

    with pytest.raises(ValueError, match="could not convert string to float"):
        Settings.from_env()
