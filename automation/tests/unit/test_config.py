from pathlib import Path

import pytest

from llm_testkit.config import Settings
from llm_testkit.reporting.steps import title
from test_support.assertions import errors as errors
from test_support.assertions import values as value_checks
from test_support.data.config import INVALID_TIMEOUT_CASES, INVALID_URL_CASES
from test_support.fixtures.unit_config import (
    clean_settings_environment as clean_settings_environment,
)

pytestmark = pytest.mark.unit


@title("Default configuration works without application credentials")
def test_defaults_work_without_credentials(unit_settings) -> None:
    value_checks.equal(Settings.from_env(), unit_settings)


@title("Environment API key takes priority and is omitted from settings representation")
def test_environment_key_has_priority_and_is_not_in_repr(monkeypatch: pytest.MonkeyPatch,
                                                         tmp_path: Path) -> None:  # fmt: skip
    monkeypatch.setenv("ANYTHINGLLM_API_KEY", "  secret-value  ")
    monkeypatch.setenv("ANYTHINGLLM_API_KEY_FILE", str(tmp_path / "missing"))
    settings = Settings.from_env()
    value_checks.equal(settings.api_key, "secret-value")
    value_checks.excludes(repr(settings), "secret-value")


@title("Explicit API key file overrides the default key file")
def test_explicit_key_file_overrides_default(monkeypatch: pytest.MonkeyPatch,
                                             tmp_path: Path) -> None:  # fmt: skip
    default = tmp_path / "default-key"
    explicit = tmp_path / "explicit-key"
    default.write_text("default", encoding="utf-8")
    explicit.write_text("file-key\n", encoding="utf-8")
    monkeypatch.setenv("ANYTHINGLLM_API_KEY_FILE", str(explicit))
    value_checks.equal(Settings.from_env(default).api_key, "file-key")


@title("Missing explicit API key file fails before requests")
def test_explicit_missing_key_file_fails_early(monkeypatch: pytest.MonkeyPatch,
                                               tmp_path: Path) -> None:  # fmt: skip
    monkeypatch.setenv("ANYTHINGLLM_API_KEY_FILE", str(tmp_path / "missing"))
    errors.rejects(
        lambda: Settings.from_env(), expected=ValueError, match="ANYTHINGLLM_API_KEY_FILE"
    )


@pytest.mark.parametrize("variable,value", INVALID_URL_CASES)
@title("Invalid URLs identify the setting requiring correction [{param_id}]")
def test_invalid_urls_fail_with_setting_name(monkeypatch: pytest.MonkeyPatch, variable: str,
                                             value: str) -> None:  # fmt: skip
    monkeypatch.setenv(variable, value)
    errors.rejects(lambda: Settings.from_env(), expected=ValueError, match=variable)


@pytest.mark.parametrize("variable,value", INVALID_TIMEOUT_CASES)
@title("Timeout configuration rejects nonpositive or nonfinite values [{param_id}]")
def test_timeouts_must_be_finite_and_positive(monkeypatch: pytest.MonkeyPatch, variable: str,
                                              value: str) -> None:  # fmt: skip
    monkeypatch.setenv(variable, value)
    errors.rejects(lambda: Settings.from_env(), expected=ValueError, match=variable)
