"""Fixtures scoped to the associated unit-test module."""

import os

import pytest


@pytest.fixture(autouse=True)
def clean_settings_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in os.environ:
        if key.startswith(("ANYTHINGLLM_", "OLLAMA_")):
            monkeypatch.delenv(key)
