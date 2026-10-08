"""Unit tests cannot accidentally call the application or local models."""

import pytest
import requests


@pytest.fixture(autouse=True)
def offline_unit_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGAS_DO_NOT_TRACK", "true")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")

    def unexpected_request(*args, **kwargs):
        pytest.fail("Unit tests must mock HTTP requests; use API/RAG tests for live calls")

    monkeypatch.setattr(requests.sessions.Session, "request", unexpected_request)


@pytest.fixture
def framework_pytester(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> pytest.Pytester:
    # Child pytest runs exercise real collection/setup/teardown without optional plugins.
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    pytester.makeini(
            """
        [pytest]
        addopts = --strict-markers --strict-config
        markers =
            conversation: opt-in conversational acceptance checks
            bias: opt-in paired counterfactual scenarios
            performance: opt-in bounded performance batches
            adversarial: opt-in adversarial scenarios
            prompt_regression: opt-in prompt comparison
            golden: opt-in golden RAG checks
            ui: live browser tests
            browser: offline browser tests
            rag: generated answer tests
            live_quality: live quality tests
            unit: offline tests
    """)
    return pytester
