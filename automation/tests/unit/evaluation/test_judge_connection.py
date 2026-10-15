"""Judge connection: the Ollama client is closed on every path and the judge model must be installed first."""

from unittest.mock import Mock

import pytest

from llm_testkit.config import Settings
from llm_testkit.evaluation import judge_connection
from llm_testkit.evaluation.judge_connection import judge_client, ollama_client
from llm_testkit.reporting.steps import title
from test_support.builders.identities import MODEL_DIGEST, TEST_MODEL
from test_support.builders.ollama import model_catalog

pytestmark = pytest.mark.unit


@pytest.fixture
def transport(monkeypatch):
    """The HTTP connection behind the Ollama client, with an installed test model."""
    http = Mock()
    monkeypatch.setattr(judge_connection, "HttpClient", Mock(return_value=http))
    monkeypatch.setattr(
            judge_connection.OllamaClient, "list_models", lambda _: model_catalog((TEST_MODEL, MODEL_DIGEST)))
    return http


@title("The Ollama client uses the configured URL and timeout and is closed on exit")
def test_ollama_client_closes_connection(transport):
    settings = Settings()

    with ollama_client(settings) as client:
        transport.close.assert_not_called()

    judge_connection.HttpClient.assert_called_once_with(settings.ollama_base_url, settings.http_timeout)
    assert client._http is transport
    transport.close.assert_called_once_with()


@title("The Ollama connection is closed when the work inside fails")
def test_ollama_client_closes_after_error(transport):
    with pytest.raises(RuntimeError, match="judge failed"), ollama_client(Settings()):
        raise RuntimeError("judge failed")

    transport.close.assert_called_once_with()


@title("The judge client yields the installed judge model's digest")
def test_judge_client_yields_digest(transport):
    with judge_client(Settings(), TEST_MODEL) as (client, digest):
        assert (client._http, digest) == (transport, MODEL_DIGEST)

    transport.close.assert_called_once_with()


NOT_INSTALLED = model_catalog(("other-model", MODEL_DIGEST))
UNAVAILABLE = model_catalog(status_code=503)


@pytest.mark.parametrize(("catalog", "error"), [
        pytest.param(NOT_INSTALLED, f"Expected installed Ollama model: {TEST_MODEL}", id="not-installed"),
        pytest.param(UNAVAILABLE, "Judge model catalog: expected HTTP 200, got 503", id="catalog-error")])
@title("A judge model that is not installed or not listed stops before any judge call [{param_id}]")
def test_judge_client_requires_installed_model(transport, monkeypatch, catalog, error):
    monkeypatch.setattr(judge_connection.OllamaClient, "list_models", lambda _: catalog)
    work = Mock()

    with pytest.raises(AssertionError, match=error), judge_client(Settings(), TEST_MODEL):
        work()

    work.assert_not_called()
    transport.close.assert_called_once_with()
