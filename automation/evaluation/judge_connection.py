"""Connections to the local Ollama server for judge evaluation."""

from collections.abc import Iterator
from contextlib import contextmanager

import assertions
from api.http_client import HttpClient
from api.ollama import OllamaClient
from config import Settings


@contextmanager
def ollama_client(settings: Settings) -> Iterator[OllamaClient]:
    """An Ollama client whose connection is closed on exit, also after an error."""
    http = HttpClient(settings.ollama_base_url, settings.http_timeout)
    try:
        yield OllamaClient(http)
    finally:
        http.close()


def installed_digest(client: OllamaClient, model: str) -> str:
    """The digest of `model`, which must be installed before any judge call."""
    catalog = client.list_models()
    assertions.assert_status_code(catalog, 200, context="Judge model catalog")
    return assertions.assert_model_available(catalog.json()["models"], model)


@contextmanager
def judge_client(settings: Settings, judge_model: str) -> Iterator[tuple[OllamaClient, str]]:
    """An Ollama client and the digest of `judge_model`, which must be installed before any judge call."""
    with ollama_client(settings) as client:
        yield client, installed_digest(client, judge_model)
