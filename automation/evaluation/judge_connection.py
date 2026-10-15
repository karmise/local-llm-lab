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


@contextmanager
def judge_client(settings: Settings, judge_model: str) -> Iterator[tuple[OllamaClient, str]]:
    """An Ollama client and the digest of `judge_model`, which must be installed before any judge call."""
    with ollama_client(settings) as client:
        catalog = client.list_models()
        assertions.assert_status_code(catalog, 200, context="Judge model catalog")
        yield client, assertions.assert_model_available(catalog.json()["models"], judge_model)
