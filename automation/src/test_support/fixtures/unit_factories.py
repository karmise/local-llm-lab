"""Fresh unit objects and factories for tests requiring several configurations."""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from unittest.mock import AsyncMock, Mock

import pytest
from requests import Response

from llm_testkit.clients.anythingllm_client import AnythingLLMClient
from llm_testkit.clients.ollama_client import OllamaClient
from llm_testkit.config import Settings
from llm_testkit.core.http_client import HttpClient


@pytest.fixture
def mock_factory() -> Callable[..., Mock]:
    return Mock


@pytest.fixture
def async_mock_factory() -> Callable[..., AsyncMock]:
    return AsyncMock


@pytest.fixture
def response_factory() -> Callable[[], Response]:
    return Response


@pytest.fixture
def unit_settings() -> Settings:
    return Settings()


@pytest.fixture
def http_factory() -> Callable[..., HttpClient]:
    return HttpClient


@pytest.fixture
def api_factory() -> Callable[..., AnythingLLMClient]:
    return AnythingLLMClient


@pytest.fixture
def ollama_factory() -> Callable[..., OllamaClient]:
    return OllamaClient


@pytest.fixture
def operation_lock():
    return Lock()


@pytest.fixture
def evidence_workers():
    with ThreadPoolExecutor(max_workers=4) as workers:
        yield workers
