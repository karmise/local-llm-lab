from collections.abc import Iterator
from pathlib import Path

import pytest

from llm_testkit.clients.anythingllm_client import AnythingLLMClient
from llm_testkit.config import Settings
from llm_testkit.core.http_client import HttpClient


@pytest.fixture(scope="session")
def settings() -> Settings:
    key_file = Path(__file__).resolve().parents[2] / ".runtime" / "anythingllm-api-key"
    return Settings.from_env(default_api_key_file=key_file)


@pytest.fixture
def http_client(settings: Settings) -> Iterator[HttpClient]:
    client = HttpClient(base_url=settings.base_url, timeout=settings.http_timeout)
    try:
        yield client
    finally:
        client.close()


@pytest.fixture
def anythingllm_api(http_client: HttpClient) -> AnythingLLMClient:
    return AnythingLLMClient(http_client)


@pytest.fixture
def authenticated_anythingllm_api(
    http_client: HttpClient, settings: Settings
) -> AnythingLLMClient:
    if not settings.api_key:
        pytest.fail(
            "Developer API key is missing. Configure ANYTHINGLLM_API_KEY or "
            "ANYTHINGLLM_API_KEY_FILE; see docs/step-03-developer-api.md.",
            pytrace=False,
        )
    return AnythingLLMClient(http_client, api_key=settings.api_key)
