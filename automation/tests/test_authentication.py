import pytest

from llm_testkit import assertions
from llm_testkit.clients.anythingllm_client import AnythingLLMClient
from llm_testkit.core.http_client import HttpClient


@pytest.mark.api
def test_valid_api_key_is_accepted(
    authenticated_anythingllm_api: AnythingLLMClient,
) -> None:
    response = authenticated_anythingllm_api.verify_authentication()

    assertions.assert_api_key_accepted(response)


@pytest.mark.api
def test_missing_api_key_is_rejected(anythingllm_api: AnythingLLMClient) -> None:
    response = anythingllm_api.verify_authentication()

    assertions.assert_api_key_rejected(response)


@pytest.mark.api
def test_invalid_api_key_is_rejected(http_client: HttpClient) -> None:
    client = AnythingLLMClient(http_client, api_key="invalid-test-key")
    response = client.verify_authentication()

    assertions.assert_api_key_rejected(response)
