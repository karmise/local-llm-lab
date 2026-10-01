import pytest

from llm_testkit import assertions
from llm_testkit.clients.anythingllm_client import AnythingLLMClient


@pytest.mark.smoke
def test_anythingllm_is_online(anythingllm_api: AnythingLLMClient) -> None:
    response = anythingllm_api.health()

    assertions.assert_online(response)
