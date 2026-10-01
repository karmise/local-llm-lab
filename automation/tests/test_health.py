import pytest

from llm_testkit.reporting.steps import title
from llm_testkit import assertions
from llm_testkit.clients.anythingllm_client import AnythingLLMClient


@pytest.mark.smoke
@title('AnythingLLM is reachable and reports online status')
def test_anythingllm_is_online(anythingllm_api: AnythingLLMClient) -> None:
    response = anythingllm_api.health()

    assertions.assert_online(response)
