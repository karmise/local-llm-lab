import pytest

import assertions
from api.anythingllm import AnythingLLMClient
from reporting.steps import title


@pytest.mark.smoke
@title("AnythingLLM is reachable and reports online status")
def test_anythingllm_is_online(anythingllm_api: AnythingLLMClient) -> None:
    response = anythingllm_api.health()

    assertions.assert_online(response)
