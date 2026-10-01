import pytest

from llm_testkit.clients.anythingllm_client import AnythingLLMClient


@pytest.mark.smoke
def test_anythingllm_is_online(anythingllm_api: AnythingLLMClient) -> None:
    response = anythingllm_api.health()

    assert response.status_code == 200, (
        f"Expected HTTP 200, got {response.status_code}: {response.text[:500]}"
    )
    assert response.json().get("online") is True, response.text[:500]
