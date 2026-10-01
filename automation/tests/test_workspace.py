import pytest

from llm_testkit.clients.anythingllm_client import AnythingLLMClient
from llm_testkit.config import Settings


@pytest.mark.api
def test_configured_workspace_is_available(
    authenticated_anythingllm_api: AnythingLLMClient, settings: Settings
) -> None:
    response = authenticated_anythingllm_api.get_workspace(settings.workspace_slug)

    assert response.status_code == 200, f"Expected HTTP 200, got {response.status_code}"
    workspaces = response.json()["workspace"]
    assert isinstance(workspaces, list), "Expected a workspace list"
    assert len(workspaces) == 1, "Expected exactly one configured workspace"
    workspace = workspaces[0]
    assert workspace["slug"] == settings.workspace_slug
    assert isinstance(workspace["id"], int) and workspace["id"] > 0
