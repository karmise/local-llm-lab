from typing import Any

import pytest

from llm_testkit import assertions
from llm_testkit.clients.anythingllm import AnythingLLMClient
from llm_testkit.config import Settings
from llm_testkit.reporting.steps import title


@pytest.mark.api
@title("Configured workspace is available with expected settings")
def test_configured_workspace_is_available(
        authenticated_anythingllm_api: AnythingLLMClient, settings: Settings) -> None:
    response = authenticated_anythingllm_api.workspaces.get(settings.workspace_slug)

    assertions.assert_workspace_matches(response, slug=settings.workspace_slug)


@pytest.mark.api
@title("Temporary workspace is created and removed successfully")
def test_temporary_workspace_lifecycle(
        authenticated_anythingllm_api: AnythingLLMClient, temporary_workspace: dict[str, Any],
        workspace_configuration: dict[str, Any]) -> None:
    response = authenticated_anythingllm_api.workspaces.get(temporary_workspace["slug"])

    assertions.assert_workspace_matches(
            response, slug=temporary_workspace["slug"], workspace_id=temporary_workspace["id"],
            name=temporary_workspace["name"], configuration=workspace_configuration)
