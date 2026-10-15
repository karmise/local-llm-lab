from typing import Any

import pytest

import assertions
from api.anythingllm import AnythingLLMClient
from config import Settings
from reporting.steps import title


@pytest.mark.api
@title("Uploaded policy is indexed and retrieved by vector search")
def test_policy_document_is_indexed_and_searchable(
        authenticated_anythingllm_api: AnythingLLMClient, indexed_workspace: dict[str, Any],
        uploaded_policy_document: dict[str, Any], paid_leave_profile: dict[str, Any], settings: Settings) -> None:
    slug = indexed_workspace["slug"]
    workspace_response = authenticated_anythingllm_api.workspaces.get(slug)
    assertions.assert_workspace_document_attached(
            workspace_response, slug=slug, location=uploaded_policy_document["location"])

    search_response = authenticated_anythingllm_api.workspaces.search(
            slug, paid_leave_profile["question"], timeout=settings.document_timeout)
    assertions.assert_search_contains(
            search_response, document_title=uploaded_policy_document["title"],
            fragments=paid_leave_profile["source_fragments"])
