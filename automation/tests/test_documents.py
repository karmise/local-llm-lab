from typing import Any

import pytest

from llm_testkit.reporting.steps import title
from llm_testkit import assertions
from llm_testkit.clients.anythingllm_client import AnythingLLMClient
from llm_testkit.config import Settings


@pytest.mark.api
@title('Uploaded policy is indexed and retrieved by vector search')
def test_policy_document_is_indexed_and_searchable(
    authenticated_anythingllm_api: AnythingLLMClient,
    indexed_workspace: dict[str, Any],
    uploaded_policy_document: dict[str, Any],
    settings: Settings,
) -> None:
    slug = indexed_workspace["slug"]
    workspace_response = authenticated_anythingllm_api.get_workspace(slug)
    assertions.assert_workspace_document_attached(
        workspace_response, slug=slug, location=uploaded_policy_document["location"]
    )

    search_response = authenticated_anythingllm_api.search_workspace(
        slug, "How much paid leave is available and when must a request be submitted?",
        timeout=settings.document_timeout,
    )
    assertions.assert_search_contains(
        search_response,
        document_title=uploaded_policy_document["title"],
        fragments=("23 working days", "12 calendar days"),
    )
