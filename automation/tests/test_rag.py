from typing import Any

import pytest

from llm_testkit import assertions
from llm_testkit.clients.anythingllm_client import AnythingLLMClient
from llm_testkit.config import Settings


@pytest.mark.rag
def test_paid_leave_answer_is_grounded_in_policy(
    authenticated_anythingllm_api: AnythingLLMClient,
    indexed_workspace: dict[str, Any],
    uploaded_policy_document: dict[str, Any],
    settings: Settings,
) -> None:
    response = authenticated_anythingllm_api.chat(
        indexed_workspace["slug"],
        "How many working days of paid leave does each employee receive per year, "
        "and how many calendar days before leave starts must a request be submitted?",
        timeout=settings.llm_timeout,
    )
    assertions.assert_rag_answer(
        response,
        fact_patterns={
            "23 working days of paid leave": r"\b(?:23|twenty[- ]three)\s+(?:working|business)\s+days\b",
            "12 calendar days before leave starts": (
                r"\b(?:12|twelve)\s+calendar\s+days\b.{0,100}\b(?:before|prior to|in advance)\b"
            ),
        },
        document_title=uploaded_policy_document["title"],
        source_fragments=("23 working days", "12 calendar days"),
    )
