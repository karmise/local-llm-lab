"""Conversation-specific evidence and explicit Chat-mode generation."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from requests import Response

from llm_testkit import assertions
from llm_testkit.clients.anythingllm_client import AnythingLLMClient
from llm_testkit.config import Settings
from llm_testkit.datasets.conversation import ConversationCase, load_conversation_catalog
from llm_testkit.reporting.steps import set_metadata


@pytest.fixture
def conversation_metadata(
    automation_root: Path,
    policy_file: Path,
    conversation_case: ConversationCase,
    record_property: Callable[[str, object], None],
) -> None:
    catalog = load_conversation_catalog(
        automation_root / "test_data/conversation-policy.json", policy_file
    )
    if conversation_case not in catalog.cases:
        pytest.fail("Conversation catalog changed after collection; collect again", pytrace=False)
    record_property("conversation_case_id", conversation_case.id)
    record_property("conversation_catalog_version", catalog.version)
    record_property("conversation_catalog_sha256", catalog.sha256)
    record_property("conversation_contract", "Chat answers; conditional retrieval is not verified")
    set_metadata(feature="Conversational policy assistant", story=conversation_case.title)


@pytest.fixture
def conversation_chat(
    rag_environment: None,
    workspace_configuration: dict[str, Any],
    authenticated_anythingllm_api: AnythingLLMClient,
    indexed_workspace: dict[str, Any],
    settings: Settings,
) -> Callable[[str], Response]:
    assertions.assert_field_equals(workspace_configuration, "chatMode", "chat")

    def chat(question: str) -> Response:
        return authenticated_anythingllm_api.chat(
            indexed_workspace["slug"], question, mode="chat", timeout=settings.llm_timeout
        )

    return chat
