from collections.abc import Callable
from typing import Any

import pytest
from requests import Response

from llm_testkit import assertions
from llm_testkit.datasets.conversation import ConversationCase
from llm_testkit.reporting.steps import title

pytestmark = [pytest.mark.rag, pytest.mark.conversation]


@title("{conversation_case.title} [{param_id}]")
def test_policy_conversation(conversation_case: ConversationCase, conversation_metadata: None,
                             conversation_chat: Callable[[str], Response],
                             uploaded_policy_document: dict[str, Any]) -> None:  # fmt: skip
    response = conversation_chat(conversation_case.question)
    assertions.assert_conversation_answer(
        response, case=conversation_case, document_title=uploaded_policy_document["title"]
    )
