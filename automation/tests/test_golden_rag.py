from collections.abc import Callable
from typing import Any

import pytest
from requests import Response

from llm_testkit import assertions
from llm_testkit.datasets.golden import GoldenCase
from llm_testkit.reporting.steps import title

pytestmark = [pytest.mark.rag, pytest.mark.golden]


@title("Golden policy answer satisfies its acceptance criteria [{param_id}]")
def test_golden_policy_answer(
    golden_case: GoldenCase,
    golden_metadata: None,
    rag_chat: Callable[[str, str], Response],
    uploaded_policy_document: dict[str, Any],
) -> None:
    response = rag_chat(golden_case.question, golden_case.reference)
    assertions.assert_golden_answer(
        response, case=golden_case, document_title=uploaded_policy_document["title"]
    )
