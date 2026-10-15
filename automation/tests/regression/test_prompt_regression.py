from collections.abc import Callable
from typing import Any

import pytest
from requests import Response

import assertions
from models.golden import GoldenCase
from models.prompts import PromptVariant
from reporting.steps import title

pytestmark = [pytest.mark.rag, pytest.mark.prompt_regression]


@title("Versioned prompt preserves golden policy acceptance criteria [{param_id}]")
def test_prompt_preserves_policy_answer(
        prompt_variant: PromptVariant, golden_case: GoldenCase, golden_metadata: None, rag_chat: Callable[[str, str],
        Response], uploaded_policy_document: dict[str, Any]) -> None:
    response = rag_chat(golden_case.question, golden_case.reference)
    assertions.assert_golden_answer(response, case=golden_case, document_title=uploaded_policy_document["title"])
