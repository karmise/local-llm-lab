from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from requests import Response

from llm_testkit import assertions
from llm_testkit.datasets.adversarial import (
    AdversarialCase,
)
from llm_testkit.reporting.steps import title
from test_support.fixtures.adversarial import adversarial_metadata as adversarial_metadata
from test_support.fixtures.adversarial import policy_file as policy_file

pytestmark = [pytest.mark.rag, pytest.mark.adversarial]


@title("Policy answer resists curated adversarial input [{param_id}]")
def test_policy_resists_adversarial_input(
    adversarial_case: AdversarialCase,
    adversarial_metadata: None,
    rag_chat: Callable[[str, str], Response],
    uploaded_policy_document: dict[str, Any],
    capture_id: str | None,
    automation_root: Path,
) -> None:
    response = rag_chat(adversarial_case.question, adversarial_case.golden_case.reference)
    assertions.assert_adversarial_context(
        adversarial_case, automation_root / f"reports/rag-samples/{capture_id}.json"
    )
    assertions.assert_adversarial_answer(
        response, case=adversarial_case, document_title=uploaded_policy_document["title"]
    )
