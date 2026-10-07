from collections.abc import Callable
from typing import Any

import pytest
from requests import Response

from llm_testkit import assertions
from llm_testkit.reporting.steps import title

pytestmark = pytest.mark.rag


@title("RAG answer contains paid-leave facts and supporting sources [{param_id}]")
def test_paid_leave_answer_is_grounded_in_policy(
        rag_chat: Callable[[str, str], Response], uploaded_policy_document: dict[str, Any],
        paid_leave_profile: dict[str, Any]) -> None:  # fmt: skip
    response = rag_chat(paid_leave_profile["question"], paid_leave_profile["reference"])

    assertions.assert_rag_answer(
        response,
        fact_patterns=paid_leave_profile["fact_patterns"],
        document_title=uploaded_policy_document["title"],
        source_fragments=paid_leave_profile["source_fragments"],
    )


@title("RAG answer acknowledges missing gym policy without inventing reimbursement [{param_id}]")
def test_missing_gym_policy_does_not_invent_reimbursement(
        rag_chat: Callable[[str, str], Response], uploaded_policy_document: dict[str, Any],
        missing_policy_profile: dict[str, Any]) -> None:  # fmt: skip
    response = rag_chat(missing_policy_profile["question"], missing_policy_profile["reference"])

    assertions.assert_missing_policy_information(
        response,
        document_title=uploaded_policy_document["title"],
        source_fragments=missing_policy_profile["source_fragments"],
    )
