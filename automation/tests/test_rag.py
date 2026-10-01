from collections.abc import Callable
from typing import Any

import pytest
from requests import Response

from llm_testkit import assertions

pytestmark = pytest.mark.usefixtures("rag_environment")


@pytest.mark.rag
def test_paid_leave_answer_is_grounded_in_policy(
    rag_chat: Callable[[str, str], Response],
    uploaded_policy_document: dict[str, Any],
) -> None:
    response = rag_chat(
        "How many working days of paid leave does each employee receive per year, "
        "and how many calendar days before leave starts must a request be submitted?",
        "Each employee receives 23 working days of paid leave per year. "
        "A request must be submitted at least 12 calendar days before leave starts.",
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


@pytest.mark.rag
def test_missing_gym_policy_does_not_invent_reimbursement(
    rag_chat: Callable[[str, str], Response],
    uploaded_policy_document: dict[str, Any],
) -> None:
    response = rag_chat(
        "What is the company's gym membership reimbursement policy, "
        "and how much does it reimburse per month?",
        "The document does not cover gym membership reimbursement. "
        "No monthly reimbursement amount is specified.",
    )
    assertions.assert_missing_policy_information(
        response,
        document_title=uploaded_policy_document["title"],
        source_fragments=("gym membership reimbursement policies are not covered",),
    )
