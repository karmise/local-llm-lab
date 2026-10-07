"""Named parameter cases for assertions scenarios."""

import pytest

from test_support.data import common as case_data

SEARCH_REJECTS_INCOMPLETE_OR_UNRELATED_RESULTS_BODY_MESSAGE_CASES = [
    pytest.param(b'{"results": []}', "vector search returned no results", id="empty-index"),
    pytest.param(
        b'{"results": [{"text": "23 working days and 12 calendar days", "metadata": {"title": "other.txt"}}]}',
        "Search did not return the uploaded document",
        id="wrong-document",
    ),
    pytest.param(
        b'{"results": [{"text": "23 working days", "metadata": {"title": "policy.txt"}}]}',
        "12 calendar days",
        id="missing-fact",
    ),
]

RAG_REJECTS_UNSUBSTANTIATED_FINAL_ANSWERS_ANSWER_SOURCE_TITLE_SOURCE_TEXT_MESSAGE_CASES = [
    pytest.param(
        "<think>23 working days</think>No policy information is available.",
        case_data.POLICY_DOCUMENT_TITLE,
        "23 working days",
        "missing expected fact",
        id="reasoning-only-fact",
    ),
    pytest.param(
        "Employees get 25 working days.",
        case_data.POLICY_DOCUMENT_TITLE,
        "23 working days",
        "missing expected fact",
        id="incorrect-amount",
    ),
    pytest.param(
        "Employees get 23 working days.",
        "other.txt",
        "23 working days",
        "did not cite the uploaded document",
        id="wrong-source",
    ),
    pytest.param(
        "Employees get 23 working days.",
        case_data.POLICY_DOCUMENT_TITLE,
        "No leave details are provided.",
        "source_text",
        id="unsupported-fact",
    ),
    pytest.param(
        "<think>Employees get 23 working days.",
        case_data.POLICY_DOCUMENT_TITLE,
        "23 working days",
        "incomplete thinking tags",
        id="unclosed-thinking",
    ),
]

MISSING_INFORMATION_REJECTS_HALLUCINATED_OR_IRRELEVANT_ANSWERS_ANSWER_MESSAGE_CASES = [
    pytest.param(
        "Gym details are not provided, but reimbursement is KGS 500.",
        "must not propose a reimbursement amount",
        id="disclaimer-with-amount",
    ),
    pytest.param(
        "Gym policy is not specified; employees can claim 500 per month.",
        "must not propose a reimbursement amount",
        id="amount-without-currency",
    ),
    pytest.param(
        "Gym policy is not specified; reimbursement is five hundred som.",
        "must not propose a reimbursement amount",
        id="written-amount",
    ),
    pytest.param(
        "<think>Gym policy is not covered.</think>The company pays for gym membership.",
        "Expected an explicit statement",
        id="reasoning-only-abstention",
    ),
    pytest.param(
        "Travel rules are not provided in the document.",
        "address gym reimbursement",
        id="wrong-topic",
    ),
]


# Input for test_rag_rejects_unsubstantiated_final_answers
PAID_LEAVE_FACT_PATTERNS = {"paid leave": "\\b23\\s+working\\s+days\\b"}
