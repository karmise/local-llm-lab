import json

import pytest
from requests import Response

from llm_testkit import assertions

pytestmark = pytest.mark.unit


def _response(body: bytes) -> Response:
    response = Response()
    response.status_code = 200
    response._content = body
    return response


def test_online_rejects_integer_instead_of_boolean() -> None:
    with pytest.raises(AssertionError, match="Field online: expected bool, got int"):
        assertions.assert_online(_response(b'{"online": 1}'))


def test_missing_field_has_a_clear_failure() -> None:
    with pytest.raises(AssertionError, match="Missing required field: online"):
        assertions.assert_online(_response(b'{}'))


def test_malformed_json_is_reported_as_an_assertion_failure() -> None:
    with pytest.raises(AssertionError, match="Response body is not valid JSON"):
        assertions.assert_online(_response(b'not-json'))


def test_workspace_list_rejects_non_object_entries() -> None:
    with pytest.raises(AssertionError, match="Expected a workspace object"):
        assertions.assert_workspace_matches(
            _response(b'{"workspace": ["invalid-entry"]}'), slug="automation-example"
        )


@pytest.mark.parametrize(
    ("body", "message"),
    [
        pytest.param(b'{"results": []}', "vector search returned no results", id="empty-index"),
        pytest.param(
            b'{"results": [{"text": "23 working days and 12 calendar days", "metadata": {"title": "other.txt"}}]}',
            "Search did not return the uploaded document", id="wrong-document",
        ),
        pytest.param(
            b'{"results": [{"text": "23 working days", "metadata": {"title": "policy.txt"}}]}',
            "12 calendar days", id="missing-fact",
        ),
    ],
)
def test_search_rejects_incomplete_or_unrelated_results(body: bytes, message: str) -> None:
    with pytest.raises(AssertionError, match=message):
        assertions.assert_search_contains(
            _response(body), document_title="policy.txt",
            fragments=("23 working days", "12 calendar days"),
        )


@pytest.mark.parametrize(
    ("answer", "source_title", "source_text", "message"),
    [
        pytest.param(
            "<think>23 working days</think>No policy information is available.",
            "policy.txt", "23 working days", "missing expected fact", id="reasoning-only-fact",
        ),
        pytest.param(
            "Employees get 25 working days.", "policy.txt", "23 working days",
            "missing expected fact", id="incorrect-amount",
        ),
        pytest.param(
            "Employees get 23 working days.", "other.txt", "23 working days",
            "did not cite the uploaded document", id="wrong-source",
        ),
        pytest.param(
            "Employees get 23 working days.", "policy.txt", "No leave details are provided.",
            "source_text", id="unsupported-fact",
        ),
        pytest.param(
            "<think>Employees get 23 working days.", "policy.txt", "23 working days",
            "incomplete thinking tags", id="unclosed-thinking",
        ),
    ],
)
def test_rag_rejects_unsubstantiated_final_answers(
    answer: str, source_title: str, source_text: str, message: str
) -> None:
    payload = {
        "type": "textResponse", "error": None, "close": True, "textResponse": answer,
        "sources": [{"title": source_title, "text": source_text}],
    }
    with pytest.raises(AssertionError, match=message):
        assertions.assert_rag_answer(
            _response(json.dumps(payload).encode()),
            fact_patterns={"paid leave": r"\b23\s+working\s+days\b"},
            document_title="policy.txt", source_fragments=("23 working days",),
        )


def _gym_response(answer: str) -> Response:
    payload = {
        "type": "textResponse", "error": None, "close": True, "textResponse": answer,
        "sources": [{"title": "policy.txt", "text": "gym membership reimbursement policies are not covered"}],
    }
    return _response(json.dumps(payload).encode())


@pytest.mark.parametrize(
    ("answer", "message"),
    [
        pytest.param(
            "Gym details are not provided, but reimbursement is KGS 500.",
            "must not propose a reimbursement amount", id="disclaimer-with-amount",
        ),
        pytest.param(
            "Gym policy is not specified; employees can claim 500 per month.",
            "must not propose a reimbursement amount", id="amount-without-currency",
        ),
        pytest.param(
            "Gym policy is not specified; reimbursement is five hundred som.",
            "must not propose a reimbursement amount", id="written-amount",
        ),
        pytest.param(
            "<think>Gym policy is not covered.</think>The company pays for gym membership.",
            "Expected an explicit statement", id="reasoning-only-abstention",
        ),
        pytest.param(
            "Travel rules are not provided in the document.",
            "address gym reimbursement", id="wrong-topic",
        ),
    ],
)
def test_missing_information_rejects_hallucinated_or_irrelevant_answers(
    answer: str, message: str
) -> None:
    with pytest.raises(AssertionError, match=message):
        assertions.assert_missing_policy_information(
            _gym_response(answer), document_title="policy.txt",
            source_fragments=("gym membership reimbursement policies are not covered",),
        )


def test_missing_information_accepts_document_section_numbers() -> None:
    assertions.assert_missing_policy_information(
        _gym_response("Gym reimbursement is not covered, as stated in section 4."),
        document_title="policy.txt",
        source_fragments=("gym membership reimbursement policies are not covered",),
    )


def test_model_selection_rejects_an_uninstalled_model() -> None:
    with pytest.raises(AssertionError, match="Expected installed Ollama model: missing-model"):
        assertions.assert_model_available([], "missing-model")


def test_model_selection_requires_a_digest() -> None:
    with pytest.raises(AssertionError, match="Expected a model digest"):
        assertions.assert_model_available([{"name": "test-model", "digest": ""}], "test-model")
