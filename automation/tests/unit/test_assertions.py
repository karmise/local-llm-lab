import json

import pytest

from llm_testkit import assertions
from llm_testkit.reporting.steps import title
from test_support.assertions import errors as errors
from test_support.builders.assertions import _gym_response, _response
from test_support.data.assertions import (
    MISSING_INFORMATION_REJECTS_HALLUCINATED_OR_IRRELEVANT_ANSWERS_ANSWER_MESSAGE_CASES,
    RAG_REJECTS_UNSUBSTANTIATED_FINAL_ANSWERS_ANSWER_SOURCE_TITLE_SOURCE_TEXT_MESSAGE_CASES,
    SEARCH_REJECTS_INCOMPLETE_OR_UNRELATED_RESULTS_BODY_MESSAGE_CASES,
)

pytestmark = pytest.mark.unit


@title("Online-status check rejects an integer in place of a boolean")
def test_online_rejects_integer_instead_of_boolean() -> None:
    errors.rejects(
        lambda: assertions.assert_online(_response(b'{"online": 1}')),
        expected=AssertionError,
        match="Field online: expected bool, got int",
    )


@title("Response check clearly reports a missing required field")
def test_missing_field_has_a_clear_failure() -> None:
    errors.rejects(
        lambda: assertions.assert_online(_response(b"{}")),
        expected=AssertionError,
        match="Missing required field: online",
    )


@title("Response check rejects malformed JSON with an assertion failure")
def test_malformed_json_is_reported_as_an_assertion_failure() -> None:
    errors.rejects(
        lambda: assertions.assert_online(_response(b"not-json")),
        expected=AssertionError,
        match="Response body is not valid JSON",
    )


@title("Workspace check rejects list entries that are not objects")
def test_workspace_list_rejects_non_object_entries() -> None:
    errors.rejects(
        lambda: assertions.assert_workspace_matches(
            _response(b'{"workspace": ["invalid-entry"]}'), slug="automation-example"
        ),
        expected=AssertionError,
        match="Expected a workspace object",
    )


@pytest.mark.parametrize(
    ("body", "message"),
    SEARCH_REJECTS_INCOMPLETE_OR_UNRELATED_RESULTS_BODY_MESSAGE_CASES,
)
@title("Vector-search check rejects incomplete or unrelated results [{param_id}]")
def test_search_rejects_incomplete_or_unrelated_results(body: bytes, message: str) -> None:
    errors.rejects(
        lambda: assertions.assert_search_contains(
            _response(body),
            document_title="policy.txt",
            fragments=("23 working days", "12 calendar days"),
        ),
        expected=AssertionError,
        match=message,
    )


@pytest.mark.parametrize(
    ("answer", "source_title", "source_text", "message"),
    RAG_REJECTS_UNSUBSTANTIATED_FINAL_ANSWERS_ANSWER_SOURCE_TITLE_SOURCE_TEXT_MESSAGE_CASES,
)
@title("RAG check rejects unsupported or incomplete final answers [{param_id}]")
def test_rag_rejects_unsubstantiated_final_answers(
    answer: str, source_title: str, source_text: str, message: str
) -> None:
    payload = {
        "type": "textResponse",
        "error": None,
        "close": True,
        "textResponse": answer,
        "sources": [{"title": source_title, "text": source_text}],
    }
    errors.rejects(
        lambda: assertions.assert_rag_answer(
            _response(json.dumps(payload).encode()),
            fact_patterns={"paid leave": "\\b23\\s+working\\s+days\\b"},
            document_title="policy.txt",
            source_fragments=("23 working days",),
        ),
        expected=AssertionError,
        match=message,
    )


@pytest.mark.parametrize(
    ("answer", "message"),
    MISSING_INFORMATION_REJECTS_HALLUCINATED_OR_IRRELEVANT_ANSWERS_ANSWER_MESSAGE_CASES,
)
@title("Missing-policy check rejects invented amounts and irrelevant answers [{param_id}]")
def test_missing_information_rejects_hallucinated_or_irrelevant_answers(
    answer: str, message: str
) -> None:
    errors.rejects(
        lambda: assertions.assert_missing_policy_information(
            _gym_response(answer),
            document_title="policy.txt",
            source_fragments=("gym membership reimbursement policies are not covered",),
        ),
        expected=AssertionError,
        match=message,
    )


@title("Missing-policy check accepts document section numbers")
def test_missing_information_accepts_document_section_numbers() -> None:
    assertions.assert_missing_policy_information(
        _gym_response("Gym reimbursement is not covered, as stated in section 4."),
        document_title="policy.txt",
        source_fragments=("gym membership reimbursement policies are not covered",),
    )


@title("Model selection rejects a model that is not installed")
def test_model_selection_rejects_an_uninstalled_model() -> None:
    errors.rejects(
        lambda: assertions.assert_model_available([], "missing-model"),
        expected=AssertionError,
        match="Expected installed Ollama model: missing-model",
    )


@title("Model selection requires the installed model digest")
def test_model_selection_requires_a_digest() -> None:
    errors.rejects(
        lambda: assertions.assert_model_available(
            [{"name": "test-model", "digest": ""}], "test-model"
        ),
        expected=AssertionError,
        match="Expected a model digest",
    )
