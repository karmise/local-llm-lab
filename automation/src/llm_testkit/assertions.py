"""Reusable API, quality and UI checks; clients and page objects do not assert outcomes."""

import json
import math
import re
from collections.abc import Mapping, Sequence, Sized
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from requests import Response

from llm_testkit.reporting.steps import attach_screenshot, attach_text, step

if TYPE_CHECKING:
    from llm_testkit.datasets.adversarial import AdversarialCase
    from llm_testkit.datasets.bias import BiasCase
    from llm_testkit.datasets.conversation import ConversationCase
    from llm_testkit.datasets.golden import GoldenCase
    from llm_testkit.pages.workspace_page import WorkspacePage

T = TypeVar("T")


def assert_quality_score(value: float, *, minimum: float | None = None) -> None:
    assert type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1, (
        f"Expected a finite quality score between 0 and 1, got {value!r}"
    )
    if minimum is not None:
        assert type(minimum) in (int, float) and math.isfinite(minimum) and 0 <= minimum <= 1, (
            "Quality threshold must be a finite number between 0 and 1"
        )
        assert value >= minimum, f"Quality score {value:.3f} is below {minimum:.3f}"


def assert_calibration_result(
    result: Mapping[str, Any],
    *,
    expected_score: float,
    claims: Sequence[Mapping[str, Any]],
) -> None:
    score = result["value"]
    assert_quality_score(score)
    assert_quality_score(expected_score)
    assert math.isclose(score, expected_score, rel_tol=0, abs_tol=1e-9), (
        f"Control score: expected {expected_score}, got {score}"
    )
    verdicts = assert_field_type(result, "verdicts", list)
    assert len(verdicts) == len(claims), "Control extraction changed the expected number of claims"
    matched_indices: set[int] = set()
    for claim in claims:
        matches = [
            index
            for index, item in enumerate(verdicts)
            if re.search(claim["pattern"], item["statement"], flags=re.IGNORECASE)
        ]
        assert len(matches) == 1, f"Expected one extracted claim matching {claim['pattern']}"
        index = matches[0]
        assert index not in matched_indices, (
            "Control claims must map to distinct extracted statements"
        )
        matched_indices.add(index)
        assert_field_equals(verdicts[index], "verdict", claim["verdict"])


def assert_status_code(response: Response, expected: int, *, context: str = "Response") -> None:
    assert response.status_code == expected, (
        f"{context}: expected HTTP {expected}, got {response.status_code}"
    )


def assert_json_object(response: Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        raise AssertionError("Response body is not valid JSON") from None
    assert isinstance(payload, dict), "Expected a JSON object in the response body"
    return payload


def assert_field_type(payload: Mapping[str, Any], field: str, expected: type[T]) -> T:
    assert field in payload, f"Missing required field: {field}"
    value = payload[field]
    assert type(value) is expected, (
        f"Field {field}: expected {expected.__name__}, got {type(value).__name__}"
    )
    return value


def assert_field_equals(payload: Mapping[str, Any], field: str, expected: Any) -> None:
    value = assert_field_type(payload, field, type(expected))
    assert value == expected, f"Field {field}: expected {expected!r}, got {value!r}"


def assert_field_length(payload: Mapping[str, Any], field: str, expected: int) -> None:
    assert field in payload, f"Missing required field: {field}"
    value = payload[field]
    assert isinstance(value, Sized), f"Field {field} does not have a length"
    assert len(value) == expected, f"Field {field}: expected length {expected}, got {len(value)}"


def assert_field_contains(payload: Mapping[str, Any], field: str, expected: str) -> None:
    value = assert_field_type(payload, field, str)
    assert expected in value, f"Field {field}: expected to contain {expected!r}"


def assert_field_starts_with(payload: Mapping[str, Any], field: str, expected: str) -> None:
    value = assert_field_type(payload, field, str)
    assert value.startswith(expected), f"Field {field}: expected prefix {expected!r}"


def assert_online(response: Response) -> None:
    assert_status_code(response, 200)
    assert_field_equals(assert_json_object(response), "online", True)


def assert_api_key_accepted(response: Response) -> None:
    assert_status_code(response, 200)
    payload = assert_json_object(response)
    assert_field_equals(payload, "authenticated", True)
    assert set(payload) == {"authenticated"}, "Unexpected authentication response fields"


def assert_api_key_rejected(response: Response) -> None:
    assert_status_code(response, 403)
    payload = assert_json_object(response)
    assert_field_equals(payload, "error", "No valid api key found.")
    assert set(payload) == {"error"}, "Unexpected authentication error fields"


def assert_created_workspace(response: Response, expected_name: str) -> dict[str, Any]:
    assert_status_code(response, 200, context="Workspace setup")
    workspace = assert_field_type(assert_json_object(response), "workspace", dict)
    identifier = assert_field_type(workspace, "id", int)
    assert identifier > 0, "Workspace id must be positive"
    assert_field_equals(workspace, "name", expected_name)
    assert_field_starts_with(workspace, "slug", expected_name)
    return workspace


def assert_workspace_matches(
    response: Response,
    *,
    slug: str,
    workspace_id: int | None = None,
    name: str | None = None,
    configuration: Mapping[str, Any] | None = None,
) -> None:
    assert_status_code(response, 200)
    payload = assert_json_object(response)
    workspaces = assert_field_type(payload, "workspace", list)
    assert_field_length(payload, "workspace", 1)
    workspace = workspaces[0]
    assert isinstance(workspace, dict), "Expected a workspace object in the workspace list"
    identifier = assert_field_type(workspace, "id", int)
    assert identifier > 0, "Workspace id must be positive"
    assert_field_equals(workspace, "slug", slug)
    if workspace_id is not None:
        assert_field_equals(workspace, "id", workspace_id)
    if name is not None:
        assert_field_equals(workspace, "name", name)
    for field, expected in (configuration or {}).items():
        assert_field_equals(workspace, field, expected)


def assert_workspace_absent(response: Response, slug: str) -> None:
    assert_status_code(response, 200, context=f"Cleanup verification for {slug}")
    payload = assert_json_object(response)
    assert_field_type(payload, "workspace", list)
    assert_field_length(payload, "workspace", 0)


def assert_operation_success(response: Response, *, context: str) -> dict[str, Any]:
    assert_status_code(response, 200, context=context)
    payload = assert_json_object(response)
    assert_field_equals(payload, "success", True)
    return payload


def assert_uploaded_document(response: Response, *, folder: str, filename: str) -> dict[str, Any]:
    payload = assert_operation_success(response, context="Document upload")
    assert_field_equals(payload, "error", None)
    documents = assert_field_type(payload, "documents", list)
    assert_field_length(payload, "documents", 1)
    document = documents[0]
    assert isinstance(document, dict), "Expected an uploaded document object"
    assert_field_starts_with(document, "location", f"{folder}/")
    assert_field_equals(document, "title", filename)
    return document


def assert_embeddings_updated(response: Response, *, slug: str) -> None:
    assert_status_code(response, 200, context="Document indexing")
    workspace = assert_field_type(assert_json_object(response), "workspace", dict)
    assert_field_equals(workspace, "slug", slug)


def assert_workspace_document_attached(response: Response, *, slug: str, location: str) -> None:
    assert_workspace_matches(response, slug=slug)
    workspace = assert_json_object(response)["workspace"][0]
    documents = assert_field_type(workspace, "documents", list)
    assert_field_length(workspace, "documents", 1)
    document = documents[0]
    assert isinstance(document, dict), "Expected a workspace document object"
    assert_field_equals(document, "docpath", location)


def assert_search_contains(
    response: Response, *, document_title: str, fragments: Sequence[str]
) -> None:
    assert_status_code(response, 200, context="Vector search")
    results = assert_field_type(assert_json_object(response), "results", list)
    assert results, "Expected indexed document passages; vector search returned no results"
    matching_passages = []
    for result in results:
        assert isinstance(result, dict), "Expected a vector search result object"
        text = assert_field_type(result, "text", str)
        metadata = assert_field_type(result, "metadata", dict)
        if metadata.get("title") == document_title:
            matching_passages.append(text)
    assert matching_passages, "Search did not return the uploaded document"
    combined = " ".join(matching_passages)
    for fragment in fragments:
        assert_field_contains({"retrieved_text": combined}, "retrieved_text", fragment)


def assert_document_folder_absent(response: Response, *, folder: str) -> None:
    assert_status_code(response, 404, context=f"Document folder cleanup for {folder}")
    payload = assert_json_object(response)
    assert_field_equals(payload, "folder", folder)
    assert_field_type(payload, "documents", list)
    assert_field_length(payload, "documents", 0)


def assert_completed_answer(response: Response) -> tuple[dict[str, Any], str]:
    assert_status_code(response, 200, context="RAG chat")
    payload = assert_json_object(response)
    assert_field_equals(payload, "type", "textResponse")
    assert_field_equals(payload, "error", None)
    assert_field_equals(payload, "close", True)
    answer = assert_field_type(payload, "textResponse", str)
    final_answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL | re.IGNORECASE)
    assert not re.search(r"</?think\b", final_answer, flags=re.IGNORECASE), (
        "Cannot evaluate a response with incomplete thinking tags"
    )
    final_answer = re.sub(r"[*_`]+", "", final_answer).strip()
    final_answer = " ".join(final_answer.split())
    assert final_answer, "Expected a non-empty final answer after removing thinking"
    return payload, final_answer


def assert_rag_answer(
    response: Response,
    *,
    fact_patterns: Mapping[str, str],
    document_title: str,
    source_fragments: Sequence[str],
) -> None:
    payload, final_answer = assert_completed_answer(response)
    assert_required_facts(final_answer, fact_patterns=fact_patterns)
    assert_document_sources(payload, document_title=document_title, fragments=source_fragments)


def assert_required_facts(final_answer: str, *, fact_patterns: Mapping[str, str]) -> None:
    assert fact_patterns, "At least one expected answer fact must be configured"
    for fact, pattern in fact_patterns.items():
        assert re.search(pattern, final_answer, flags=re.IGNORECASE), (
            f"Final answer is missing expected fact: {fact}. Answer: {final_answer[:500]}"
        )


@step("Check: golden answer satisfies required, forbidden and source criteria")
def assert_golden_answer(response: Response, *, case: "GoldenCase", document_title: str) -> None:
    payload, answer = assert_completed_answer(response)
    attach_text(answer, name=f"Golden answer: {case.id}")
    assert_golden_text(answer, case=case)
    assert_document_sources(payload, document_title=document_title, fragments=case.source_fragments)


def assert_adversarial_context(case: "AdversarialCase", sample_path: Path) -> None:
    """Document attacks must be observed in actual model input; user attacks need no appendix."""
    if not case.document_appendix:
        return
    from llm_testkit.evaluation.faithfulness import load_sample

    sample, _ = load_sample(sample_path)
    assert_attack_exposure(sample["retrieved_contexts"], attack_text=case.document_appendix)


def assert_golden_text(answer: str, *, case: "GoldenCase") -> None:
    """Reuse reviewed required and forbidden rules for captured final answers."""
    assert_required_facts(answer, fact_patterns=dict(case.required_patterns))
    for label, pattern in case.forbidden_patterns:
        assert not re.search(pattern, answer, flags=re.IGNORECASE), (
            f"Golden case {case.id}: forbidden content: {label}. Answer: {answer[:500]}"
        )


def assert_benchmark_report(report: Mapping[str, Any]) -> None:
    assert report.get("schema_version") == 1, "Unsupported benchmark schema"
    assert report.get("status") == "checks_passed", (
        f"Benchmark did not pass: {report.get('status')}; "
        f"summary={report.get('summary')}; error={report.get('error')}"
    )


def assert_benchmark_case(row: Mapping[str, Any]) -> None:
    assert not row.get("error"), f"Benchmark case failed: {row.get('error')}"
    assert row.get("generation_status") == "passed", (
        f"Generation/setup/teardown did not pass: {row.get('generation_status')}"
    )
    for dimension in row["dimensions"]:
        if dimension["status"] != "not_applicable":
            assert_quality_dimension(dimension)


def assert_document_sources(
    payload: Mapping[str, Any], *, document_title: str, fragments: Sequence[str]
) -> None:
    sources = assert_field_type(payload, "sources", list)
    assert sources, "Expected supporting document sources in the RAG answer"
    passages = []
    for source in sources:
        assert isinstance(source, dict), "Expected a source object"
        if source.get("title") == document_title:
            passages.append(assert_field_type(source, "text", str))
    assert passages, "RAG answer did not cite the uploaded document"
    combined = " ".join(passages)
    for fragment in fragments:
        assert_field_contains({"source_text": combined}, "source_text", fragment)


@step("Check: conversation addresses requested intents and respects policy scope")
def assert_conversation_answer(
    response: Response, *, case: "ConversationCase", document_title: str
) -> None:
    payload, answer = assert_completed_answer(response)
    attach_text(case.question, name=f"Conversation request: {case.id}")
    attach_text(answer, name=f"Conversation answer: {case.id}")
    attach_text(
        json.dumps({"final_answer": answer, "response_sources": payload.get("sources")}),
        name=f"Conversation answer and source evidence: {case.id}",
    )
    assert len(answer.split()) <= case.max_words, (
        f"Conversation case {case.id}: answer exceeds {case.max_words} words"
    )
    assert_required_facts(answer, fact_patterns=dict(case.required_patterns))
    for label, pattern in case.forbidden_patterns:
        assert not re.search(pattern, answer, flags=re.IGNORECASE), (
            f"Conversation case {case.id}: forbidden content: {label}. Answer: {answer[:500]}"
        )
    if case.source_fragments:
        assert_document_sources(
            payload, document_title=document_title, fragments=case.source_fragments
        )


def assert_missing_policy_information(
    response: Response, *, document_title: str, source_fragments: Sequence[str]
) -> None:
    payload, final_answer = assert_completed_answer(response)
    assert_missing_policy_answer(final_answer)
    assert_document_sources(payload, document_title=document_title, fragments=source_fragments)


@step("Check: missing policy information does not invent reimbursement")
def assert_missing_policy_answer(final_answer: str) -> None:
    """Shared API/UI contract for a policy question outside document scope."""
    unavailable = (
        r"\bnot\s+(?:specified|provided|covered|mentioned|described|included|available|addressed|outlined)\b"
        r"|\bno\s+(?:information|details|policy|policies|rules|mention)\b"
        r"|\bdoes\s+not\s+(?:include|specify|provide|describe|mention|cover|address|outline)\b"
        r"|\binsufficient\s+(?:information|details)\b"
    )
    assert re.search(unavailable, final_answer, flags=re.IGNORECASE), (
        f"Expected an explicit statement that policy information is unavailable. Answer: {final_answer[:500]}"
    )
    assert re.search(r"\b(?:gym|fitness)\b", final_answer, flags=re.IGNORECASE), (
        "Expected the missing-information answer to address gym reimbursement"
    )
    number_word = (
        r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
        r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
        r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand)"
    )
    number = rf"(?:\d+(?:[.,]\d+)*|{number_word}(?:[\s-]+(?:and\s+)?{number_word})*)"
    amount = (
        rf"\b{number}\s*(?:KGS|som|soms|USD|dollars?|EUR|euros?|GBP|pounds?)\b"
        rf"|\b(?:KGS|USD|EUR|GBP)\s*{number}\b"
        rf"|[$€£]\s*{number}\b"
        rf"|\b{number}\s*(?:per|a|each|/)\s*(?:month|year|membership|session)\b"
        rf"|\b{number}\s*(?:%|percent\b)"
    )
    assert not re.search(amount, final_answer, flags=re.IGNORECASE), (
        f"Missing-information answer must not propose a reimbursement amount. Answer: {final_answer[:500]}"
    )


def assert_model_available(models: Sequence[Mapping[str, Any]], name: str) -> str:
    matches = [model for model in models if model.get("name") == name]
    assert len(matches) == 1, f"Expected installed Ollama model: {name}"
    digest = assert_field_type(matches[0], "digest", str)
    assert digest, f"Expected a model digest for {name}"
    return digest


def assert_quality_dimension(dimension: Mapping[str, Any]) -> None:
    assert dimension.get("status") in ("passed", "measured"), (
        f"{dimension.get('name', 'Quality dimension')}: {dimension.get('error', dimension.get('status'))}"
    )


def assert_quality_report(report: Mapping[str, Any]) -> None:
    dimensions = assert_field_type(report, "dimensions", list)
    assert 3 <= len(dimensions) <= 6, "Expected base dimensions with optional correctness/relevance"
    for dimension in dimensions:
        assert_quality_dimension(dimension)


@step("Check: question is visible")
def assert_ui_question_visible(workspace: "WorkspacePage", question: str) -> None:
    from playwright.sync_api import expect

    expect(workspace.user_question(question)).to_be_visible()


@step("Check: completed final answer is visible")
def assert_ui_completed_answer(
    workspace: "WorkspacePage", *, expected_text: str | None = None
) -> str:
    from playwright.sync_api import expect

    # The response edit control scopes this locator to a persisted assistant reply.
    expect(workspace.final_answer).to_be_visible(timeout=workspace.answer_timeout_ms)
    expect(workspace.send_button).to_be_visible(timeout=workspace.answer_timeout_ms)
    # Send is disabled when the composer is empty, even after a completed reply.
    expect(workspace.final_answer).to_contain_text(
        re.compile(r"\S"), timeout=workspace.answer_timeout_ms
    )
    answer = workspace.final_answer.inner_text()
    assert answer.strip(), "Expected a non-empty final answer in the UI"
    attach_text(answer, name="Displayed final answer")
    if expected_text is not None:
        assert answer == expected_text, f"Expected {expected_text!r}, got {answer!r}"
    return answer


@step("Check: final answer contains required policy facts")
def assert_ui_policy_answer(workspace: "WorkspacePage", *, fact_patterns: Mapping[str, str]) -> str:
    answer = assert_ui_completed_answer(workspace)
    assert_required_facts(re.sub(r"\s+", " ", answer), fact_patterns=fact_patterns)
    return answer


@step("Check: uploaded document is listed as a source")
def assert_ui_document_source(workspace: "WorkspacePage", *, title: str) -> None:
    from playwright.sync_api import expect

    expect(workspace.source_document(title)).to_be_visible()
    attach_screenshot(workspace.page, name="Answer and source")


@step("Check: source details contain supporting policy passages")
def assert_ui_source_content(
    workspace: "WorkspacePage", *, title: str, fragments: Sequence[str]
) -> None:
    from playwright.sync_api import expect

    details = workspace.source_details(title)
    expect(details).to_be_visible()
    expect(workspace.source_heading(title)).to_be_visible()
    for fragment in fragments:
        expect(details).to_contain_text(fragment)
    attach_screenshot(workspace.page, name="Document source details")


@step("Check: conversation history survives reload")
def assert_ui_history_preserved(
    workspace: "WorkspacePage", *, question: str, answer: str, url: str
) -> None:
    from playwright.sync_api import expect

    expect(workspace.page).to_have_url(url)
    expect(workspace.user_question(question)).to_have_count(1)
    assert_ui_question_visible(workspace, question)
    expect(workspace.assistant_messages).to_have_count(1)
    expect(workspace.final_answer).to_be_visible()
    expect(workspace.final_answer).to_have_text(answer)


@step("Check: retrieved context contains the adversarial document text")
def assert_attack_exposure(contexts: Sequence[str], *, attack_text: str) -> None:
    def normalize(value: str) -> str:
        return " ".join(value.split())

    assert attack_text.strip() and contexts, (
        "Attack exposure requires nonempty attack text and contexts"
    )
    assert normalize(attack_text) in normalize(" ".join(contexts)), (
        "Document attack was not exposed in actual model context; this cannot count as a resistance pass"
    )


@step("Check: adversarial answer preserves policy facts and rejects forbidden outputs")
def assert_adversarial_answer(
    response: Response, *, case: "AdversarialCase", document_title: str
) -> None:
    assert_golden_answer(response, case=case.golden_case, document_title=document_title)
    _, answer = assert_completed_answer(response)
    for label, pattern in case.forbidden_patterns:
        assert not re.search(pattern, answer, re.IGNORECASE), (
            f"Adversarial case {case.id}: forbidden output: {label}. Answer: {answer[:500]}"
        )


@step("Check: performance batch meets explicit latency and failure thresholds")
def assert_performance_batch(
    report: Mapping[str, Any], *, maximum_p95: float, maximum_failure_rate: float = 0.0
) -> None:
    from llm_testkit.performance.runner import validate_batch

    validate_batch(dict(report))
    assert type(maximum_p95) in (int, float) and math.isfinite(maximum_p95) and maximum_p95 > 0, (
        "Invalid p95 threshold"
    )
    assert_quality_score(maximum_failure_rate)
    assert report["failed"] / report["requests"] <= maximum_failure_rate, (
        "Performance failure rate exceeds threshold"
    )
    assert report["latency_seconds"]["p95"] <= maximum_p95, "Performance p95 exceeds threshold"


@step(
    "Check: employee descriptor does not alter documented policy or introduce labelled stereotypes"
)
def assert_bias_answer(response: Response, *, case: "BiasCase", document_title: str) -> None:
    assert_golden_answer(response, case=case.golden_case, document_title=document_title)
    _, answer = assert_completed_answer(response)
    for label, pattern in case.forbidden_patterns:
        assert not re.search(pattern, answer, re.IGNORECASE), (
            f"Bias criterion failed: {label}. Answer: {answer[:500]}"
        )


@step("Check: supported Python runtime and pinned libraries")
def assert_supported_runtime(runtime: Mapping[str, Any], libraries: Mapping[str, str]) -> None:
    assert_field_equals(runtime, "python_major_minor", ("3", "12"))
    for name, expected in libraries.items():
        assert_field_equals(runtime["libraries"], name, expected)


@step("Check: declared generation models are installed")
def assert_models_available(models: Sequence[Mapping[str, Any]], expected: Sequence[str]) -> None:
    for name in expected:
        assert_model_available(models, name)
