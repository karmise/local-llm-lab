"""Reusable response checks; transport and API clients do not assert outcomes."""

import re
from collections.abc import Mapping, Sequence, Sized
from typing import Any, TypeVar

from requests import Response

T = TypeVar("T")


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
    assert len(value) == expected, (
        f"Field {field}: expected length {expected}, got {len(value)}"
    )


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


def assert_uploaded_document(
    response: Response, *, folder: str, filename: str
) -> dict[str, Any]:
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


def assert_rag_answer(
    response: Response, *, fact_patterns: Mapping[str, str],
    document_title: str, source_fragments: Sequence[str],
) -> None:
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
    assert fact_patterns, "At least one expected answer fact must be configured"
    for fact, pattern in fact_patterns.items():
        assert re.search(pattern, final_answer, flags=re.IGNORECASE), (
            f"Final answer is missing expected fact: {fact}. Answer: {final_answer[:500]}"
        )

    sources = assert_field_type(payload, "sources", list)
    assert sources, "Expected supporting document sources in the RAG answer"
    passages = []
    for source in sources:
        assert isinstance(source, dict), "Expected a source object"
        if source.get("title") == document_title:
            passages.append(assert_field_type(source, "text", str))
    assert passages, "RAG answer did not cite the uploaded document"
    combined = " ".join(passages)
    for fragment in source_fragments:
        assert_field_contains({"source_text": combined}, "source_text", fragment)
