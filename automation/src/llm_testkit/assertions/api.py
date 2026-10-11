"""AnythingLLM and Ollama API responses: service health, authentication, resources and installed models."""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from requests import Response

from llm_testkit.reporting.steps import step

if TYPE_CHECKING:
    pass
from llm_testkit.assertions.fields import (
        assert_field_contains, assert_field_equals, assert_field_length, assert_field_starts_with, assert_field_type,
        assert_json_object, assert_status_code)


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
        response: Response, *, slug: str, workspace_id: int | None = None, name: str | None = None,
        configuration: Mapping[str, Any] | None = None) -> None:
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


def assert_search_contains(response: Response, *, document_title: str, fragments: Sequence[str]) -> None:
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


def assert_model_available(models: Sequence[Mapping[str, Any]], name: str) -> str:
    matches = [model for model in models if model.get("name") == name]
    assert len(matches) == 1, f"Expected installed Ollama model: {name}"
    digest = assert_field_type(matches[0], "digest", str)
    assert digest, f"Expected a model digest for {name}"
    return digest


@step("Check: declared generation models are installed")
def assert_models_available(models: Sequence[Mapping[str, Any]], expected: Sequence[str]) -> None:
    for name in expected:
        assert_model_available(models, name)


@step("Check: supported Python runtime and pinned libraries")
def assert_supported_runtime(runtime: Mapping[str, Any], libraries: Mapping[str, str]) -> None:
    assert_field_equals(runtime, "python_major_minor", ("3", "12"))
    for name, expected in libraries.items():
        assert_field_equals(runtime["libraries"], name, expected)
