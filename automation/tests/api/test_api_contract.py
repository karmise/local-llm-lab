"""Developer API contract: protected routes reject missing credentials, and unknown or malformed requests change
nothing. None of these requests reaches a model."""

from uuid import uuid4

import pytest

from llm_testkit import assertions
from llm_testkit.config import Settings
from llm_testkit.core.http_client import HttpClient
from llm_testkit.reporting.steps import title

pytestmark = pytest.mark.api

MISSING = "automation-missing-{}"

PROTECTED_ROUTES = [
        pytest.param("GET", "workspaces", None, id="list-workspaces"),
        pytest.param("GET", "workspace/{slug}", None, id="get-workspace"),
        pytest.param("POST", "workspace/new", {"name": "unauthorised"}, id="create-workspace"),
        pytest.param("POST", "workspace/{slug}/chat", {
        "message": "hi",
        "mode": "query"}, id="chat"),
        pytest.param("POST", "workspace/{slug}/vector-search", {"query": "leave"}, id="vector-search"),
        pytest.param("DELETE", "workspace/{slug}", None, id="delete-workspace"),
        pytest.param("GET", "documents", None, id="list-documents"),
        pytest.param("POST", "document/create-folder", {"name": "unauthorised"}, id="create-folder")]


@pytest.fixture
def missing_slug() -> str:
    """A workspace and folder name that does not exist."""
    return MISSING.format(uuid4().hex)


def developer_request(http_client: HttpClient, method: str, path: str, *, key: str | None, headers=None, **kwargs):
    """A raw developer API request, for shapes the client never sends."""
    authorization = {"Authorization": f"Bearer {key}"} if key else {}
    return http_client.request(method, f"/api/v1/{path}", headers=authorization | (headers or {}), **kwargs)


def workspace_slugs(http_client: HttpClient, settings: Settings) -> list[str]:
    response = developer_request(http_client, "GET", "workspaces", key=settings.api_key)
    assertions.assert_status_code(response, 200, context="Workspace list")
    workspaces = assertions.assert_field_type(assertions.assert_json_object(response), "workspaces", list)
    return sorted(workspace["slug"] for workspace in workspaces)


@pytest.mark.parametrize(
        "key", [pytest.param(None, id="missing-key"),
        pytest.param("invalid-test-key", id="invalid-key")])
@pytest.mark.parametrize(("method", "path", "body"), PROTECTED_ROUTES)
@title("Every protected developer route rejects a missing or invalid API key with the documented error [{param_id}]")
def test_protected_route_rejects_unauthenticated_request(http_client, missing_slug, method, path, body, key):
    response = developer_request(http_client, method, path.format(slug=missing_slug), key=key, json=body)

    assertions.assert_api_key_rejected(response)


@title("Unauthenticated create requests leave no workspace behind")
def test_unauthenticated_create_changes_nothing(authenticated_anythingllm_api, http_client, settings):
    before = workspace_slugs(http_client, settings)

    developer_request(http_client, "POST", "workspace/new", key=None, json={"name": "unauthorised"})

    assert workspace_slugs(http_client, settings) == before


@title("Looking up an unknown workspace returns an empty list, not an error")
def test_unknown_workspace_lookup(authenticated_anythingllm_api, missing_slug):
    response = authenticated_anythingllm_api.get_workspace(missing_slug)

    assertions.assert_workspace_absent(response, missing_slug)


@title("Chat with an unknown workspace is aborted before any generation and names the workspace")
def test_chat_with_unknown_workspace_is_aborted(authenticated_anythingllm_api, missing_slug):
    response = authenticated_anythingllm_api.chat(missing_slug, "What is the leave policy?", timeout=10)

    assertions.assert_status_code(response, 400, context="Chat with unknown workspace")
    payload = assertions.assert_json_object(response)
    assertions.assert_field_equals(payload, "type", "abort")
    assertions.assert_field_equals(payload, "textResponse", None)
    assertions.assert_field_equals(payload, "close", True)
    assertions.assert_field_length(payload, "sources", 0)
    assertions.assert_field_contains(payload, "error", missing_slug)


@title("Vector search on an unknown workspace is rejected with a message naming it")
def test_search_unknown_workspace_is_rejected(authenticated_anythingllm_api, missing_slug):
    response = authenticated_anythingllm_api.search_workspace(missing_slug, "leave", timeout=10)

    assertions.assert_status_code(response, 400, context="Search unknown workspace")
    assertions.assert_field_equals(
            assertions.assert_json_object(response), "message", f"Workspace {missing_slug} is not a valid workspace.")


@pytest.mark.parametrize(
        "operation", [
        pytest.param(lambda api, slug: api.delete_workspace(slug), id="delete"),
        pytest.param(lambda api, slug: api.update_workspace(slug, {"openAiTemp": 0.1}), id="update"),
        pytest.param(lambda api, slug: api.add_workspace_documents(slug, [], timeout=10), id="index")])
@title("Changing an unknown workspace is rejected and creates nothing [{param_id}]")
def test_change_unknown_workspace_is_rejected(
        authenticated_anythingllm_api, http_client, settings, missing_slug, operation):
    before = workspace_slugs(http_client, settings)

    response = operation(authenticated_anythingllm_api, missing_slug)

    assertions.assert_status_code(response, 400, context="Change unknown workspace")
    assert workspace_slugs(http_client, settings) == before


@title("Malformed JSON when creating a workspace is rejected and creates nothing")
def test_malformed_workspace_request_is_rejected(authenticated_anythingllm_api, http_client, settings):
    before = workspace_slugs(http_client, settings)

    response = developer_request(
            http_client, "POST", "workspace/new", key=settings.api_key, data="not-json",
            headers={"Content-Type": "application/json"})

    assertions.assert_status_code(response, 400, context="Malformed workspace request")
    assert workspace_slugs(http_client, settings) == before


@title("An unknown document folder is reported as absent")
def test_unknown_document_folder_is_absent(authenticated_anythingllm_api, missing_slug):
    response = authenticated_anythingllm_api.get_document_folder(missing_slug)

    assertions.assert_document_folder_absent(response, folder=missing_slug)


@title("Removing an unknown document folder succeeds, so cleanup is idempotent")
def test_remove_unknown_folder_is_idempotent(authenticated_anythingllm_api, missing_slug):
    response = authenticated_anythingllm_api.delete_document_folder(missing_slug, timeout=10)

    assertions.assert_operation_success(response, context="Remove unknown folder")


@title("An upload without a file is rejected and stores no document")
def test_upload_without_file_is_rejected(authenticated_anythingllm_api, http_client, settings, missing_slug):
    response = developer_request(http_client, "POST", f"document/upload/{missing_slug}", key=settings.api_key)

    # AnythingLLM 1.16.2 answers 500 rather than 4xx: rejected, but reported as a server error.
    assert response.status_code >= 400, f"Upload without a file must be rejected, got HTTP {response.status_code}"
    assertions.assert_document_folder_absent(
            authenticated_anythingllm_api.get_document_folder(missing_slug), folder=missing_slug)


@title("An unknown API route answers with the web app page, which JSON checks reject")
def test_unknown_route_is_not_json(http_client, settings):
    response = developer_request(http_client, "GET", "no-such-endpoint", key=settings.api_key)

    with pytest.raises(AssertionError, match="Response body is not valid JSON"):
        assertions.assert_json_object(response)
