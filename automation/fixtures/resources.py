"""Function-scoped resources with cleanup registered immediately after creation."""

from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

import assertions
from api.anythingllm import AnythingLLMClient
from config import Settings


@pytest.fixture
def missing_slug() -> str:
    """A workspace and document folder name that does not exist."""
    return f"automation-missing-{uuid4().hex}"


@pytest.fixture
def workspace_slugs(authenticated_anythingllm_api: AnythingLLMClient) -> Callable[[], list[str]]:
    """Reads the current workspace slugs, to show that a rejected request created nothing."""
    def slugs() -> list[str]:
        response = authenticated_anythingllm_api.workspaces.get_all()
        assertions.assert_status_code(response, 200, context="Workspace list")
        workspaces = assertions.assert_field_type(assertions.assert_json_object(response), "workspaces", list)
        return sorted(workspace["slug"] for workspace in workspaces)

    return slugs


@pytest.fixture
def temporary_workspace(
        request: pytest.FixtureRequest, authenticated_anythingllm_api: AnythingLLMClient,
        workspace_configuration: dict[str, Any]) -> dict[str, Any]:
    name = f"automation-{uuid4().hex}"
    response = authenticated_anythingllm_api.workspaces.create(name, workspace_configuration)
    assertions.assert_status_code(response, 200, context="Workspace setup")
    workspace = assertions.assert_field_type(assertions.assert_json_object(response), "workspace", dict)
    slug = assertions.assert_field_type(workspace, "slug", str)
    assert slug, "Created workspace must have a nonempty slug for cleanup"

    def cleanup() -> None:
        deletion = authenticated_anythingllm_api.workspaces.delete(slug)
        assertions.assert_status_code(deletion, 200, context=f"Cleanup for {slug}")
        lookup = authenticated_anythingllm_api.workspaces.get(slug)
        assertions.assert_workspace_absent(lookup, slug)

    request.addfinalizer(cleanup)
    return assertions.assert_created_workspace(response, expected_name=name)


@pytest.fixture
def document_folder(
        request: pytest.FixtureRequest, authenticated_anythingllm_api: AnythingLLMClient, temporary_workspace: dict[str,
        Any], settings: Settings) -> str:
    folder = temporary_workspace["slug"]
    creation = authenticated_anythingllm_api.documents.create_folder(folder)
    assertions.assert_status_code(creation, 200, context="Test document folder setup")

    def cleanup() -> None:
        deletion = authenticated_anythingllm_api.documents.delete_folder(folder, timeout=settings.document_timeout)
        assertions.assert_operation_success(deletion, context=f"Document cleanup for {folder}")
        lookup = authenticated_anythingllm_api.documents.get_folder(folder)
        assertions.assert_document_folder_absent(lookup, folder=folder)

    request.addfinalizer(cleanup)
    assertions.assert_operation_success(creation, context="Test document folder setup")
    return folder


@pytest.fixture
def uploaded_policy_document(
        authenticated_anythingllm_api: AnythingLLMClient, document_folder: str, settings: Settings,
        policy_file: Path) -> dict[str, Any]:
    # A unique file name keeps concurrent uploads apart; the title the model sees is the same in every run.
    response = authenticated_anythingllm_api.documents.upload(
            policy_file, document_folder, filename=f"{document_folder}-{policy_file.name}", title=policy_file.name,
            timeout=settings.document_timeout)
    return assertions.assert_uploaded_document(response, folder=document_folder, title=policy_file.name)


@pytest.fixture
def indexed_workspace(
        authenticated_anythingllm_api: AnythingLLMClient, uploaded_policy_document: dict[str, Any],
        temporary_workspace: dict[str, Any], settings: Settings) -> dict[str, Any]:
    response = authenticated_anythingllm_api.workspaces.add_documents(
            temporary_workspace["slug"], [uploaded_policy_document["location"]], timeout=settings.document_timeout)
    assertions.assert_embeddings_updated(response, slug=temporary_workspace["slug"])
    return temporary_workspace
