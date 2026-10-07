"""Function-scoped resources with cleanup registered immediately after creation."""

from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from llm_testkit import assertions
from llm_testkit.clients.anythingllm_client import AnythingLLMClient
from llm_testkit.config import Settings


@pytest.fixture
def temporary_workspace(
    request: pytest.FixtureRequest,
    authenticated_anythingllm_api: AnythingLLMClient,
    workspace_configuration: dict[str, Any],
) -> dict[str, Any]:
    name = f"automation-{uuid4().hex}"
    response = authenticated_anythingllm_api.create_workspace(name, workspace_configuration)
    assertions.assert_status_code(response, 200, context="Workspace setup")
    workspace = assertions.assert_field_type(
        assertions.assert_json_object(response), "workspace", dict
    )
    slug = assertions.assert_field_type(workspace, "slug", str)
    assert slug, "Created workspace must have a nonempty slug for cleanup"

    def cleanup() -> None:
        deletion = authenticated_anythingllm_api.delete_workspace(slug)
        assertions.assert_status_code(deletion, 200, context=f"Cleanup for {slug}")
        lookup = authenticated_anythingllm_api.get_workspace(slug)
        assertions.assert_workspace_absent(lookup, slug)

    request.addfinalizer(cleanup)
    return assertions.assert_created_workspace(response, expected_name=name)


@pytest.fixture
def document_folder(
    request: pytest.FixtureRequest,
    authenticated_anythingllm_api: AnythingLLMClient,
    temporary_workspace: dict[str, Any],
    settings: Settings,
) -> str:
    folder = temporary_workspace["slug"]
    creation = authenticated_anythingllm_api.create_document_folder(folder)
    assertions.assert_status_code(creation, 200, context="Test document folder setup")

    def cleanup() -> None:
        deletion = authenticated_anythingllm_api.delete_document_folder(
            folder, timeout=settings.document_timeout
        )
        assertions.assert_operation_success(deletion, context=f"Document cleanup for {folder}")
        lookup = authenticated_anythingllm_api.get_document_folder(folder)
        assertions.assert_document_folder_absent(lookup, folder=folder)

    request.addfinalizer(cleanup)
    assertions.assert_operation_success(creation, context="Test document folder setup")
    return folder


@pytest.fixture
def uploaded_policy_document(
    authenticated_anythingllm_api: AnythingLLMClient,
    document_folder: str,
    settings: Settings,
    policy_file: Path,
) -> dict[str, Any]:
    filename = f"{document_folder}-company-policy.txt"
    response = authenticated_anythingllm_api.upload_document(
        policy_file, document_folder, filename=filename, timeout=settings.document_timeout
    )
    return assertions.assert_uploaded_document(response, folder=document_folder, filename=filename)


@pytest.fixture
def indexed_workspace(
    authenticated_anythingllm_api: AnythingLLMClient,
    uploaded_policy_document: dict[str, Any],
    temporary_workspace: dict[str, Any],
    settings: Settings,
) -> dict[str, Any]:
    response = authenticated_anythingllm_api.add_workspace_documents(
        temporary_workspace["slug"],
        [uploaded_policy_document["location"]],
        timeout=settings.document_timeout,
    )
    assertions.assert_embeddings_updated(response, slug=temporary_workspace["slug"])
    return temporary_workspace
