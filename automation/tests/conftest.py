import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

pytest.register_assert_rewrite("llm_testkit.assertions")

from llm_testkit import assertions
from llm_testkit.clients.anythingllm_client import AnythingLLMClient
from llm_testkit.config import Settings
from llm_testkit.core.http_client import HttpClient


@pytest.fixture(scope="session")
def settings() -> Settings:
    key_file = Path(__file__).resolve().parents[2] / ".runtime" / "anythingllm-api-key"
    return Settings.from_env(default_api_key_file=key_file)


@pytest.fixture
def http_client(settings: Settings) -> Iterator[HttpClient]:
    client = HttpClient(base_url=settings.base_url, timeout=settings.http_timeout)
    try:
        yield client
    finally:
        client.close()


@pytest.fixture
def anythingllm_api(http_client: HttpClient) -> AnythingLLMClient:
    return AnythingLLMClient(http_client)


@pytest.fixture
def authenticated_anythingllm_api(
    http_client: HttpClient, settings: Settings
) -> AnythingLLMClient:
    if not settings.api_key:
        pytest.fail(
            "Developer API key is missing. Configure ANYTHINGLLM_API_KEY or "
            "ANYTHINGLLM_API_KEY_FILE; see docs/step-03-developer-api.md.",
            pytrace=False,
        )
    return AnythingLLMClient(http_client, api_key=settings.api_key)


@pytest.fixture
def workspace_configuration() -> dict[str, Any]:
    configuration_file = Path(__file__).resolve().parents[2] / "config" / "workspace.json"
    return json.loads(configuration_file.read_text(encoding="utf-8"))


@pytest.fixture
def temporary_workspace(
    authenticated_anythingllm_api: AnythingLLMClient,
    workspace_configuration: dict[str, Any],
) -> Iterator[dict[str, Any]]:
    name = f"automation-{uuid4().hex}"
    response = authenticated_anythingllm_api.create_workspace(name, workspace_configuration)
    workspace = assertions.assert_created_workspace(response, expected_name=name)
    slug = workspace["slug"]

    try:
        yield workspace
    finally:
        deletion = authenticated_anythingllm_api.delete_workspace(slug)
        assertions.assert_status_code(deletion, 200, context=f"Cleanup for {slug}")
        lookup = authenticated_anythingllm_api.get_workspace(slug)
        assertions.assert_workspace_absent(lookup, slug)


@pytest.fixture
def uploaded_policy_document(
    authenticated_anythingllm_api: AnythingLLMClient,
    temporary_workspace: dict[str, Any],
    settings: Settings,
) -> Iterator[dict[str, Any]]:
    folder = temporary_workspace["slug"]
    creation = authenticated_anythingllm_api.create_document_folder(folder)
    assertions.assert_operation_success(creation, context="Test document folder setup")
    try:
        path = Path(__file__).resolve().parents[1] / "test_data" / "company-policy.txt"
        filename = f"{folder}-company-policy.txt"
        response = authenticated_anythingllm_api.upload_document(
            path, folder, filename=filename, timeout=settings.document_timeout
        )
        document = assertions.assert_uploaded_document(response, folder=folder, filename=filename)
        yield document
    finally:
        deletion = authenticated_anythingllm_api.delete_document_folder(
            folder, timeout=settings.document_timeout
        )
        assertions.assert_operation_success(deletion, context=f"Document cleanup for {folder}")
        lookup = authenticated_anythingllm_api.get_document_folder(folder)
        assertions.assert_document_folder_absent(lookup, folder=folder)


@pytest.fixture
def indexed_workspace(
    authenticated_anythingllm_api: AnythingLLMClient,
    uploaded_policy_document: dict[str, Any],
    temporary_workspace: dict[str, Any],
    settings: Settings,
) -> dict[str, Any]:
    response = authenticated_anythingllm_api.add_workspace_documents(
        temporary_workspace["slug"], [uploaded_policy_document["location"]],
        timeout=settings.document_timeout,
    )
    assertions.assert_embeddings_updated(response, slug=temporary_workspace["slug"])
    return temporary_workspace
