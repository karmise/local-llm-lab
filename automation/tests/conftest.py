import hashlib
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from requests import Response

pytest.register_assert_rewrite("llm_testkit.assertions")

from llm_testkit import assertions
from llm_testkit.clients.anythingllm_client import AnythingLLMClient
from llm_testkit.clients.ollama_client import OllamaClient
from llm_testkit.config import Settings
from llm_testkit.core.http_client import HttpClient
from llm_testkit.observation.evaluation_sample import build_sample, write_sample

DEFAULT_RAG_MODELS = ("qwen3.5:4b", "qwen2.5:7b")


def _positive_repeat(value: str) -> int:
    try:
        count = int(value)
    except ValueError:
        raise pytest.UsageError("--rag-repeat must be a positive integer") from None
    if count < 1:
        raise pytest.UsageError("--rag-repeat must be a positive integer")
    return count


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--run-live-quality", action="store_true", help="Enable one live RAG-to-Allure scenario (three model calls maximum).")
    parser.addoption("--judge-model", default="qwen3.5:4b", help="Local judge model for the live quality scenario.")
    parser.addoption("--quality-sample", type=Path, help="Captured sample for an offline quality report.")
    parser.addoption("--faithfulness-report", type=Path, help="Existing judge report bound to the sample checksum.")
    parser.addoption(
        "--capture-rag", action="store_true", default=False,
        help="Save actual model messages and evaluation samples; requires compose.capture.yaml.",
    )
    parser.addoption(
        "--rag-model", action="append", dest="rag_models", default=None,
        help="Generation model for RAG tests; repeat to select multiple models.",
    )
    parser.addoption(
        "--rag-repeat", type=_positive_repeat, default=1,
        help="Independent repetitions per RAG scenario and model (default: 1).",
    )


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if metafunc.definition.get_closest_marker("rag") and "generation_model" in metafunc.fixturenames:
        defaults = ("qwen3.5:4b",) if metafunc.definition.get_closest_marker("live_quality") else DEFAULT_RAG_MODELS
        models = list(dict.fromkeys(metafunc.config.getoption("rag_models") or defaults))
        count = metafunc.config.getoption("rag_repeat")
        cases = [(model, iteration) for model in models for iteration in range(1, count + 1)]
        ids = [model if count == 1 else f"{model}-run-{iteration}" for model, iteration in cases]
        metafunc.parametrize(("generation_model", "rag_iteration"), cases, ids=ids)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    live_items = [item for item in items if item.get_closest_marker("live_quality")]
    if not config.getoption("run_live_quality"):
        for item in live_items:
            item.add_marker(pytest.mark.skip(reason="Enable explicitly with --run-live-quality"))
        return
    if live_items:
        if not config.getoption("capture_rag"):
            raise pytest.UsageError("Live quality requires --capture-rag and the capture Compose overlay")
        if len(live_items) != 1:
            raise pytest.UsageError("Live quality is limited to one scenario/model/repetition per run")
        if not config.getoption("allure_report_dir", default=None):
            raise pytest.UsageError("Live quality requires the reporting extra and --alluredir")


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


@pytest.fixture(scope="session")
def workspace_template() -> dict[str, Any]:
    configuration_file = Path(__file__).resolve().parents[2] / "config" / "workspace.json"
    return json.loads(configuration_file.read_text(encoding="utf-8"))


@pytest.fixture
def generation_model(workspace_template: dict[str, Any]) -> str:
    return workspace_template["chatModel"]


@pytest.fixture
def rag_iteration() -> int:
    return 1


@pytest.fixture
def capture_id(request: pytest.FixtureRequest) -> str | None:
    return uuid4().hex if request.config.getoption("capture_rag") else None


@pytest.fixture
def workspace_configuration(
    workspace_template: dict[str, Any], generation_model: str, capture_id: str | None,
) -> dict[str, Any]:
    configuration = {**workspace_template, "chatModel": generation_model}
    if capture_id:
        configuration["openAiPrompt"] += f"\n[LLM_TESTKIT_CAPTURE:{capture_id}]"
    return configuration


@pytest.fixture
def policy_file() -> Path:
    return Path(__file__).resolve().parents[1] / "test_data" / "company-policy.txt"


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
    policy_file: Path,
) -> Iterator[dict[str, Any]]:
    folder = temporary_workspace["slug"]
    creation = authenticated_anythingllm_api.create_document_folder(folder)
    assertions.assert_operation_success(creation, context="Test document folder setup")
    try:
        filename = f"{folder}-company-policy.txt"
        response = authenticated_anythingllm_api.upload_document(
            policy_file, folder, filename=filename, timeout=settings.document_timeout
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


@pytest.fixture(scope="session")
def ollama_models(settings: Settings) -> list[dict[str, Any]]:
    http = HttpClient(settings.ollama_base_url, settings.http_timeout)
    try:
        response = OllamaClient(http).list_models()
        assertions.assert_status_code(response, 200, context="Ollama model catalog")
        return assertions.assert_field_type(assertions.assert_json_object(response), "models", list)
    finally:
        http.close()


@pytest.fixture
def rag_environment(
    generation_model: str,
    rag_iteration: int,
    ollama_models: list[dict[str, Any]],
    authenticated_anythingllm_api: AnythingLLMClient,
    indexed_workspace: dict[str, Any],
    workspace_configuration: dict[str, Any],
    policy_file: Path,
    record_property: Callable[[str, object], None],
) -> None:
    digest = assertions.assert_model_available(ollama_models, generation_model)
    response = authenticated_anythingllm_api.get_workspace(indexed_workspace["slug"])
    assertions.assert_workspace_matches(
        response, slug=indexed_workspace["slug"], configuration=workspace_configuration
    )
    record_property("generation_model", generation_model)
    record_property("rag_iteration", rag_iteration)
    record_property("model_digest", digest)
    record_property("policy_sha256", hashlib.sha256(policy_file.read_bytes()).hexdigest())
    record_property("workspace_configuration", json.dumps(workspace_configuration, sort_keys=True))
    record_property("thinking_mode", "Ollama/model default; not explicitly controlled")


@pytest.fixture
def rag_chat(
    authenticated_anythingllm_api: AnythingLLMClient,
    indexed_workspace: dict[str, Any],
    settings: Settings,
    capture_id: str | None,
    generation_model: str,
    rag_iteration: int,
    ollama_models: list[dict[str, Any]],
    workspace_configuration: dict[str, Any],
    policy_file: Path,
    record_property: Callable[[str, object], None],
) -> Callable[[str, str], Response]:
    def chat(question: str, reference: str) -> Response:
        response = authenticated_anythingllm_api.chat(
            indexed_workspace["slug"], question, timeout=settings.llm_timeout,
        )
        if capture_id:
            root = Path(__file__).resolve().parents[2]
            files = list((root / ".runtime" / "ollama-capture").glob(f"{capture_id}-*.json"))
            assertions.assert_field_length({"captures": files}, "captures", 1)
            capture = json.loads(files[0].read_text(encoding="utf-8"))
            payload, answer = assertions.assert_completed_answer(response)
            sample = build_sample(
                capture, question=question, answer=answer, reference=reference,
                expected_model=generation_model, capture_id=capture_id,
            )
            sample["response_sources"] = payload.get("sources", [])
            sample["metadata"] = {
                "workspace_slug": indexed_workspace["slug"],
                "model_digest": assertions.assert_model_available(ollama_models, generation_model),
                "policy_sha256": hashlib.sha256(policy_file.read_bytes()).hexdigest(),
                "workspace_configuration": workspace_configuration,
                "rag_iteration": rag_iteration,
                "thinking_mode": "Ollama/model default; not explicitly controlled",
            }
            path = root / "automation" / "reports" / "rag-samples" / f"{capture_id}.json"
            write_sample(path, sample)
            record_property("evaluation_sample", str(path))
        return response

    return chat


@pytest.fixture
def captured_sample_path(capture_id: str | None) -> Path:
    if capture_id is None:
        pytest.fail("A captured sample requires --capture-rag", pytrace=False)
    return Path(__file__).resolve().parents[1] / "reports/rag-samples" / f"{capture_id}.json"
