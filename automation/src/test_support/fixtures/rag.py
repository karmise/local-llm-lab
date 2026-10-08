"""Model selection, generation metadata and optional context capture."""

import hashlib
import json
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

import pytest
from requests import Response

from llm_testkit import assertions
from llm_testkit.clients.anythingllm_client import AnythingLLMClient
from llm_testkit.clients.ollama_client import OllamaClient
from llm_testkit.config import Settings
from llm_testkit.core.http_client import HttpClient
from llm_testkit.datasets.prompts import PromptVariant, load_prompt_catalog
from llm_testkit.observation.evaluation_sample import build_sample, write_sample


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
def prompt_variant() -> PromptVariant | None:
    return None


@pytest.fixture
def workspace_configuration(
        workspace_template: dict[str, Any], generation_model: str, capture_id: str | None,
        prompt_variant: PromptVariant | None) -> dict[str, Any]:
    configuration = {**deepcopy(workspace_template), "chatModel": generation_model}
    if prompt_variant is not None:
        configuration["openAiPrompt"] = prompt_variant.prompt
    if capture_id:
        configuration["openAiPrompt"] += f"\n[LLM_TESTKIT_CAPTURE:{capture_id}]"
    return configuration


@pytest.fixture(scope="session")
def ollama_models(settings: Settings) -> list[dict[str, Any]]:
    with HttpClient(settings.ollama_base_url, settings.http_timeout) as http:
        response = OllamaClient(http).list_models()
        assertions.assert_status_code(response, 200, context="Ollama model catalog")
        return assertions.assert_field_type(assertions.assert_json_object(response), "models", list)


@pytest.fixture
def generation_model_digest(generation_model: str, ollama_models: list[dict[str, Any]]) -> str:
    return assertions.assert_model_available(ollama_models, generation_model)


@pytest.fixture
def rag_environment(
        generation_model: str, rag_iteration: int, prompt_variant: PromptVariant | None, automation_root: Path,
        workspace_template: dict[str, Any], generation_model_digest: str,
        authenticated_anythingllm_api: AnythingLLMClient, indexed_workspace: dict[str,
        Any], workspace_configuration: dict[str, Any], policy_file: Path, record_property: Callable[[str, object],
        None]) -> None:
    response = authenticated_anythingllm_api.get_workspace(indexed_workspace["slug"])
    assertions.assert_workspace_matches(response, slug=indexed_workspace["slug"], configuration=workspace_configuration)
    if prompt_variant is not None:
        catalog = load_prompt_catalog(automation_root / "test_data/prompt-variants.json")
        baseline = next(v for v in catalog.variants if v.id == catalog.baseline)
        if (baseline.prompt != workspace_template["openAiPrompt"] or prompt_variant not in catalog.variants):
            pytest.fail("Prompt catalog differs from collected variants or workspace baseline", pytrace=False)
        record_property("prompt_id", prompt_variant.id)
        record_property("prompt_version", prompt_variant.version)
        record_property("prompt_sha256", prompt_variant.sha256)
        record_property("prompt_catalog_sha256", catalog.sha256)
    record_property("generation_model", generation_model)
    record_property("rag_iteration", rag_iteration)
    record_property("model_digest", generation_model_digest)
    record_property("policy_sha256", hashlib.sha256(policy_file.read_bytes()).hexdigest())
    record_property("workspace_configuration", json.dumps(workspace_configuration, sort_keys=True))
    record_property("thinking_mode", "Ollama/model default; not explicitly controlled")


@pytest.fixture
def rag_chat(
        request: pytest.FixtureRequest, rag_environment: None, automation_root: Path,
        authenticated_anythingllm_api: AnythingLLMClient, indexed_workspace: dict[str,
        Any], settings: Settings, capture_id: str | None, generation_model: str, rag_iteration: int,
        generation_model_digest: str, workspace_configuration: dict[str,
        Any], policy_file: Path, record_property: Callable[[str, object], None]) -> Callable[[str, str], Response]:
    def chat(question: str, reference: str) -> Response:
        started = perf_counter()
        response = authenticated_anythingllm_api.chat(indexed_workspace["slug"], question, timeout=settings.llm_timeout)
        answer_request_seconds = perf_counter() - started
        record_property("answer_request_seconds", answer_request_seconds)
        if capture_id:
            root = automation_root.parent
            files = list((root / ".runtime" / "ollama-capture").glob(f"{capture_id}-*.json"))
            assertions.assert_field_length({"captures": files}, "captures", 1)
            capture = json.loads(files[0].read_text(encoding="utf-8"))
            payload, answer = assertions.assert_completed_answer(response)
            sample = build_sample(
                    capture, question=question, answer=answer, reference=reference, expected_model=generation_model,
                    capture_id=capture_id)
            sample["response_sources"] = payload.get("sources", [])
            sample["metadata"] = {
                    "workspace_slug": indexed_workspace["slug"],
                    "model_digest": generation_model_digest,
                    "policy_sha256": hashlib.sha256(policy_file.read_bytes()).hexdigest(),
                    "workspace_configuration": workspace_configuration,
                    "rag_iteration": rag_iteration,
                    "answer_request_seconds": answer_request_seconds,
                    "thinking_mode": "Ollama/model default; not explicitly controlled"}
            sample["metadata"].update((name, value) for name, value in request.node.user_properties
                    if name.startswith(("golden_", "prompt_", "adversarial_", "bias_", "qualification_")) or name in (
                    "requirement_ids", "test_node_id", "test_source_sha256", "framework_source_sha256"))
            path = automation_root / "reports" / "rag-samples" / f"{capture_id}.json"
            write_sample(path, sample)
            record_property("evaluation_sample", str(path))
        return response

    return chat


@pytest.fixture
def captured_sample_path(capture_id: str | None, automation_root: Path) -> Path:
    if capture_id is None:
        pytest.fail("A captured sample requires --capture-rag", pytrace=False)
    return automation_root / "reports/rag-samples" / f"{capture_id}.json"
