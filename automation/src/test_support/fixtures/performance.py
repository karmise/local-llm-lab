"""Prepare bounded workloads and reporting context outside performance scenarios."""

import hashlib

import pytest

from llm_testkit import assertions
from llm_testkit.clients.anythingllm_client import AnythingLLMClient
from llm_testkit.core.http_client import HttpClient
from llm_testkit.datasets.golden import load_golden_dataset


@pytest.fixture
def performance_budget(pytestconfig):
    return {
            "requests": pytestconfig.getoption("performance_requests"),
            "users": pytestconfig.getoption("performance_users")}


@pytest.fixture
def performance_report_directory(automation_root):
    return automation_root / "reports/performance"


@pytest.fixture
def health_workload(settings):
    def request_health():
        with HttpClient(settings.base_url, settings.http_timeout) as http:
            assertions.assert_online(AnythingLLMClient(http).health())

    return request_health


@pytest.fixture
def health_performance_context(settings, pytestconfig):
    return {
            "base_url": settings.base_url,
            "timeout": settings.http_timeout,
            "maximum_p95": pytestconfig.getoption("performance_p95"),
            "metadata": {
            "workload": "health"}}


@pytest.fixture
def performance_case(automation_root, policy_file):
    dataset = load_golden_dataset(automation_root / "test_data/golden-policy.json", policy_file)
    return next(c for c in dataset.cases if c.id == "carryover_limit")


@pytest.fixture
def rag_workload(settings, rag_environment, indexed_workspace, uploaded_policy_document, performance_case):
    def request_answer():
        # Concurrent requests own their transport; setup/indexing is outside timing.
        with HttpClient(settings.base_url, settings.http_timeout) as http:
            response = AnythingLLMClient(http, settings.api_key).chat(
                    indexed_workspace["slug"], performance_case.question, timeout=settings.llm_timeout)
            assertions.assert_golden_answer(
                    response, case=performance_case, document_title=uploaded_policy_document["title"])

    return request_answer


@pytest.fixture
def rag_performance_context(
        settings, pytestconfig, automation_root, generation_model, generation_model_digest, workspace_configuration,
        policy_file):
    dataset = load_golden_dataset(automation_root / "test_data/golden-policy.json", policy_file)
    return {
            "base_url": settings.base_url,
            "timeout": settings.llm_timeout,
            "maximum_p95": pytestconfig.getoption("performance_p95"),
            "metadata": {
            "workload": "rag",
            "generation_model": generation_model,
            "model_digest": generation_model_digest,
            "configuration": workspace_configuration,
            "policy_sha256": hashlib.sha256(policy_file.read_bytes()).hexdigest(),
            "golden_dataset_sha256": dataset.sha256,
            "case_id": "carryover_limit"}}
