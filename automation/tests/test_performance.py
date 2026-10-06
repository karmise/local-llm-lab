import hashlib
import platform
from pathlib import Path
from uuid import uuid4

import pytest

from llm_testkit import assertions
from llm_testkit.clients.anythingllm_client import AnythingLLMClient
from llm_testkit.core.http_client import HttpClient
from llm_testkit.datasets.golden import load_golden_dataset
from llm_testkit.observation.evaluation_sample import write_sample
from llm_testkit.performance.runner import run_batch
from llm_testkit.reporting.steps import attach_file, title

pytestmark = pytest.mark.performance


def save_batch(request, automation_root: Path, report: dict, settings, *, metadata: dict) -> None:
    report["metadata"] = {
        **metadata,
        "python": platform.python_version(),
        "system": platform.platform(),
        "base_url": settings.base_url,
        "timeout": settings.llm_timeout,
        "warmup_requests": 0,
    }
    report["thresholds"] = {
        "maximum_p95": request.config.getoption("performance_p95"),
        "maximum_failure_rate": 0.0,
    }
    path = automation_root / f"reports/performance/{uuid4().hex}.json"
    write_sample(path, report)
    attach_file(
        path,
        name="Performance attempts and thresholds",
        media_type="application/json",
        extension="json",
    )
    assertions.assert_performance_batch(report, maximum_p95=report["thresholds"]["maximum_p95"])


@title("Bounded concurrent health requests meet declared latency and error thresholds")
def test_health_performance(request, settings, automation_root):
    def operation():
        with HttpClient(settings.base_url, settings.http_timeout) as http:
            assertions.assert_online(AnythingLLMClient(http).health())

    report = run_batch(
        operation,
        requests=request.config.getoption("performance_requests"),
        users=request.config.getoption("performance_users"),
    )
    save_batch(request, automation_root, report, settings, metadata={"workload": "health"})


@pytest.mark.rag
@title("Bounded RAG requests preserve answer correctness under declared concurrency [{param_id}]")
def test_rag_performance(
    request,
    settings,
    automation_root,
    rag_environment,
    indexed_workspace,
    uploaded_policy_document,
    generation_model,
    generation_model_digest,
    workspace_configuration,
    policy_file,
):
    if request.config.getoption("capture_rag"):
        pytest.fail(
            "Performance batches do not support the single-answer capture mode", pytrace=False
        )
    dataset = load_golden_dataset(automation_root / "test_data/golden-policy.json", policy_file)
    case = next(c for c in dataset.cases if c.id == "carryover_limit")

    def operation():
        # Every concurrent request owns its HTTP session. Setup/indexing is outside the timing window.
        with HttpClient(settings.base_url, settings.http_timeout) as http:
            response = AnythingLLMClient(http, settings.api_key).chat(
                indexed_workspace["slug"], case.question, timeout=settings.llm_timeout
            )
            assertions.assert_golden_answer(
                response, case=case, document_title=uploaded_policy_document["title"]
            )

    report = run_batch(
        operation,
        requests=request.config.getoption("performance_requests"),
        users=request.config.getoption("performance_users"),
    )
    save_batch(
        request,
        automation_root,
        report,
        settings,
        metadata={
            "workload": "rag",
            "generation_model": generation_model,
            "model_digest": generation_model_digest,
            "configuration": workspace_configuration,
            "policy_sha256": hashlib.sha256(policy_file.read_bytes()).hexdigest(),
            "golden_dataset_sha256": dataset.sha256,
            "case_id": case.id,
        },
    )
