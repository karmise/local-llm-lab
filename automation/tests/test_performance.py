import pytest

from llm_testkit.performance.reporting import record_batch
from llm_testkit.performance.runner import run_batch
from llm_testkit.reporting.steps import title

pytestmark = pytest.mark.performance


@title("Bounded concurrent health requests meet declared latency and error thresholds")
def test_health_performance(
        health_workload, performance_budget, performance_report_directory, health_performance_context):
    report = run_batch(health_workload, **performance_budget)
    record_batch(report, performance_report_directory, **health_performance_context)


@pytest.mark.rag
@title("Bounded RAG requests preserve answer correctness under declared concurrency [{param_id}]")
def test_rag_performance(rag_workload, performance_budget, performance_report_directory, rag_performance_context):
    report = run_batch(rag_workload, **performance_budget)
    record_batch(report, performance_report_directory, **rag_performance_context)
