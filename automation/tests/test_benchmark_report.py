import pytest

from llm_testkit.reporting.benchmark import present_benchmark_report
from llm_testkit.reporting.benchmark_evidence import load_saved_benchmark
from llm_testkit.reporting.steps import title

pytestmark = pytest.mark.quality


@title("Policy dataset benchmark satisfies reviewed rules and experimental metric gates")
def test_saved_benchmark_report(pytestconfig: pytest.Config) -> None:
    path = pytestconfig.getoption("benchmark_report")
    if path is None:
        pytest.skip("Supply --benchmark-report; no model calls are performed")
    report = load_saved_benchmark(path)
    present_benchmark_report(report)
