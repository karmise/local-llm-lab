import pytest

from llm_testkit.reporting.benchmark import present_benchmark_report
from llm_testkit.reporting.steps import title

pytestmark = pytest.mark.quality


@title("Fresh CI policy benchmark matches the tested commit and satisfies experimental quality gates")
def test_ci_benchmark_quality(saved_ci_benchmark_report) -> None:
    present_benchmark_report(saved_ci_benchmark_report)
