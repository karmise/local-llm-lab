import pytest

from llm_testkit.reporting.benchmark import present_benchmark_report
from llm_testkit.reporting.steps import title

pytestmark = pytest.mark.quality


@title("Policy dataset benchmark satisfies reviewed rules and experimental metric gates")
def test_saved_benchmark_report(saved_benchmark_report) -> None:
    present_benchmark_report(saved_benchmark_report)
