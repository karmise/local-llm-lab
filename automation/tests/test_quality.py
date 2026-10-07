import pytest

from llm_testkit.reporting.allure_report import present_quality_report
from llm_testkit.reporting.steps import title

pytestmark = pytest.mark.quality


@title("Paid-leave quality checks use saved answer and judge evidence")
def test_saved_paid_leave_quality_report(saved_quality_report) -> None:
    report, evidence = saved_quality_report
    present_quality_report(report, evidence_path=evidence)
