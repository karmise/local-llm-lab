import importlib.util
from pathlib import Path
from uuid import uuid4

import pytest

from llm_testkit.observation.evaluation_sample import write_sample
from llm_testkit.reporting.allure_report import present_quality_report
from llm_testkit.reporting.quality import build_quality_report
from llm_testkit.reporting.steps import title

pytestmark = pytest.mark.quality


@title("Paid-leave quality checks use saved answer and judge evidence")
def test_saved_paid_leave_quality_report(
    request: pytest.FixtureRequest,
    automation_root: Path,
    paid_leave_profile_path: Path,
) -> None:
    sample = request.config.getoption("quality_sample")
    evidence = request.config.getoption("faithfulness_report")
    correctness = request.config.getoption("correctness_report")
    if sample is None and evidence is None and correctness is None:
        pytest.skip(
            "Supply --quality-sample and --faithfulness-report for offline quality reporting"
        )
    if sample is None or evidence is None:
        pytest.fail("Both --quality-sample and --faithfulness-report are required", pytrace=False)
    if importlib.util.find_spec("allure") is None:
        pytest.fail(
            'Install the reporting extra: python -m pip install -e ".[reporting]"', pytrace=False
        )
    report = build_quality_report(
        sample, evidence, paid_leave_profile_path, correctness_path=correctness
    )
    path = automation_root / "reports/quality" / f"{uuid4().hex}.json"
    write_sample(path, report)
    present_quality_report(report, evidence_path=evidence)
