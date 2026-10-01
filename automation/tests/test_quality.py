from pathlib import Path
from uuid import uuid4

import pytest

from llm_testkit.observation.evaluation_sample import write_sample
from llm_testkit.reporting.quality import build_quality_report
from llm_testkit.reporting.allure_report import present_quality_report

pytestmark = pytest.mark.quality


def test_saved_paid_leave_quality_report(request: pytest.FixtureRequest) -> None:
    sample = request.config.getoption("quality_sample")
    evidence = request.config.getoption("faithfulness_report")
    if sample is None and evidence is None:
        pytest.skip("Supply --quality-sample and --faithfulness-report for offline quality reporting")
    if sample is None or evidence is None:
        pytest.fail("Both --quality-sample and --faithfulness-report are required", pytrace=False)
    try:
        import allure
    except ImportError:
        pytest.fail('Install the reporting extra: python -m pip install -e ".[reporting]"', pytrace=False)
    root = Path(__file__).resolve().parents[1]
    report = build_quality_report(sample, evidence, root / "test_data/quality-paid-leave.json")
    path = root / "reports/quality" / f"{uuid4().hex}.json"
    write_sample(path, report)
    allure.attach.file(str(evidence), name="Original judge evidence", attachment_type=allure.attachment_type.JSON)
    present_quality_report(report)
