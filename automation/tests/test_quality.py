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
    relevance = request.config.getoption("relevance_report")
    gates = request.config.getoption("quality_gates")
    if (
        sample is None
        and evidence is None
        and correctness is None
        and relevance is None
        and gates is None
    ):
        pytest.skip(
            "Supply --quality-sample and --faithfulness-report for offline quality reporting"
        )
    if sample is None or evidence is None:
        pytest.fail("Both --quality-sample and --faithfulness-report are required", pytrace=False)
    if importlib.util.find_spec("allure") is None:
        pytest.fail(
            'Install the reporting extra: python -m pip install -e ".[reporting]"', pytrace=False
        )
    request.node.user_properties.append(
        ("quality_gates_enabled", "true" if gates is not None else "false")
    )
    report = build_quality_report(
        sample,
        evidence,
        paid_leave_profile_path,
        correctness_path=correctness,
        relevance_path=relevance,
        gates_path=gates,
    )
    path = automation_root / "reports/quality" / f"{uuid4().hex}.json"
    write_sample(path, report)
    present_quality_report(report, evidence_path=evidence)
