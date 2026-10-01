from collections.abc import Callable
import importlib.util
import json
from pathlib import Path

import pytest
from requests import Response

from llm_testkit.config import Settings
from llm_testkit.evaluation.faithfulness import evaluate_sample_report
from llm_testkit.observation.evaluation_sample import write_sample
from llm_testkit.reporting.allure_report import present_quality_report
from llm_testkit.reporting.quality import build_quality_report

pytestmark = [pytest.mark.rag, pytest.mark.live_quality, pytest.mark.usefixtures("rag_environment")]


@pytest.fixture(scope="session", autouse=True)
def live_quality_dependencies(request: pytest.FixtureRequest) -> None:
    if request.config.getoption("run_live_quality"):
        if any(importlib.util.find_spec(name) is None for name in ("ragas", "allure")):
            pytest.fail('Install evaluation and reporting extras: python -m pip install -e ".[evaluation,reporting]"', pytrace=False)


def test_live_paid_leave_quality(
    rag_chat: Callable[[str, str], Response], captured_sample_path: Path,
    settings: Settings, request: pytest.FixtureRequest,
) -> None:
    import allure

    root = Path(__file__).resolve().parents[1]
    profile_path = root / "test_data/quality-paid-leave.json"
    profile = json.loads(profile_path.read_text())
    with allure.step("Generate one answer and capture actual model context"):
        rag_chat(
            profile["question"],
            "Each employee receives 23 working days of paid leave per year. "
            "A request must be submitted at least 12 calendar days before leave starts.",
        )
        allure.attach.file(str(captured_sample_path), name="Captured evaluation sample", attachment_type=allure.attachment_type.JSON)
    directory = root / "reports/live-quality" / captured_sample_path.stem
    evidence_path = directory / "faithfulness.json"
    with allure.step("Run local judge and preserve its evidence"):
        evidence = evaluate_sample_report(
            captured_sample_path, settings=settings,
            judge_model=request.config.getoption("judge_model"),
        )
        write_sample(evidence_path, evidence)
        allure.attach.file(str(evidence_path), name="Original judge evidence", attachment_type=allure.attachment_type.JSON)
    report = build_quality_report(captured_sample_path, evidence_path, profile_path)
    report["execution_mode"] = "live"
    write_sample(directory / "quality.json", report)
    present_quality_report(report)
