from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from requests import Response

from llm_testkit.config import Settings
from llm_testkit.observation.evaluation_sample import write_sample
from llm_testkit.reporting.allure_report import present_quality_report
from llm_testkit.reporting.live_quality import evaluate_captured_answer, generate_captured_answer
from llm_testkit.reporting.quality import build_quality_report
from llm_testkit.reporting.steps import title

pytestmark = [pytest.mark.rag, pytest.mark.live_quality]


@title("Live paid-leave answer passes quality checks with local judge evidence [{param_id}]")
def test_live_paid_leave_quality(
    rag_chat: Callable[[str, str], Response],
    captured_sample_path: Path,
    settings: Settings,
    request: pytest.FixtureRequest,
    automation_root: Path,
    paid_leave_profile: dict[str, Any],
    paid_leave_profile_path: Path,
) -> None:
    generate_captured_answer(rag_chat, paid_leave_profile, captured_sample_path)
    directory = automation_root / "reports/live-quality" / captured_sample_path.stem
    evidence_path = directory / "faithfulness.json"
    evaluate_captured_answer(
        captured_sample_path,
        evidence_path,
        settings=settings,
        judge_model=request.config.getoption("judge_model"),
    )
    report = build_quality_report(captured_sample_path, evidence_path, paid_leave_profile_path)
    report["execution_mode"] = "live"
    write_sample(directory / "quality.json", report)
    present_quality_report(report)
