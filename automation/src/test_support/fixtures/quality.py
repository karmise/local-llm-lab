"""Validate optional saved-report inputs before readable report scenarios run."""

import importlib.util
from pathlib import Path
from uuid import uuid4

import pytest

from llm_testkit.observation.evaluation_sample import write_sample
from llm_testkit.reporting.benchmark_evidence import load_saved_benchmark
from llm_testkit.reporting.quality import build_quality_report


@pytest.fixture
def saved_quality_report(request, automation_root: Path, paid_leave_profile_path: Path):
    paths = {
            name: request.config.getoption(name)
            for name in (
            "quality_sample", "faithfulness_report", "correctness_report", "relevance_report", "quality_gates")}
    if all(path is None for path in paths.values()):
        pytest.skip("Supply --quality-sample and --faithfulness-report for offline quality reporting")
    if paths["quality_sample"] is None or paths["faithfulness_report"] is None:
        pytest.fail("Both --quality-sample and --faithfulness-report are required", pytrace=False)
    if importlib.util.find_spec("allure") is None:
        pytest.fail('Install the reporting extra: python -m pip install -e ".[reporting]"', pytrace=False)
    request.node.user_properties.append(("quality_gates_enabled", "true" if paths["quality_gates"] else "false"))
    report = build_quality_report(
            paths["quality_sample"], paths["faithfulness_report"], paid_leave_profile_path,
            correctness_path=paths["correctness_report"], relevance_path=paths["relevance_report"],
            gates_path=paths["quality_gates"])
    write_sample(automation_root / "reports/quality" / f"{uuid4().hex}.json", report)
    return report, paths["faithfulness_report"]


@pytest.fixture
def saved_benchmark_report(pytestconfig):
    path = pytestconfig.getoption("benchmark_report")
    if path is None:
        pytest.skip("Supply --benchmark-report; no model calls are performed")
    return load_saved_benchmark(path)
