"""Fixtures scoped to the associated unit-test module."""

import pytest

from llm_testkit.datasets.benchmark import make_plan, manifest
from llm_testkit.datasets.golden import load_golden_dataset
from llm_testkit.reporting.gates import load_quality_gates
from test_support.builders.benchmark import ROOT


@pytest.fixture
def benchmark_data():
    dataset = load_golden_dataset(
        ROOT / "test_data/golden-policy.json", ROOT / "test_data/company-policy.txt"
    )
    gates = load_quality_gates(ROOT / "test_data/quality-gates.json")
    plan = make_plan(dataset, case_ids=["paid_leave", "gym_missing"])
    definition = manifest(plan, dataset, gates, ROOT / "test_data/faithfulness-controls.json")
    calibration = {
        "status": "matched",
        "judge_model": plan.judge_model,
        "judge_model_digest": "judge-digest",
        "controls_sha256": definition["controls_sha256"],
        "control_ids": definition["control_ids"],
        "results": [{"status": "matched"} for _ in range(3)],
    }
    return dataset, gates, plan, definition, calibration
