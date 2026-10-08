"""Fixtures scoped to the associated unit-test module."""

import json
from unittest.mock import Mock

import pytest

from llm_testkit.config import Settings
from llm_testkit.datasets.benchmark import make_plan, manifest
from llm_testkit.datasets.golden import load_golden_dataset
from llm_testkit.reporting.gates import load_quality_gates
from test_support.builders.benchmark import TimedSample, _sample
from test_support.data.benchmark import ANSWER_DURATION_SECONDS, ROOT
from test_support.fixtures.rag import rag_chat


@pytest.fixture
def benchmark_data():
    dataset = load_golden_dataset(ROOT / "test_data/golden-policy.json", ROOT / "test_data/company-policy.txt")
    gates = load_quality_gates(ROOT / "test_data/quality-gates.json")
    plan = make_plan(dataset, case_ids=["paid_leave", "gym_missing"])
    definition = manifest(plan, dataset, gates, ROOT / "test_data/faithfulness-controls.json")
    calibration = {
            "status": "matched",
            "judge_model": plan.judge_model,
            "judge_model_digest": "judge-digest",
            "controls_sha256": definition["controls_sha256"],
            "control_ids": definition["control_ids"],
            "results": [{
            "status": "matched"} for _ in range(3)]}
    return dataset, gates, plan, definition, calibration


@pytest.fixture
def timed_sample(tmp_path, benchmark_data):
    dataset = benchmark_data[0]
    case = dataset.cases[0]
    sample = _sample(case, dataset)
    sample["metadata"]["answer_request_seconds"] = ANSWER_DURATION_SECONDS
    path = tmp_path / "sample.json"
    path.write_text(json.dumps(sample))
    return TimedSample(path, dataset, case)


@pytest.fixture
def measured_chat(monkeypatch, tmp_path):
    clock = Mock(side_effect=(100.0, 112.5))
    monkeypatch.setattr("test_support.fixtures.rag.perf_counter", clock)
    client = Mock()
    record_property = Mock()
    chat = rag_chat.__wrapped__(
            request=Mock(), rag_environment=None, automation_root=tmp_path, authenticated_anythingllm_api=client,
            indexed_workspace={"slug": "temporary"}, settings=Settings(), capture_id=None,
            generation_model="qwen3.5:4b", rag_iteration=1, generation_model_digest="digest",
            workspace_configuration={}, policy_file=tmp_path / "unused.txt", record_property=record_property)
    return chat, client, record_property, clock
