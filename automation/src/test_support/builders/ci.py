"""Test data builders for CI evidence: a saved benchmark run bound to a tested checkout and model lock."""

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_testkit.ci.benchmark import execution_context, inputs, validate_ci_benchmark
from llm_testkit.datasets.benchmark import manifest
from llm_testkit.evaluation import benchmark_runner as runner
from llm_testkit.observation.evaluation_sample import write_sample
from llm_testkit.qualification.plan import framework_checksum
from test_support.builders.benchmark import (
        JUDGE_DIGEST, MODEL, make_calibrate_stub, make_calibration, make_generate_stub)
from test_support.builders.golden import TEST_DATA
from test_support.builders.ollama import model_catalog
from test_support.paths import AUTOMATION_ROOT

REVISION = "a" * 40
OTHER_REVISION = "b" * 40
COMPARISON_MODEL = "qwen2.5:7b"
GOLDEN_TEST = AUTOMATION_ROOT / "tests/test_golden_rag.py"


def ci_metadata(root: Path, case_id: str, model: str) -> dict[str, str]:
    """Sample metadata showing the sample came from the checkout's framework and golden test."""
    return {
            "framework_source_sha256": framework_checksum(root),
            "test_source_sha256": hashlib.sha256((root / "tests/test_golden_rag.py").read_bytes()).hexdigest(),
            "test_node_id": f"tests/test_golden_rag.py::test_golden_policy_answer[{case_id}-{model}]"}


@dataclass(frozen=True)
class SavedCIRun:
    """A saved smoke benchmark run (``directory``) produced from a checkout (``root``)."""

    root: Path
    directory: Path
    models: str

    @property
    def lock(self) -> Path:
        return self.root.parent / "config/ci-models.json"

    def validate(self, *, revision: str = REVISION, profile: str = "smoke", models: str | None = None) -> dict:
        return validate_ci_benchmark(
                self.root, self.directory / "benchmark.json", revision=revision, profile=profile, models=models
                or self.models)

    def read(self, name: str) -> Any:
        return json.loads((self.directory / name).read_text())

    def write(self, name: str, value: Any) -> None:
        (self.directory / name).write_text(json.dumps(value))


def save_ci_run(
        tmp_path: Path, monkeypatch, *, models: str = "primary", lock: dict[str, str] | None = None,
        metadata=ci_metadata) -> SavedCIRun:
    """Run the CI smoke profile with deterministic doubles and record its CI context.

    Every model reports the judge digest; ``lock`` is the reviewed model lock (by default it matches).
    ``metadata(root, case_id, model)`` supplies the CI metadata saved in each sample.
    """
    root = tmp_path / "automation"
    shutil.copytree(TEST_DATA, root / "test_data")
    (root / "tests").mkdir()
    shutil.copyfile(GOLDEN_TEST, root / "tests/test_golden_rag.py")
    (tmp_path / "config").mkdir()
    (tmp_path / "config/ci-models.json").write_text(
            json.dumps(lock or {
            MODEL: JUDGE_DIGEST,
            COMPARISON_MODEL: JUDGE_DIGEST}))
    monkeypatch.delenv("ANYTHINGLLM_API_KEY", raising=False)
    monkeypatch.setattr(
            runner.OllamaClient, "list_models", lambda _: model_catalog((MODEL, JUDGE_DIGEST),
            (COMPARISON_MODEL, JUDGE_DIGEST)))
    dataset, gates, plan = inputs(root, "smoke", models)
    definition = manifest(plan, dataset, gates, root / "test_data/faithfulness-controls.json")
    monkeypatch.setattr(runner, "calibrate", make_calibrate_stub(make_calibration(definition)))
    monkeypatch.setattr(runner, "generate_sample", make_generate_stub(dataset, monkeypatch, metadata=metadata))
    output = tmp_path / "run"
    runner.run(root, output, plan, dataset, gates, notify=lambda *args, **kwargs: None)
    write_sample(output / "ci-context.json", execution_context(root, REVISION, "smoke", models))
    return SavedCIRun(root, output, models)
