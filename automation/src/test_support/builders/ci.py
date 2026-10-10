"""Prepared CI evidence and service doubles for readable offline scenarios."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from llm_testkit.ci.benchmark import execution_context, validate_ci_benchmark
from llm_testkit.evaluation import benchmark_runner as runner
from llm_testkit.observation.evaluation_sample import write_sample
from llm_testkit.qualification.plan import framework_checksum
from test_support.builders.benchmark import _mock_metrics, _sample
from test_support.data.ci import OTHER_REVISION, REVISION


def ci_generator(dataset, monkeypatch):
    def generate(root, directory, identifier, model):
        case = next(c for c in dataset.cases if c.id == identifier)
        sample = _sample(case, dataset, model)
        sample["metadata"].update(
                model_digest="judge-digest", framework_source_sha256=framework_checksum(root),
                test_source_sha256=hashlib.sha256((root / "tests/test_golden_rag.py").read_bytes()).hexdigest(),
                test_node_id=f"tests/test_golden_rag.py::test_golden_policy_answer[{identifier}-{model}]")
        write_sample(directory / "sample.json", sample)
        _mock_metrics(monkeypatch, case, dataset, directory / "sample.json")
        junit = directory / "generation.xml"
        junit.write_text(f'<testsuite><testcase name="test_golden_policy_answer[{identifier}-{model}]"/></testsuite>')
        return {
                "generation_status": "passed",
                "generation_junit_sha256": hashlib.sha256(junit.read_bytes()).hexdigest()}

    return generate


@dataclass
class SavedCIRun:
    root: Path
    directory: Path

    def validate(self):
        return validate_ci_benchmark(
                self.root, self.directory / "benchmark.json", revision=REVISION, profile="smoke", models="primary")

    def change(self, target: str) -> None:
        context_path = self.directory / "ci-context.json"
        context = json.loads(context_path.read_text())
        if target in ("revision", "source", "profile"):
            context[{
                    "revision": "revision",
                    "source": "framework_source_sha256",
                    "profile": "profile"}[target]] = OTHER_REVISION
            context_path.write_text(json.dumps(context))
        elif target == "baseline":
            with (self.directory / "quality-gates.json").open("a") as file:
                file.write("\n")
        elif target == "plan":
            path = self.directory / "manifest.json"
            value = json.loads(path.read_text())
            value["expected_rows"].pop()
            path.write_text(json.dumps(value))
        elif target == "judge_weights":
            path = self.root.parent / "config/ci-models.json"
            value = json.loads(path.read_text())
            value["qwen3.5:4b"] = "changed-weights"
            path.write_text(json.dumps(value))
        elif target == "generation_weights":
            sample_path = self.directory / "case-002/sample.json"
            sample = json.loads(sample_path.read_text())
            sample["metadata"]["model_digest"] = "changed-weights"
            sample_path.write_text(json.dumps(sample))
        else:
            (self.root / "tests/test_golden_rag.py").write_text("# Different golden test\n")

    def fail_generation(self) -> None:
        path = self.directory / "benchmark.json"
        report = json.loads(path.read_text())
        row = report["results"][0]
        row["error"] = "Generation could not complete"
        (self.directory / row["artifact_directory"] / "result.json").write_text(json.dumps(row))
        path.write_text(json.dumps(report))


def context_for(root: Path) -> dict:
    return execution_context(root, REVISION, "smoke", "primary")


def create_run(root, output, dataset, gates, plan):
    runner.run(root, output, plan, dataset, gates, notify=lambda *args, **kwargs: None)
    write_sample(output / "ci-context.json", context_for(root))
    return SavedCIRun(root, output)
