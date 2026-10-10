"""Test data builders and doubles for the curated policy benchmark."""

import copy
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any
from unittest.mock import Mock

from llm_testkit.datasets.benchmark import make_plan, manifest
from llm_testkit.datasets.golden import GoldenCase
from llm_testkit.evaluation import benchmark as evaluation
from llm_testkit.evaluation import benchmark_runner as runner
from llm_testkit.evaluation.correctness import METRIC_CONFIGURATION
from llm_testkit.observation.evaluation_sample import write_sample
from llm_testkit.reporting.gates import METRICS, load_quality_gates
from test_support.builders.calibration import CONTROLS_FILE
from test_support.builders.golden import CASES, GOLDEN_DATASET, TEST_DATA, make_case_sample
from test_support.builders.ollama import model_catalog

QUALITY_GATES_FILE = TEST_DATA / "quality-gates.json"
QUALITY_GATES = load_quality_gates(QUALITY_GATES_FILE)
MINIMA = QUALITY_GATES["minimum_scores"]
MODEL = "qwen3.5:4b"
GENERATION_DIGEST = "generation-digest"
JUDGE_DIGEST = "judge-digest"
EVIDENCE_FILES = ("faithfulness", "correctness", "relevance")


def make_benchmark_sample(case: GoldenCase, model: str = MODEL, *, model_digest: str = GENERATION_DIGEST) -> dict[str,
        Any]:
    """A captured sample with the golden provenance and generation metadata the benchmark requires."""
    sample = make_case_sample(case, model=model)
    sample["metadata"] = {
            "golden_case_id": case.id,
            "golden_category": case.category,
            "golden_dataset_sha256": GOLDEN_DATASET.sha256,
            "policy_sha256": GOLDEN_DATASET.policy_sha256,
            "model_digest": model_digest,
            "workspace_configuration": {
            "openAiPrompt": "Policy only",
            "chatModel": model}}
    return sample


def make_metric_evidence(case: GoldenCase, sample_path: Path) -> dict[str, dict[str, Any]]:
    """Completed faithfulness, correctness and relevance evidence that a perfect answer would produce."""
    checksum = hashlib.sha256(sample_path.read_bytes()).hexdigest()
    verdict = {"statement": case.reference, "verdict": 1, "reason": "Supported"}
    judge = {"judge_model": MODEL, "judge_model_digest": JUDGE_DIGEST, "judge_configuration": {"think": False}}
    common = {"schema_version": 1, "status": "completed", "sample_sha256": checksum, **judge}
    faithfulness = {
            **common, "metric": "faithfulness",
            "ragas_version": "0.4.3",
            "created_at": "now",
            "result": {
            "value": 1.0,
            "statements": [case.reference],
            "verdicts": [verdict]},
            "judge_calls": [{
            "output": {
            "statements": [case.reference]}}, {
            "output": {
            "statements": [verdict]}}]}
    golden = {
            "golden_dataset_sha256": GOLDEN_DATASET.sha256,
            "golden_case_id": case.id,
            "question": case.question,
            "reference": case.reference}
    correctness_result = {
            "value": 1.0,
            "response_claims": [case.reference],
            "reference_claims": [case.reference],
            "response_verdicts": [verdict],
            "reference_verdicts": [verdict]}
    correctness = {
            **common,
            **golden, "metric":
            "factual_correctness",
            "response_origin":
            "application_sample",
            "metric_configuration":
            METRIC_CONFIGURATION,
            "response":
            case.reference,
            "reference_sha256":
            hashlib.sha256(case.reference.encode()).hexdigest(),
            "result":
            correctness_result,
            "judge_calls": [{
            "output": {
            "claims": [case.reference]}}, {
            "output": {
            "statements": [verdict]}}, {
            "output": {
            "claims": [case.reference]}}, {
            "output": {
            "statements": [verdict]}}]}
    precision = {"verdict": 1, "reason": "Relevant"}
    recall = {"statement": case.reference, "attributed": 1, "reason": "Supported"}
    relevance_result = {
            "context_precision": 1 / (1 + 1e-10),
            "context_recall": 1.0,
            "precision_verdicts": [precision],
            "recall_classifications": [recall]}
    relevance = {
            **common,
            **golden, "metric": "context_relevance",
            "result": relevance_result,
            "precision_calls": [{
            "output": precision}],
            "recall_calls": [{
            "output": {
            "classifications": [recall]}}]}
    return {"faithfulness": faithfulness, "correctness": correctness, "relevance": relevance}


def patch_metric_reports(monkeypatch, evidence: dict[str, dict[str, Any]]) -> dict[str, Mock]:
    """Replace the three judge report services with doubles returning the given evidence."""
    reports = {
            "faithfulness": Mock(return_value=evidence["faithfulness"]),
            "correctness": Mock(return_value=evidence["correctness"]),
            "relevance": Mock(return_value=evidence["relevance"])}
    monkeypatch.setattr(evaluation, "evaluate_sample_report", reports["faithfulness"])
    monkeypatch.setattr(evaluation, "evaluate_correctness_report", reports["correctness"])
    monkeypatch.setattr(evaluation, "evaluate_relevance_report", reports["relevance"])
    return reports


def make_row(case_id: str = "paid_leave", category: str = "multi_fact", *, model: str = MODEL) -> dict[str, Any]:
    """A benchmark result row whose two checks pass and whose four metrics pass, or are N/A for refusals."""
    refusal = category == "missing_information"
    metrics = [{
            "name": metric,
            "metric": metric,
            "status": "not_applicable" if refusal else "passed",
            "value": 1.0,
            "minimum": MINIMA[metric]} for metric in sorted(METRICS)]
    checks = [{"name": "Reviewed answer rules", "status": "passed"}, {"name": "Document sources", "status": "passed"}]
    return {
            "case_id": case_id,
            "category": category,
            "model": model,
            "model_digest": GENERATION_DIGEST,
            "workspace_configuration": {
            "openAiPrompt": "Policy only"},
            "generation_status": "passed",
            "dimensions": [*checks, *metrics]}


def make_definition(
        case_ids: tuple[str, ...] = ("paid_leave", "gym_missing"), models: tuple[str, ...] = (MODEL, )) -> dict:
    """The manifest of a small matrix: one answered case and one missing-information case by default."""
    plan = make_plan(GOLDEN_DATASET, case_ids=list(case_ids), models=list(models))
    return manifest(plan, GOLDEN_DATASET, QUALITY_GATES, CONTROLS_FILE)


def make_calibration(definition: dict) -> dict:
    """A matched judge-control summary bound to the manifest's judge, catalog and control identifiers."""
    return {
            "status": "matched",
            "judge_model": definition["judge_model"],
            "judge_model_digest": JUDGE_DIGEST,
            "controls_sha256": definition["controls_sha256"],
            "control_ids": definition["control_ids"],
            "results": [{
            "status": "matched"} for _ in definition["control_ids"]]}


# Run doubles shared with the CI and judge-validation tests.


def make_calibrate_stub(calibration):
    def calibrate(path, controls, settings, model, digest):
        calibrated = copy.deepcopy(calibration)
        calibrated["sample_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        catalog = json.loads(controls.read_text())["cases"]
        calibrated["results"] = []
        for identifier in calibrated["control_ids"]:
            control = next(c for c in catalog if c["id"] == identifier)
            statements = control["response"].split(". ")
            verdicts = [{
                    "statement": s,
                    "verdict": c["verdict"],
                    "reason": "Control label"} for s, c in zip(statements, control["claims"], strict=True)]
            calibrated["results"].append({
                    "status":
                    "matched",
                    "control":
                    control,
                    "result": {
                    "value": control["expected_score"],
                    "statements": statements,
                    "verdicts": verdicts},
                    "judge_calls": [{
                    "output": {
                    "statements": statements}}, {
                    "output": {
                    "statements": verdicts}}]})
        return calibrated

    return calibrate


def make_generate_stub(dataset, monkeypatch, *, metadata=None):
    """Generation that writes a passing sample, its JUnit and judge evidence; ``metadata(root, case, model)``
    adds sample metadata."""
    def generate(root, directory, identifier, model):
        case = CASES[identifier]
        sample = make_benchmark_sample(case, model, model_digest=JUDGE_DIGEST)
        sample["metadata"].update(metadata(root, identifier, model) if metadata else {})
        write_sample(directory / "sample.json", sample)
        patch_metric_reports(monkeypatch, make_metric_evidence(case, directory / "sample.json"))
        junit = directory / "generation.xml"
        junit.write_text(
                f'<testsuite><testcase classname="tests.test_golden_rag" name="test_golden_policy_answer[{identifier}-{model}]"/></testsuite>'
        )
        return {
                "generation_status": "passed",
                "generation_junit_sha256": hashlib.sha256(junit.read_bytes()).hexdigest()}

    return generate


def make_judge_model_catalog():
    return {"models": [{"name": MODEL, "digest": JUDGE_DIGEST}]}


def disputed_correctness(evidence: dict[str, Any]) -> dict[str, Any]:
    """Correctness evidence in which the judge rejects every claim of a correct answer (F1 = 0)."""
    disputed = copy.deepcopy(evidence)
    disputed["result"]["value"] = 0.0
    for direction in ("response", "reference"):
        for row in disputed["result"][f"{direction}_verdicts"]:
            row["verdict"] = 0
    for call in disputed["judge_calls"]:
        for row in call["output"].get("statements", []):
            row["verdict"] = 0
    return disputed


def save_benchmark_run(tmp_path: Path, monkeypatch, *, catalog=None, dispute_correctness: bool = False) -> Path:
    """Run the paid-leave and gym-refusal benchmark with deterministic doubles; return its output directory.

    Case-001 is paid leave and case-002 the refusal. A disputed run keeps every answer check passing while
    the correctness judge rejects the paid-leave answer.
    """
    root = tmp_path / "automation"
    shutil.copytree(TEST_DATA, root / "test_data")
    monkeypatch.delenv("ANYTHINGLLM_API_KEY", raising=False)
    monkeypatch.setattr(runner.OllamaClient, "list_models", catalog or (lambda _: model_catalog((MODEL, JUDGE_DIGEST))))
    definition = make_definition()
    monkeypatch.setattr(runner, "calibrate", make_calibrate_stub(make_calibration(definition)))
    generate = make_generate_stub(GOLDEN_DATASET, monkeypatch)

    def generate_with_dispute(*arguments):
        result = generate(*arguments)
        if dispute_correctness:
            correctness = evaluation.evaluate_correctness_report
            correctness.return_value = disputed_correctness(correctness.return_value)
        return result

    monkeypatch.setattr(runner, "generate_sample", generate_with_dispute)
    output = tmp_path / "run"
    plan = make_plan(GOLDEN_DATASET, case_ids=["paid_leave", "gym_missing"])
    runner.run(root, output, plan, GOLDEN_DATASET, QUALITY_GATES, notify=lambda *args, **kwargs: None)
    return output
