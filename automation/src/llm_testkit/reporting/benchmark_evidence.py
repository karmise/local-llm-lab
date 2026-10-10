"""Revalidate local benchmark artifacts without generating answers or judge calls."""

import hashlib
import json
import re
from pathlib import Path

from llm_testkit.datasets.golden import load_golden_dataset
from llm_testkit.evaluation.benchmark import EVIDENCE_METRICS, acceptance_dimensions, check_sample, metric_dimensions
from llm_testkit.evaluation.calibration import load_controls
from llm_testkit.evaluation.control_match import calibration_mismatch
from llm_testkit.evaluation.faithfulness import validate_result
from llm_testkit.reporting.benchmark import summarize
from llm_testkit.reporting.gates import load_quality_gates
from llm_testkit.reporting.junit import read_junit


def load_saved_benchmark(path: Path) -> dict:
    saved = json.loads(path.read_text())
    root = path.parent
    definition = json.loads((root / "manifest.json").read_text())
    if saved["manifest"] != definition:
        raise ValueError("Saved benchmark manifest differs from its artifact")
    dataset = load_golden_dataset(root / "golden-policy.json", root / "company-policy.txt")
    if (dataset.sha256 != definition["golden_dataset_sha256"] or dataset.policy_sha256 != definition["policy_sha256"]
                or dataset.version != definition["golden_dataset_version"]):
        raise ValueError("Benchmark dataset checksum mismatch")
    gates = load_quality_gates(root / "quality-gates.json")
    if gates != definition["quality_gates"]:
        raise ValueError("Benchmark gates differ from their recorded baseline")
    cases = {case.id: case for case in dataset.cases}
    for planned in definition["expected_rows"]:
        if (planned["case_id"] not in cases or cases[planned["case_id"]].category != planned["category"]):
            raise ValueError("Benchmark matrix differs from its golden cases")
    calibration = saved["calibration"]
    if calibration.get("status") == "matched":
        original = json.loads((root / "judge-controls.json").read_text())
        if original != calibration:
            raise ValueError("Judge control summary differs from its artifact")
        anchor = next((
                r for r in saved["results"]
                if r.get("sample_sha256") == calibration.get("sample_sha256") and r["case_id"] == "paid_leave"), None)
        if anchor is None:
            raise ValueError("Judge controls are not bound to a paid-leave benchmark sample")
        sample = check_sample(
                root / anchor["artifact_directory"] / "sample.json", dataset, cases["paid_leave"], anchor["model"])
        controls, checksum = load_controls(root / "faithfulness-controls.json", sample["retrieved_contexts"])
        if checksum != definition["controls_sha256"]:
            raise ValueError("Benchmark control catalog checksum mismatch")
        catalog = {c["id"]: c for c in controls}
        for result, identifier in zip(calibration["results"], definition["control_ids"], strict=True):
            control = catalog[identifier]
            if result["control"] != control:
                raise ValueError("Judge control labels differ from their catalog")
            evaluated = validate_result(result["result"])
            calls = result["judge_calls"]
            if (len(calls) != 2 or calls[0]["output"]["statements"] != evaluated["statements"]
                        or calls[1]["output"]["statements"] != evaluated["verdicts"]):
                raise ValueError("Judge control verdicts differ from their raw calls")
            mismatch = calibration_mismatch(
                    evaluated, expected_score=control["expected_score"], claims=control["claims"])
            if mismatch:
                raise ValueError(f"Judge control result differs from its label: {mismatch}")
    for row in saved["results"]:
        name = row["artifact_directory"]
        if not re.fullmatch(r"case-[0-9]{3}", name):
            raise ValueError("Invalid benchmark artifact directory")
        directory = root / name
        if json.loads((directory / "result.json").read_text()) != row:
            raise ValueError("Benchmark row differs from its original case artifact")
        if row.get("error"):
            continue
        junit = read_junit(directory / "generation.xml")
        if junit.sha256 != row["generation_junit_sha256"] or len(junit.cases) != 1:
            raise ValueError("Benchmark generation JUnit checksum/size mismatch")
        if next(iter(junit.cases.values())).status != row["generation_status"]:
            raise ValueError("Benchmark generation status differs from JUnit")
        case = cases[row["case_id"]]
        sample = check_sample(directory / "sample.json", dataset, case, row["model"])
        checksum = hashlib.sha256((directory / "sample.json").read_bytes()).hexdigest()
        if checksum != row["sample_sha256"] or sample["response"] != row["answer"]:
            raise ValueError("Benchmark sample/answer checksum mismatch")
        duration = sample["metadata"].get("answer_request_seconds")
        if row.get("answer_request_seconds") != duration:
            raise ValueError("Benchmark timing differs from captured sample metadata")
        if duration is not None and next(iter(
                junit.cases.values())).properties.get("answer_request_seconds") != str(duration):
            raise ValueError("Benchmark timing differs from generation JUnit")
        if row["dimensions"][:2] != acceptance_dimensions(sample, case):
            raise ValueError("Benchmark answer/source checks differ from actual inputs")
        if case.category == "missing_information":
            continue
        for name, metrics in EVIDENCE_METRICS.items():
            evidence_path = directory / f"{name}.json"
            if (hashlib.sha256(evidence_path.read_bytes()).hexdigest() != row["evidence_sha256"][name]):
                raise ValueError("Benchmark metric evidence checksum mismatch")
            actual = metric_dimensions(
                    name, json.loads(evidence_path.read_text()), checksum, sample, dataset, definition["judge_model"],
                    calibration.get("judge_model_digest"), gates["minimum_scores"])
            recorded = [d for d in row["dimensions"] if d.get("metric") in metrics]
            if recorded != actual:
                raise ValueError("Benchmark metric summary differs from original judge evidence")
    return summarize(definition, saved["results"], calibration)
