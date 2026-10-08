"""Reference-based RAGAS factual F1 with preserved bidirectional claim evidence."""

import argparse
import asyncio
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from llm_testkit import assertions
from llm_testkit.clients.ollama_client import OllamaClient
from llm_testkit.config import Settings
from llm_testkit.core.http_client import HttpClient
from llm_testkit.datasets.golden import GoldenCase, GoldenDataset, load_golden_dataset
from llm_testkit.evaluation.faithfulness import load_sample
from llm_testkit.observation.evaluation_sample import write_sample

METRIC_CONFIGURATION = {"mode": "f1", "beta": 1.0, "atomicity": "high", "coverage": "high"}


def validate_control(control: dict[str, Any]) -> None:
    for field in ("id", "case_id", "response"):
        if not isinstance(control.get(field), str) or not control[field].strip():
            raise ValueError(f"Control requires nonempty {field}")
    bounds = control.get("expected_f1_range")
    if not isinstance(bounds, list) or len(bounds) != 2:
        raise ValueError("Control requires an F1 range")
    for value in bounds:
        assertions.assert_quality_score(value)
    if bounds[0] > bounds[1]:
        raise ValueError("Invalid control F1 range")
    if type(control.get("response_verdict")) is not int or control["response_verdict"] not in (0, 1):
        raise ValueError("Invalid control response verdict")
    rules = control.get("reference_rules")
    if not isinstance(rules, dict) or not rules:
        raise ValueError("Control requires hand-labelled reference rules")
    for rule in rules.values():
        if type(rule.get("verdict")) is not int or rule["verdict"] not in (0, 1):
            raise ValueError("Invalid control reference verdict")
        if not isinstance(rule.get("pattern"), str) or not rule["pattern"]:
            raise ValueError("Missing control reference pattern")
        compiled = re.compile(rule["pattern"], re.IGNORECASE)
        if compiled.search(""):
            raise ValueError("Control pattern must not match empty text")


def check_control(result: dict[str, Any], control: dict[str, Any]) -> None:
    validate_control(control)
    validate_result(result)
    lower, upper = control["expected_f1_range"]
    if not lower <= result["value"] <= upper:
        raise ValueError("Control F1 is outside its expected range")
    if any(v["verdict"] != control["response_verdict"] for v in result["response_verdicts"]):
        raise ValueError("Control response verdict disagrees with labels")
    covered = set()
    for label, rule in control["reference_rules"].items():
        hits = [
                i for i, v in enumerate(result["reference_verdicts"])
                if re.search(rule["pattern"], v["statement"], re.IGNORECASE)]
        if not hits or any(result["reference_verdicts"][i]["verdict"] != rule["verdict"] for i in hits):
            raise ValueError(f"Control reference verdict disagrees with label: {label}")
        covered.update(hits)
    if len(covered) != len(result["reference_verdicts"]):
        raise ValueError("Control includes unlabelled reference claims")


def validate_result(result: dict[str, Any]) -> dict[str, Any]:
    for direction in ("response", "reference"):
        claims = result[f"{direction}_claims"]
        verdicts = result[f"{direction}_verdicts"]
        if (not isinstance(claims, list) or not claims or any(not isinstance(c, str) or not c.strip() for c in claims)
                    or len(set(claims)) != len(claims) or Counter(claims) != Counter(v["statement"] for v in verdicts)):
            raise ValueError(f"Incomplete or duplicate {direction} claim evidence")
        for v in verdicts:
            if type(v["verdict"]) is not int or v["verdict"] not in (0, 1):
                raise ValueError("Correctness verdict must be zero or one")
            if not isinstance(v.get("reason"), str) or not v["reason"].strip():
                raise ValueError("Correctness verdict requires a reason")
    tp = sum(v["verdict"] for v in result["response_verdicts"])
    fp = len(result["response_claims"]) - tp
    fn = sum(1 - v["verdict"] for v in result["reference_verdicts"])
    # RAGAS 0.4.3 uses response-side TP plus reference-side FN for factual F1.
    expected = round(2 * tp / (2 * tp + fp + fn), 2)
    assertions.assert_quality_score(result["value"])
    if not math.isclose(result["value"], expected, abs_tol=1e-9):
        raise ValueError("Correctness F1 does not match bidirectional verdicts")
    counts = {"tp": tp, "fp": fp, "fn": fn}
    if "counts" in result and result["counts"] != counts:
        raise ValueError("Correctness counts do not match verdicts")
    return {**result, "counts": counts}


def result_from_calls(calls: list[dict[str, Any]], value: float) -> dict[str, Any]:
    if len(calls) != 4:
        raise ValueError("Expected two decompositions and two claim verifications")
    return validate_result({
            "value": value,
            "response_claims": calls[0]["output"]["claims"],
            "response_verdicts": calls[1]["output"]["statements"],
            "reference_claims": calls[2]["output"]["claims"],
            "reference_verdicts": calls[3]["output"]["statements"]})


async def score_correctness(sample: dict[str, Any], judge: Any) -> dict[str, Any]:
    from ragas.metrics.collections import FactualCorrectness

    result = await FactualCorrectness(llm=judge,
            **METRIC_CONFIGURATION).ascore(response=sample["response"], reference=sample["reference"])
    return result_from_calls(judge.calls, result.value)


def bind_case(sample: dict[str, Any], dataset: GoldenDataset, case_id: str) -> GoldenCase:
    matches = [case for case in dataset.cases if case.id == case_id]
    if len(matches) != 1:
        raise ValueError("Unknown golden case")
    case = matches[0]
    if sample["user_input"] != case.question or sample["reference"] != case.reference:
        raise ValueError("Sample question/reference does not match the golden case")
    metadata = sample.get("metadata", {})
    for field, expected in (("golden_case_id", case.id), ("golden_dataset_sha256", dataset.sha256), ("policy_sha256",
            dataset.policy_sha256)):
        if field in metadata and metadata[field] != expected:
            raise ValueError(f"Sample {field} does not match golden provenance")
    return case


def evaluate_correctness_report(
        sample_path: Path, *, dataset_path: Path, policy_file: Path, case_id: str, settings: Settings,
        judge_model: str = "qwen3.5:4b", control: dict[str, Any] | None = None) -> dict[str, Any]:
    from llm_testkit.evaluation.ollama_judge import OllamaJudge

    report: dict[str, Any] = {
            "schema_version": 1,
            "metric": "factual_correctness",
            "status": "error",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "ragas_version": version("ragas"),
            "metric_configuration": METRIC_CONFIGURATION.copy(),
            "threshold": None,
            "interpretation": "Exploratory reference-based factual F1; not a calibrated quality gate",
            "response_origin": "synthetic_control" if control is not None else "application_sample",
            "judge_model": judge_model}
    judge = None
    http = None
    try:
        sample, checksum = load_sample(sample_path)
        dataset = load_golden_dataset(dataset_path, policy_file)
        case = bind_case(sample, dataset, case_id)
        report.update(
                sample_sha256=checksum, golden_dataset_sha256=dataset.sha256, golden_dataset_version=dataset.version,
                golden_case_id=case.id, reference_sha256=hashlib.sha256(case.reference.encode()).hexdigest(),
                reference=case.reference, question=case.question,
                generation_model=sample["observation"]["request"]["model"])
        if control is not None:
            validate_control(control)
            if control["case_id"] != case.id:
                raise ValueError("Control belongs to a different golden case")
            if not isinstance(control["response"], str) or not control["response"].strip():
                raise ValueError("Control response must be nonempty")
            sample = {**sample, "response": control["response"]}
            report["control"] = control
        report["response"] = sample["response"]
        report["same_generation_and_judge_model"] = report["generation_model"] == judge_model
        http = HttpClient(settings.ollama_base_url, settings.http_timeout)
        client = OllamaClient(http)
        catalog = client.list_models()
        assertions.assert_status_code(catalog, 200, context="Correctness judge catalog")
        report["judge_model_digest"] = assertions.assert_model_available(catalog.json()["models"], judge_model)
        judge = OllamaJudge(client, judge_model, settings.llm_timeout, max_calls=4)
        report["judge_configuration"] = {"options": judge.options, "think": False, "retries": 0, "max_calls": 4}
        report["result"] = asyncio.run(score_correctness(sample, judge))
        report["status"] = "completed"
        if control is not None:
            try:
                check_control(report["result"], control)
                report["control_status"] = "matched"
            except (ValueError, AssertionError) as error:
                report.update(control_status="mismatch", control_error=str(error))
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        if judge:
            report["judge_calls"] = judge.calls
        if http:
            http.close()
    return report


def check_correctness_evidence(evidence: dict[str, Any], checksum: str, sample: dict[str, Any],
        dataset: GoldenDataset) -> dict[str, Any]:
    if evidence.get("schema_version") != 1 or evidence.get("metric") != "factual_correctness":
        raise ValueError("Unsupported correctness report")
    if (evidence.get("status") != "completed" or evidence.get("response_origin") != "application_sample"):
        raise ValueError("Correctness requires completed application evidence, not synthetic controls")
    if (evidence.get("sample_sha256") != checksum or evidence.get("golden_dataset_sha256") != dataset.sha256):
        raise ValueError("Correctness sample/dataset checksum mismatch")
    case = bind_case(sample, dataset, evidence["golden_case_id"])
    if (evidence.get("metric_configuration") != METRIC_CONFIGURATION or evidence.get("question") != case.question
                or evidence.get("reference") != case.reference or evidence.get("response") != sample["response"]
                or evidence.get("reference_sha256") != hashlib.sha256(case.reference.encode()).hexdigest()):
        raise ValueError("Correctness inputs/configuration mismatch")
    result = validate_result(evidence["result"])
    if result_from_calls(evidence["judge_calls"], result["value"]) != result:
        raise ValueError("Correctness summary differs from saved judge calls")
    return {
            **result, "threshold": None,
            "metric_configuration": METRIC_CONFIGURATION,
            "judge_model": evidence["judge_model"],
            "judge_model_digest": evidence["judge_model_digest"],
            "golden_case_id": case.id,
            "golden_dataset_sha256": dataset.sha256}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample", type=Path)
    parser.add_argument("--case", required=True, dest="case_id")
    parser.add_argument("--dataset", type=Path, default=Path("test_data/golden-policy.json"))
    parser.add_argument("--policy", type=Path, default=Path("test_data/company-policy.txt"))
    parser.add_argument("--judge-model", default="qwen3.5:4b")
    parser.add_argument("--control", help="Evaluate one synthetic control instead of the application response")
    parser.add_argument("--controls", type=Path, default=Path("test_data/correctness-controls.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new file")
    if find_spec("ragas") is None:
        parser.error("Install requirements-evaluation.lock before running correctness")
    control = None
    if args.control:
        data = json.loads(args.controls.read_text())
        matches = [c for c in data["cases"] if c["id"] == args.control]
        if data.get("schema_version") != 1 or len(matches) != 1:
            parser.error("Unknown or duplicate correctness control")
        control = matches[0]
    report = evaluate_correctness_report(
            args.sample, dataset_path=args.dataset, policy_file=args.policy, case_id=args.case_id,
            settings=Settings.from_env(), judge_model=args.judge_model, control=control)
    write_sample(args.output, report)
    print(f"Correctness: {report['status']}; details saved to {args.output}")
    return (0 if report["status"] == "completed" and report.get("control_status", "matched") == "matched" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
