"""Check local faithfulness judgement against hand-labelled answer controls."""

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
from typing import Any, Callable

from llm_testkit import assertions
from llm_testkit.clients.ollama_client import OllamaClient
from llm_testkit.config import Settings
from llm_testkit.core.http_client import HttpClient
from llm_testkit.evaluation.faithfulness import load_sample, score_sample
from llm_testkit.observation.evaluation_sample import write_sample


def load_controls(path: Path, contexts: list[str]) -> tuple[list[dict[str, Any]], str]:
    raw = path.read_bytes()
    controls = json.loads(raw)
    if controls.get("schema_version") != 1:
        raise ValueError("Unsupported calibration schema")
    anchors = controls.get("required_context_fragments")
    if not isinstance(anchors, list) or not anchors or any(not isinstance(x, str) or not x for x in anchors):
        raise ValueError("Controls must specify nonempty context anchors")
    combined = "\n".join(contexts)
    if any(fragment not in combined for fragment in anchors):
        raise ValueError("Captured context does not contain the policy required by these controls")
    cases = controls.get("cases")
    if not isinstance(cases, list) or not 1 <= len(cases) <= 3:
        raise ValueError("Select between one and three controls (maximum six judge calls)")
    identifiers = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Each control must be an object")
        identifier = case.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError("Control identifiers must be nonempty and unique")
        identifiers.add(identifier)
        if not isinstance(case.get("response"), str) or not case["response"].strip():
            raise ValueError("Control response must be nonempty")
        score = case.get("expected_score")
        if type(score) not in (int, float) or not 0 <= score <= 1:
            raise ValueError("Expected control score must be between zero and one")
        claims = case.get("claims")
        if not isinstance(claims, list) or not claims:
            raise ValueError("Controls require hand-labelled claims")
        for claim in claims:
            if not isinstance(claim, dict):
                raise ValueError("Each hand-labelled claim must be an object")
            if type(claim.get("verdict")) is not int or claim["verdict"] not in (0, 1):
                raise ValueError("Hand-labelled verdicts must be zero or one")
            if not isinstance(claim.get("pattern"), str) or not claim["pattern"]:
                raise ValueError("Control claim patterns must be nonempty")
            re.compile(claim["pattern"])
        if abs(sum(claim["verdict"] for claim in claims) / len(claims) - score) > 1e-9:
            raise ValueError("Expected score does not match hand-labelled verdicts")
    return cases, hashlib.sha256(raw).hexdigest()


async def evaluate_controls(
    sample: dict[str, Any], cases: list[dict[str, Any]], judge_factory: Callable[[], Any],
) -> list[dict[str, Any]]:
    if not 1 <= len(cases) <= 3:
        raise ValueError("Run between one and three controls")
    results = []
    for case in cases:
        judge = judge_factory()
        row: dict[str, Any] = {"control": case, "status": "error"}
        try:
            # Synthetic responses are controls, never application-generated evidence.
            control_sample = {**sample, "response": case["response"]}
            row["result"] = await score_sample(control_sample, judge)
            try:
                assertions.assert_calibration_result(
                    row["result"], expected_score=case["expected_score"], claims=case["claims"],
                )
                row["status"] = "matched"
            except AssertionError as error:
                row["status"] = "mismatch"
                row["mismatch"] = str(error)
        except Exception as error:
            row["error"] = {"type": type(error).__name__, "message": str(error)}
        row["judge_calls"] = judge.calls
        results.append(row)
    return results


def select_controls(cases: list[dict[str, Any]], identifiers: list[str] | None) -> list[dict[str, Any]]:
    if identifiers is None:
        return cases
    selected = set(identifiers)
    if not selected:
        raise ValueError("Select at least one control")
    unknown = selected - {case["id"] for case in cases}
    if unknown:
        raise ValueError(f"Unknown controls: {', '.join(sorted(unknown))}")
    return [case for case in cases if case["id"] in selected]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample", type=Path)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--judge-model", default="qwen3.5:4b")
    parser.add_argument("--control", action="append", dest="control_ids", help="Run only this control; repeat to select several.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new file")
    try:
        from llm_testkit.evaluation.ollama_judge import OllamaJudge
    except ImportError:
        parser.error('Install evaluation dependencies: python -m pip install -e ".[evaluation]"')
    report: dict[str, Any] = {
        "schema_version": 1, "status": "error", "metric": "faithfulness",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ragas_version": version("ragas"), "judge_model": args.judge_model,
        "sample_path": str(args.sample.resolve()), "controls_path": str(args.controls.resolve()),
        "interpretation": "Small hand-labelled control check, not statistical judge calibration",
        "response_origin": "synthetic_hand_labelled_controls",
    }
    http = None
    try:
        sample, report["sample_sha256"] = load_sample(args.sample)
        cases, report["controls_sha256"] = load_controls(args.controls, sample["retrieved_contexts"])
        cases = select_controls(cases, args.control_ids)
        report["selected_controls"] = [case["id"] for case in cases]
        settings = Settings.from_env()
        http = HttpClient(settings.ollama_base_url, settings.http_timeout)
        client = OllamaClient(http)
        catalog = client.list_models()
        assertions.assert_status_code(catalog, 200, context="Calibration judge model catalog")
        report["judge_model_digest"] = assertions.assert_model_available(catalog.json()["models"], args.judge_model)
        configuration = OllamaJudge(client, args.judge_model, settings.llm_timeout)
        report["judge_configuration"] = {"options": configuration.options, "think": False, "retries": 0}
        report["results"] = asyncio.run(evaluate_controls(
            sample, cases, lambda: OllamaJudge(client, args.judge_model, settings.llm_timeout),
        ))
        statuses = [row["status"] for row in report["results"]]
        report["status"] = "error" if "error" in statuses else "matched" if all(s == "matched" for s in statuses) else "mismatch"
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        if http:
            http.close()
    write_sample(args.output, report)
    for row in report.get("results", []):
        value = row.get("result", {}).get("value", "unavailable")
        print(f"{row['control']['id']}: {row['status']}; score={value}")
    print(f"Calibration status: {report['status']}; details saved to {args.output}")
    return 0 if report["status"] == "matched" else 1


if __name__ == "__main__":
    raise SystemExit(main())
