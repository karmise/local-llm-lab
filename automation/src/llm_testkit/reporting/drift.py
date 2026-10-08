"""Immutable metric history and explicit baseline regression checks; no model calls."""

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from llm_testkit import assertions
from llm_testkit.core.provenance import normalize_configuration
from llm_testkit.datasets.golden import load_golden_dataset
from llm_testkit.evaluation.correctness import bind_case
from llm_testkit.evaluation.faithfulness import load_sample
from llm_testkit.observation.evaluation_sample import write_sample
from llm_testkit.reporting.gates import METRICS
from llm_testkit.reporting.quality import build_quality_report


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def seal_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    payload = {k: v for k, v in snapshot.items() if k != "snapshot_sha256"}
    return {**payload, "snapshot_sha256": digest(payload)}


def validate_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    payload = {k: v for k, v in snapshot.items() if k != "snapshot_sha256"}
    if snapshot.get("snapshot_sha256") != digest(payload):
        raise ValueError("Snapshot integrity checksum mismatch")
    if (type(snapshot.get("schema_version")) is not int or snapshot["schema_version"] != 1
                or snapshot.get("kind") != "quality_snapshot"):
        raise ValueError("Unsupported history snapshot")
    if set(snapshot["metrics"]) != METRICS:
        raise ValueError("History requires all four measured metrics")
    for value in snapshot["metrics"].values():
        assertions.assert_quality_score(value)
    for field in ("sample_sha256", "model_digest", "policy_sha256", "dataset_sha256", "prompt_sha256", "judge_sha256"):
        if not isinstance(snapshot.get(field), str) or not re.fullmatch(r"[a-f0-9]{64}", snapshot[field]):
            raise ValueError(f"Missing or invalid history provenance: {field}")
    for field in ("run_id", "recorded_at", "case_id", "model", "thinking_mode", "context_parser"):
        if not isinstance(snapshot.get(field), str) or not snapshot[field].strip():
            raise ValueError(f"Missing history identity: {field}")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", snapshot["run_id"]):
        raise ValueError("History run_id must be a single safe file identifier")
    configuration = normalize_configuration(snapshot.get("configuration"))
    prompt = configuration.get("openAiPrompt")
    if not isinstance(prompt, str) or configuration.get("chatModel") != snapshot["model"]:
        raise ValueError("History configuration must bind to the generation model and prompt")
    if hashlib.sha256(prompt.encode()).hexdigest() != snapshot["prompt_sha256"]:
        raise ValueError("History prompt checksum differs from configuration")
    judge_metrics = {"faithfulness", "correctness", "relevance"}
    judges = snapshot.get("judges")
    if not isinstance(judges, dict) or set(judges) != judge_metrics:
        raise ValueError("History requires all three judge configurations")
    if digest(judges) != snapshot["judge_sha256"]:
        raise ValueError("History judge checksum differs from configurations")
    for judge in judges.values():
        if (not isinstance(judge, dict) or any(not isinstance(judge.get(k), str) or not judge[k].strip()
                for k in ("judge_model", "judge_model_digest", "ragas_version"))
                    or not isinstance(judge.get("judge_configuration"), dict)):
            raise ValueError("Incomplete history judge provenance")
    evidence = snapshot.get("evidence_sha256")
    if (not isinstance(evidence, dict) or set(evidence) != judge_metrics
                or any(not isinstance(v, str) or not re.fullmatch(r"[a-f0-9]{64}", v) for v in evidence.values())):
        raise ValueError("History requires checksums for all measured evidence")
    for status in snapshot["acceptance"].values():
        if status not in ("passed", "failed"):
            raise ValueError("Incomplete deterministic acceptance evidence")
    if set(snapshot["acceptance"]) != {"facts", "sources"}:
        raise ValueError("History requires facts and source outcomes")
    return snapshot


def make_snapshot(
        sample_path: Path, *, faithfulness: Path, correctness: Path, relevance: Path, profile: Path, dataset_path: Path,
        policy: Path, case_id: str) -> dict[str, Any]:
    report = build_quality_report(
            sample_path, faithfulness, profile, correctness_path=correctness, relevance_path=relevance,
            golden_dataset_path=dataset_path, policy_file=policy)
    if report["status"] == "error":
        raise ValueError("Incomplete or invalid quality evidence cannot enter metric history")
    sample, checksum = load_sample(sample_path)
    dataset = load_golden_dataset(dataset_path, policy)
    if report["sample_sha256"] != checksum or report["golden_dataset_sha256"] != dataset.sha256:
        raise ValueError("Sample or dataset changed during history assembly")
    case = bind_case(sample, dataset, case_id)
    metadata = sample["metadata"]
    if metadata["policy_sha256"] != dataset.policy_sha256:
        raise ValueError("Captured policy does not match reviewed dataset")
    configuration = normalize_configuration(metadata["workspace_configuration"])
    if configuration.get("chatModel") != report["generation_model"]:
        raise ValueError("Recorded generation configuration mismatch")
    prompt = configuration["openAiPrompt"]
    metrics = {d["metric"]: d["details"]["value"] for d in report["dimensions"] if d.get("metric") in METRICS}
    judges = {}
    evidence_sha256 = {}
    for name, path in [("faithfulness", faithfulness), ("correctness", correctness), ("relevance", relevance)]:
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != report["evidence_sha256"].get(name):
            raise ValueError("Judge evidence changed during history assembly: " + name)
        evidence = json.loads(raw)
        judges[name] = {
                k: evidence[k]
                for k in ("judge_model", "judge_model_digest", "judge_configuration", "ragas_version")}
        evidence_sha256[name] = hashlib.sha256(raw).hexdigest()
    snapshot = {
            "schema_version": 1,
            "kind": "quality_snapshot",
            "run_id": uuid4().hex,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "case_id": case.id,
            "sample_sha256": checksum,
            "model": report["generation_model"],
            "model_digest": metadata["model_digest"],
            "policy_sha256": dataset.policy_sha256,
            "dataset_sha256": dataset.sha256,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "configuration": configuration,
            "thinking_mode": metadata["thinking_mode"],
            "context_parser": sample["context_parser"],
            "judge_sha256": digest(judges),
            "judges": judges,
            "evidence_sha256": evidence_sha256,
            "metrics": metrics,
            "acceptance": {
            "facts": report["dimensions"][0]["status"],
            "sources": report["dimensions"][1]["status"]}}
    return validate_snapshot(seal_snapshot(snapshot))


def record_snapshot(history: Path, snapshot: dict[str, Any]) -> Path:
    validate_snapshot(snapshot)
    for path in history.glob("*.json"):
        existing = validate_snapshot(json.loads(path.read_text()))
        if existing["sample_sha256"] == snapshot["sample_sha256"]:
            raise ValueError("This captured sample is already recorded; it is not a new observation")
    target = history / f"{snapshot['run_id']}.json"
    write_sample(target, snapshot)
    return target


def compare_snapshots(baseline: dict[str, Any], current: dict[str, Any], *, maximum_drop: float = 0.05) -> dict[str,
        Any]:
    validate_snapshot(baseline)
    validate_snapshot(current)
    assertions.assert_quality_score(maximum_drop)

    def config(row):
        return {k: v for k, v in normalize_configuration(row["configuration"]).items() if k != "chatModel"}

    fields = (
            "case_id", "policy_sha256", "dataset_sha256", "prompt_sha256", "thinking_mode", "context_parser",
            "judge_sha256")
    differences = [f for f in fields if baseline[f] != current[f]]
    if config(baseline) != config(current):
        differences.append("retrieval_configuration")
    deltas = {k: current["metrics"][k] - baseline["metrics"][k] for k in sorted(METRICS)}
    drops = [k for k, delta in deltas.items() if delta < -maximum_drop - 1e-12]
    failed = [k for k, status in current["acceptance"].items() if status == "failed"]
    duplicate = baseline["sample_sha256"] == current["sample_sha256"]
    status = (
            "incomparable"
            if differences else "duplicate" if duplicate else "regression" if drops or failed else "passed")
    return {
            "schema_version":
            1,
            "status":
            status,
            "baseline_run":
            baseline["run_id"],
            "current_run":
            current["run_id"],
            "baseline_sha256":
            baseline["snapshot_sha256"],
            "current_sha256":
            current["snapshot_sha256"],
            "maximum_drop":
            maximum_drop,
            "deltas":
            deltas,
            "dropped_metrics":
            drops,
            "failed_acceptance":
            failed,
            "incomparable_fields":
            differences,
            "generation_model_changed": (baseline["model"], baseline["model_digest"])
            != (current["model"], current["model_digest"]),
            "interpretation":
            "Scoped quality-regression signal against an explicit baseline; not statistical drift detection or an attribution of its cause"
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    record = sub.add_parser("record")
    record.add_argument("--sample", type=Path, required=True)
    for name in ("faithfulness", "correctness", "relevance"):
        record.add_argument("--" + name, type=Path, required=True)
    record.add_argument("--profile", type=Path, default=Path("test_data/quality-paid-leave.json"))
    record.add_argument("--dataset", type=Path, default=Path("test_data/golden-policy.json"))
    record.add_argument("--policy", type=Path, default=Path("test_data/company-policy.txt"))
    record.add_argument("--case", default="paid_leave")
    record.add_argument("--history", type=Path, default=Path("reports/history"))
    compare = sub.add_parser("compare")
    compare.add_argument("baseline", type=Path)
    compare.add_argument("current", type=Path)
    compare.add_argument("--maximum-drop", type=float, default=0.05)
    compare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "record":
        snapshot = make_snapshot(
                args.sample, faithfulness=args.faithfulness, correctness=args.correctness, relevance=args.relevance,
                profile=args.profile, dataset_path=args.dataset, policy=args.policy, case_id=args.case)
        print(record_snapshot(args.history, snapshot))
        return 0
    if args.output.exists():
        parser.error("Output already exists")
    report = compare_snapshots(
            json.loads(args.baseline.read_text()), json.loads(args.current.read_text()), maximum_drop=args.maximum_drop)
    write_sample(args.output, report)
    print(f"History comparison: {report['status']}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
