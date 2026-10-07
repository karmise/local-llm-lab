"""Scenario data builders and deterministic test doubles."""

import hashlib
import json
from copy import deepcopy

from llm_testkit.reporting.drift import (
    digest,
    seal_snapshot,
)
from llm_testkit.reporting.gates import METRICS
from test_support.data import common as case_data


def snapshot(sample="a", **changes):
    payload = {
        "schema_version": 1,
        "kind": "quality_snapshot",
        "run_id": sample,
        "recorded_at": "2026-10-06T00:00:00+00:00",
        "case_id": "paid_leave",
        "sample_sha256": sample * 64,
        "model": "model",
        "model_digest": "b" * 64,
        "policy_sha256": "c" * 64,
        "dataset_sha256": "d" * 64,
        "configuration": {"chatModel": "model", "openAiPrompt": "Policy", "topN": 4},
        "judges": {
            name: {
                "judge_model": "judge",
                "judge_model_digest": "f" * 64,
                "judge_configuration": {"think": False},
                "ragas_version": "test",
            }
            for name in ("faithfulness", "correctness", "relevance")
        },
        "evidence_sha256": dict.fromkeys(("faithfulness", "correctness", "relevance"), "0" * 64),
        "thinking_mode": "default",
        "context_parser": "test",
        "metrics": dict.fromkeys(METRICS, 1.0),
        "acceptance": {"facts": "passed", "sources": "passed"},
        **changes,
    }
    payload["configuration"] = {
        "chatModel": "model",
        "openAiPrompt": "Policy",
        "topN": 4,
        **payload["configuration"],
    }
    payload["prompt_sha256"] = changes.get(
        "prompt_sha256",
        hashlib.sha256(payload["configuration"]["openAiPrompt"].encode()).hexdigest(),
    )
    payload["judge_sha256"] = changes.get("judge_sha256", digest(payload["judges"]))
    return seal_snapshot(payload)


def prepare_comparison_case(base, change, changes):
    if change == "drop":
        changes["metrics"] = {**base["metrics"], "context_recall": 0.8}
    elif change == "facts":
        changes["acceptance"] = {"facts": "failed", "sources": "passed"}
    elif change == "judge":
        changes["judges"] = deepcopy(base["judges"])
        changes["judges"]["faithfulness"]["judge_model"] = "other-judge"
    elif change == "prompt":
        changes["configuration"] = {"chatModel": "model", "topN": 4, "openAiPrompt": "Changed"}
    elif change == "config":
        changes["configuration"] = {"chatModel": "model", "topN": 2}
    elif change == "model":
        changes.update(
            model="new", model_digest="2" * 64, configuration={"chatModel": "new", "topN": 4}
        )


def prepare_invalid_snapshot_case(change, row):
    if change == "missing":
        row["metrics"].pop("faithfulness")
    elif change == "boolean":
        row["metrics"]["faithfulness"] = True
    elif change == case_data.MODEL_DIGEST:
        row["model_digest"] = ""
    else:
        row["acceptance"]["facts"] = "error"


def prepare_snapshot_assembly_case(change, sample):
    if change == "policy":
        sample["metadata"]["policy_sha256"] = "changed"
    elif change == "configuration":
        sample["metadata"]["workspace_configuration"]["chatModel"] = "other"


def prepare_snapshot_assembly_case_2(change, evidence, path, quality, sample):
    if change == "changed-sample":
        sample["metadata"]["thinking_mode"] = "changed"
        path.write_text(json.dumps(sample))
    elif change == "changed-evidence":
        evidence.write_text(evidence.read_text() + "\n")
    elif change == "changed-dataset":
        quality["golden_dataset_sha256"] = "0" * 64


def prepare_resealed_history_case(field, row):
    if field == "configuration":
        row[field]["openAiPrompt"] = "Changed without updating its checksum"
    elif field == "judges":
        row[field]["faithfulness"]["judge_model"] = "Other judge with old checksum"
    else:
        row[field] = {}


def mutate_snapshot_evidence(change, dataset, evidence, path):
    quality = {
        "status": "error" if change == "error" else "checks_passed",
        "generation_model": "model",
        "sample_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "golden_dataset_sha256": dataset.sha256,
        "evidence_sha256": dict.fromkeys(
            ("faithfulness", "correctness", "relevance"),
            hashlib.sha256(evidence.read_bytes()).hexdigest(),
        ),
        "dimensions": [
            {"status": "passed"},
            {"status": "passed"},
            *[{"metric": m, "details": {"value": 1.0}} for m in sorted(METRICS)],
        ],
    }

    return quality


def make_snapshot_metadata(dataset, identifier):
    """Build input for test_snapshot_assembly."""
    return {
        "policy_sha256": dataset.policy_sha256,
        "model_digest": "b" * 64,
        "thinking_mode": "default",
        "workspace_configuration": {
            "chatModel": "model",
            "openAiPrompt": f"Policy\n[LLM_TESTKIT_CAPTURE:{identifier}]",
        },
    }


def make_snapshot_sources(evidence, root, case):
    """Build input for test_snapshot_assembly."""
    return {
        "faithfulness": evidence,
        "correctness": evidence,
        "relevance": evidence,
        "profile": root / "quality-paid-leave.json",
        "dataset_path": root / "golden-policy.json",
        "policy": root / "company-policy.txt",
        "case_id": case.id,
    }


def make_snapshot_capture(case, identifier):
    """Build input for test_snapshot_assembly."""
    return {
        "schema_version": 1,
        "boundary": "ollama-sdk-chat",
        "request": {
            "model": "model",
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": f"[LLM_TESTKIT_CAPTURE:{identifier}]\n[CONTEXT 0]:\nPolicy\n[END CONTEXT 0]",
                },
                {"role": "user", "content": case.question},
            ],
        },
    }


def make_judge_identity():
    """Build input for test_snapshot_assembly."""
    return {
        "judge_model": "judge",
        "judge_model_digest": "d" * 64,
        "judge_configuration": {},
        "ragas_version": "test",
    }
