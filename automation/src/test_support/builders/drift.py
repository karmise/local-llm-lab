"""Scenario data builders and deterministic test doubles."""

import hashlib

from llm_testkit.reporting.drift import (
    digest,
    seal_snapshot,
)
from llm_testkit.reporting.gates import METRICS


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
