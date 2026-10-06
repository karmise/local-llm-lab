from copy import deepcopy

import pytest

from llm_testkit.reporting.drift import (
    compare_snapshots,
    record_snapshot,
    seal_snapshot,
    validate_snapshot,
)
from llm_testkit.reporting.gates import METRICS
from llm_testkit.reporting.steps import title

pytestmark = pytest.mark.unit


def snapshot(sample="a", **changes):
    return seal_snapshot(
        {
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
            "prompt_sha256": "e" * 64,
            "judge_sha256": "f" * 64,
            "configuration": {"chatModel": "model", "topN": 4},
            "thinking_mode": "default",
            "context_parser": "test",
            "metrics": dict.fromkeys(METRICS, 1.0),
            "acceptance": {"facts": "passed", "sources": "passed"},
            **changes,
        }
    )


@pytest.mark.parametrize(
    ("change", "status"),
    [
        ("none", "passed"),
        ("drop", "regression"),
        ("facts", "regression"),
        ("judge", "incomparable"),
        ("prompt", "incomparable"),
        ("config", "incomparable"),
        ("model", "passed"),
    ],
)
@title("Baseline history separates quality drops from changed evaluation conditions [{param_id}]")
def test_comparison(change, status):
    base = snapshot()
    changes = {}
    if change == "drop":
        changes["metrics"] = {**base["metrics"], "context_recall": 0.8}
    elif change == "facts":
        changes["acceptance"] = {"facts": "failed", "sources": "passed"}
    elif change in ("judge", "prompt"):
        changes[change + "_sha256"] = "1" * 64
    elif change == "config":
        changes["configuration"] = {"chatModel": "model", "topN": 2}
    elif change == "model":
        changes.update(
            model="new", model_digest="2" * 64, configuration={"chatModel": "new", "topN": 4}
        )
    report = compare_snapshots(base, snapshot("b", **changes))
    assert report["status"] == status
    assert report["generation_model_changed"] == (change == "model")


@title("An identical captured answer is not a new time-series observation")
def test_duplicate_and_integrity(tmp_path):
    base = snapshot()
    path = record_snapshot(tmp_path, base)
    assert path.exists()
    with pytest.raises(ValueError, match="already recorded"):
        record_snapshot(tmp_path, snapshot(run_id="other"))
    assert compare_snapshots(base, base)["status"] == "duplicate"
    edited = deepcopy(base)
    edited["metrics"]["faithfulness"] = 0.0
    with pytest.raises(ValueError, match="integrity"):
        validate_snapshot(edited)


@pytest.mark.parametrize("change", ["missing", "boolean", "digest", "acceptance"])
@title("History rejects incomplete metrics and provenance [{param_id}]")
def test_invalid_snapshot(change):
    row = snapshot()
    row.pop("snapshot_sha256")
    if change == "missing":
        row["metrics"].pop("faithfulness")
    elif change == "boolean":
        row["metrics"]["faithfulness"] = True
    elif change == "digest":
        row["model_digest"] = ""
    else:
        row["acceptance"]["facts"] = "error"
    with pytest.raises((ValueError, AssertionError)):
        validate_snapshot(seal_snapshot(row))


@title("Allowed metric-drop boundary passes without rounding small regressions away")
def test_drop_boundary():
    base = snapshot()
    assert (
        compare_snapshots(base, snapshot("b", metrics=dict.fromkeys(METRICS, 0.95)))["status"]
        == "passed"
    )
    assert (
        compare_snapshots(base, snapshot("b", metrics=dict.fromkeys(METRICS, 0.949)))["status"]
        == "regression"
    )


@pytest.mark.parametrize("change", ["none", "error", "policy", "configuration"])
@title(
    "Snapshot assembly binds measured evidence to captured configuration and reviewed policy [{param_id}]"
)
def test_snapshot_assembly(tmp_path, monkeypatch, change):
    import json
    from pathlib import Path

    from llm_testkit.datasets.golden import load_golden_dataset
    from llm_testkit.observation.evaluation_sample import build_sample
    from llm_testkit.reporting.drift import make_snapshot

    root = Path(__file__).resolve().parents[2] / "test_data"
    dataset = load_golden_dataset(root / "golden-policy.json", root / "company-policy.txt")
    case = next(c for c in dataset.cases if c.id == "paid_leave")
    identifier = "a" * 32
    sample = build_sample(
        {
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
        },
        question=case.question,
        answer=case.reference,
        reference=case.reference,
        expected_model="model",
        capture_id=identifier,
    )
    sample["metadata"] = {
        "policy_sha256": dataset.policy_sha256,
        "model_digest": "b" * 64,
        "thinking_mode": "default",
        "workspace_configuration": {
            "chatModel": "model",
            "openAiPrompt": f"Policy\n[LLM_TESTKIT_CAPTURE:{identifier}]",
        },
    }
    if change == "policy":
        sample["metadata"]["policy_sha256"] = "changed"
    elif change == "configuration":
        sample["metadata"]["workspace_configuration"]["chatModel"] = "other"
    path = tmp_path / "sample.json"
    path.write_text(json.dumps(sample))
    evidence = tmp_path / "judge.json"
    evidence.write_text(
        json.dumps(
            {
                "judge_model": "judge",
                "judge_model_digest": "d" * 64,
                "judge_configuration": {},
                "ragas_version": "test",
            }
        )
    )
    quality = {
        "status": "error" if change == "error" else "checks_passed",
        "generation_model": "model",
        "dimensions": [
            {"status": "passed"},
            {"status": "passed"},
            *[{"metric": m, "details": {"value": 1.0}} for m in sorted(METRICS)],
        ],
    }
    monkeypatch.setattr(
        "llm_testkit.reporting.drift.build_quality_report", lambda *args, **kwargs: quality
    )
    arguments = {
        "faithfulness": evidence,
        "correctness": evidence,
        "relevance": evidence,
        "profile": root / "quality-paid-leave.json",
        "dataset_path": root / "golden-policy.json",
        "policy": root / "company-policy.txt",
        "case_id": case.id,
    }
    if change != "none":
        with pytest.raises(ValueError):
            make_snapshot(path, **arguments)
    else:
        row = make_snapshot(path, **arguments)
        assert row["configuration"]["openAiPrompt"] == "Policy"
        assert row["case_id"] == case.id
        assert set(row["evidence_sha256"]) == {"faithfulness", "correctness", "relevance"}
