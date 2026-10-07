import json
from copy import deepcopy

import pytest

from llm_testkit.datasets.golden import load_golden_dataset
from llm_testkit.observation.evaluation_sample import build_sample
from llm_testkit.reporting.drift import (
    compare_snapshots,
    record_snapshot,
    seal_snapshot,
    validate_snapshot,
)
from llm_testkit.reporting.gates import METRICS
from llm_testkit.reporting.steps import title
from test_support.assertions import errors as errors
from test_support.assertions import values as value_checks
from test_support.assertions.drift import check_snapshot_assembly_outcome
from test_support.builders.drift import (
    mutate_snapshot_evidence,
    prepare_comparison_case,
    prepare_invalid_snapshot_case,
    prepare_resealed_history_case,
    prepare_snapshot_assembly_case,
    prepare_snapshot_assembly_case_2,
    snapshot,
)
from test_support.data.drift import (
    COMPARISON_CHANGE_STATUS_CASES,
    HISTORY_ID_CANNOT_ESCAPE_DESTINATION_RUN_ID_CASES,
    INVALID_SNAPSHOT_CHANGE_CASES,
    RESEALED_HISTORY_FIELD_CASES,
    SNAPSHOT_ASSEMBLY_CHANGE_CASES,
)
from test_support.paths import AUTOMATION_ROOT

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("change", "status"),
    COMPARISON_CHANGE_STATUS_CASES,
)
@title("Baseline history separates quality drops from changed evaluation conditions [{param_id}]")
def test_comparison(change, status):
    base = snapshot()
    changes = {}
    prepare_comparison_case(base, change, changes)
    report = compare_snapshots(base, snapshot("b", **changes))
    value_checks.equal(report["status"], status)
    value_checks.equal(report["generation_model_changed"], change == "model")


@title("An identical captured answer is not a new time-series observation")
def test_duplicate_and_integrity(tmp_path):
    base = snapshot()
    path = record_snapshot(tmp_path, base)
    value_checks.truthy(path.exists())
    errors.rejects(
        lambda: record_snapshot(tmp_path, snapshot(run_id="other")),
        expected=ValueError,
        match="already recorded",
    )
    value_checks.equal(compare_snapshots(base, base)["status"], "duplicate")
    edited = deepcopy(base)
    edited["metrics"]["faithfulness"] = 0.0
    errors.rejects(lambda: validate_snapshot(edited), expected=ValueError, match="integrity")


@pytest.mark.parametrize("change", INVALID_SNAPSHOT_CHANGE_CASES)
@title("History rejects incomplete metrics and provenance [{param_id}]")
def test_invalid_snapshot(change):
    row = snapshot()
    row.pop("snapshot_sha256")
    prepare_invalid_snapshot_case(change, row)
    errors.rejects(
        lambda: validate_snapshot(seal_snapshot(row)), expected=(ValueError, AssertionError)
    )


@title("Allowed metric-drop boundary passes without rounding small regressions away")
def test_drop_boundary():
    base = snapshot()
    value_checks.equal(
        compare_snapshots(base, snapshot("b", metrics=dict.fromkeys(METRICS, 0.95)))["status"],
        "passed",
    )
    value_checks.equal(
        compare_snapshots(base, snapshot("b", metrics=dict.fromkeys(METRICS, 0.949)))["status"],
        "regression",
    )


@pytest.mark.parametrize(
    "change",
    SNAPSHOT_ASSEMBLY_CHANGE_CASES,
)
@title(
    "Snapshot assembly binds measured evidence to captured configuration and reviewed policy [{param_id}]"
)
def test_snapshot_assembly(tmp_path, monkeypatch, change):

    root = AUTOMATION_ROOT / "test_data"
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
    prepare_snapshot_assembly_case(change, sample)
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
    quality = mutate_snapshot_evidence(change, dataset, evidence, path)
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
    prepare_snapshot_assembly_case_2(change, evidence, path, quality, sample)
    check_snapshot_assembly_outcome(arguments, case, change, path)


@pytest.mark.parametrize("run_id", HISTORY_ID_CANNOT_ESCAPE_DESTINATION_RUN_ID_CASES)
def test_history_id_cannot_escape_destination(tmp_path, run_id):
    errors.rejects(
        lambda: record_snapshot(tmp_path / "history", snapshot(run_id=run_id)), expected=ValueError
    )
    value_checks.falsy((tmp_path / "escaped.json").exists())


def test_sealing_snapshot_is_idempotent():
    row = snapshot()
    value_checks.equal(seal_snapshot(row), row)


@pytest.mark.parametrize("field", RESEALED_HISTORY_FIELD_CASES)
def test_resealed_history_revalidates_provenance_contents(field):
    row = snapshot()
    prepare_resealed_history_case(field, row)
    errors.rejects(lambda: validate_snapshot(seal_snapshot(row)), expected=ValueError)
