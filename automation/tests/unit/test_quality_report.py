import hashlib
import json
from pathlib import Path

import pytest

from llm_testkit import assertions
from llm_testkit.reporting import quality
from llm_testkit.reporting.allure_report import present_quality_report
from llm_testkit.reporting.quality import build_quality_report
from llm_testkit.reporting.steps import title
from test_support.assertions import errors as errors
from test_support.assertions import values as value_checks
from test_support.builders.quality_report import (
    _files,
    make_check_stub,
    make_claim_extraction_call,
    make_load_dataset_stub,
    make_step_stub,
    prepare_invalid_judge_evidence_case,
)
from test_support.data import common as case_data
from test_support.data.quality_report import (
    INCONSISTENT_FAITHFULNESS_CALL,
    INVALID_JUDGE_EVIDENCE_CHANGE_CASES,
    ORIGINAL_RELEVANCE_OBSERVATION,
)
from test_support.paths import AUTOMATION_ROOT

pytestmark = pytest.mark.unit


@title("Quality report combines fact checks, source checks and faithfulness measurement")
def test_report_combines_checks_and_measurement_without_quality_threshold(tmp_path: Path) -> None:
    report = build_quality_report(*_files(tmp_path))
    value_checks.equal(report["status"], "checks_passed")
    value_checks.equal(
        [d["status"] for d in report["dimensions"]], ["passed", "passed", "measured"]
    )
    value_checks.identical(report["dimensions"][2]["details"]["threshold"], None)


@title("Perfect faithfulness does not hide missing required answer facts")
def test_faithfulness_one_does_not_hide_incomplete_answer(tmp_path: Path) -> None:
    report = build_quality_report(*_files(tmp_path, incomplete=True))
    value_checks.equal(report["status"], "failed")
    value_checks.equal(
        [d["status"] for d in report["dimensions"]], ["failed", "passed", "measured"]
    )
    errors.rejects(
        lambda: assertions.assert_quality_report(report),
        expected=AssertionError,
        match="notice period",
    )


@pytest.mark.parametrize("change", INVALID_JUDGE_EVIDENCE_CHANGE_CASES)
@title("Invalid judge evidence is reported as an independent quality error [{param_id}]")
def test_invalid_judge_evidence_remains_an_independent_error(tmp_path: Path, change: str) -> None:
    paths = _files(tmp_path)
    evidence = json.loads(paths[1].read_text())
    prepare_invalid_judge_evidence_case(change, evidence)
    paths[1].write_text(json.dumps(evidence))
    report = build_quality_report(*paths)
    value_checks.equal(report["status"], "error")
    value_checks.equal([d["status"] for d in report["dimensions"]], ["passed", "passed", "error"])


@title("Expected source identity comes from captured context rather than response citations")
def test_expected_source_is_derived_from_context_not_citations(tmp_path: Path) -> None:
    paths = _files(tmp_path)
    sample = json.loads(paths[0].read_text())
    sample["response_sources"][0]["title"] = f"automation-{'b' * 32}-company-policy.txt"
    paths[0].write_text(json.dumps(sample))
    evidence = json.loads(paths[1].read_text())
    evidence["sample_sha256"] = hashlib.sha256(paths[0].read_bytes()).hexdigest()
    paths[1].write_text(json.dumps(evidence))
    report = build_quality_report(*paths)
    value_checks.equal(report["dimensions"][1]["status"], "failed")
    value_checks.equal(report["dimensions"][2]["status"], "measured")


@title("Allure renders all quality dimensions even when one check fails")
def test_allure_renders_remaining_steps_after_a_failed_dimension(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mock_factory) -> None:  # fmt: skip
    report = build_quality_report(*_files(tmp_path, incomplete=True))
    fake_allure = mock_factory()
    visited = []

    step = make_step_stub(visited)

    fake_allure.step.side_effect = step
    monkeypatch.setitem(__import__("sys").modules, "allure", fake_allure)
    errors.rejects(lambda: present_quality_report(report), expected=AssertionError)
    value_checks.equal(visited, [d["name"] for d in report["dimensions"]])


@title("Missing context evidence creates two visible errors without hiding other quality checks")
def test_optional_relevance_has_independent_dimensions(tmp_path: Path) -> None:
    report = build_quality_report(*_files(tmp_path), relevance_path=tmp_path / "absent.json")
    value_checks.equal(
        [d["status"] for d in report["dimensions"]],
        ["passed", "passed", "measured", "error", "error"],
    )
    value_checks.equal(
        [d["name"] for d in report["dimensions"][-2:]], ["context_precision", "context_recall"]
    )


@title(
    "Gated quality reports reject omitted semantic evidence instead of silently passing measurements"
)
def test_saved_report_gates_require_all_evidence(tmp_path: Path) -> None:
    gates = AUTOMATION_ROOT / "test_data/quality-gates.json"
    report = build_quality_report(*_files(tmp_path), gates_path=gates)
    value_checks.equal(report["status"], "error")
    value_checks.equal(report["dimensions"][2]["status"], "passed")
    value_checks.length(report["dimensions"], 6)
    errors.rejects(
        lambda: assertions.assert_quality_report(report),
        expected=AssertionError,
        match="not supplied",
    )


def test_relevance_dimensions_share_one_validated_evidence(tmp_path, monkeypatch):

    paths = _files(tmp_path)
    evidence_path = tmp_path / "relevance.json"
    evidence_path.write_text('{"observation": 1}')
    loads = []
    checks = []

    load_dataset = make_load_dataset_stub(loads)

    check = make_check_stub(checks, evidence_path)

    monkeypatch.setattr(quality, "load_golden_dataset", load_dataset)
    monkeypatch.setattr(quality, "check_relevance_evidence", check)
    report = build_quality_report(*paths, relevance_path=evidence_path)
    value_checks.equal([d["details"]["value"] for d in report["dimensions"][-2:]], [0.5, 0.5])
    value_checks.equal(loads, [1])
    value_checks.equal(checks, [case_data.fresh(ORIGINAL_RELEVANCE_OBSERVATION)])


def test_saved_faithfulness_rejects_summary_changed_from_raw_judge_calls(tmp_path):
    paths = _files(tmp_path)
    evidence = json.loads(paths[1].read_text())
    evidence["judge_calls"] = [
        make_claim_extraction_call(evidence),
        case_data.fresh(INCONSISTENT_FAITHFULNESS_CALL),
    ]
    paths[1].write_text(json.dumps(evidence))
    value_checks.equal(build_quality_report(*paths)["status"], "error")
