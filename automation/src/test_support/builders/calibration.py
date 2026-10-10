"""Scenario data builders and deterministic test doubles."""

import json
from dataclasses import dataclass
from unittest.mock import Mock

from llm_testkit import assertions
from test_support.data import common as case_data
from test_support.data.benchmark import ROOT
from test_support.data.calibration import LEAVE_CONTROL_STATEMENT, NOTICE_CONTROL_STATEMENT


def _case() -> dict:
    return {
            "id": "mixed",
            "response": "Leave is 23 days. Gym reimbursement is 5000.",
            "expected_score": 0.5,
            "claims": [{
            "pattern": "23",
            "verdict": 1}, {
            "pattern": "5000",
            "verdict": 0}]}


def _result() -> dict:
    return {
            "value":
            0.5,
            "verdicts": [{
            "statement": "Leave is 23 days.",
            "verdict": 1}, {
            "statement": "Gym reimbursement is 5000.",
            "verdict": 0}]}


def prepare_control_loader_rejects_wrong_context_or_bad_labels_case(change, controls):
    if change == "score":
        controls["cases"][0]["expected_score"] = 1.0
    elif change == "duplicate":
        controls["cases"].append(_case())
    elif change == "empty":
        controls["cases"] = []


def make_fake_score_stub(observed_responses):
    async def fake_score(sample: dict, judge: object) -> dict:
        observed_responses.append(sample["response"])
        if len(observed_responses) == 3:
            raise ValueError("Truncated judge response")
        return _result() if len(observed_responses) == 1 else {**_result(), "value": 1.0}

    return fake_score


def make_factory_stub(judges):
    def factory() -> Mock:
        judge = Mock(calls=[])
        judges.append(judge)
        return judge

    return factory


def make_alternative_control():
    """Build input for test_targeted_selection_deduplicates_controls."""
    return {**_case(), "id": "other"}


def make_control_catalog():
    """Build input for test_control_loader_rejects_wrong_context_or_bad_labels."""
    return {"schema_version": 1, "required_context_fragments": ["Policy"], "cases": [_case()]}


def make_mismatched_control():
    """Build input for test_control_runner_preserves_mismatch_and_error_then_continues."""
    return {**_case(), "id": "mismatch"}


def make_error_control():
    """Build input for test_control_runner_preserves_mismatch_and_error_then_continues."""
    return {**_case(), "id": "error"}


def make_large_control_catalog():
    """Build input for test_large_catalog_can_be_loaded_without_executing_all_cases."""
    return {
            "schema_version": 1,
            "required_context_fragments": ["Policy"],
            "cases": [{
            **_case(), "id": f"control-{i}"} for i in range(6)]}


def make_numbered_control(i):
    """Build input for test_explicit_oversized_batch_is_rejected_before_model_calls."""
    return {**_case(), "id": f"control-{i}"}


def make_incomplete_policy_reply():
    """Build input for test_faithful_incomplete_answer_still_fails_required_fact_check."""
    return {
            "type": "textResponse",
            "error": None,
            "close": True,
            "textResponse": "Each employee receives 23 working days of paid leave per year.",
            "sources": [{
            "title": case_data.POLICY_DOCUMENT_TITLE,
            "text": "23 working days; 12 calendar days"}]}


@dataclass
class NumericControl:
    case: dict
    result: dict

    def check(self) -> None:
        assertions.assert_calibration_result(
                self.result, expected_score=self.case["expected_score"], claims=self.case["claims"])

    def replace_leave_claim(self, statement: str) -> None:
        self.result["verdicts"][0]["statement"] = statement


def make_numeric_control() -> NumericControl:
    catalog = json.loads((ROOT / "test_data/faithfulness-controls.json").read_text())
    case = next(c for c in catalog["cases"] if c["id"] == "wrong_numbers")
    result = {
            "value":
            0.0,
            "verdicts": [{
            "statement": LEAVE_CONTROL_STATEMENT,
            "verdict": 0}, {
            "statement": NOTICE_CONTROL_STATEMENT,
            "verdict": 0}]}
    return NumericControl(case, result)
