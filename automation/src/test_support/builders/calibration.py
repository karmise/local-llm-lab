"""Test data builders for hand-labelled faithfulness judge controls."""

from typing import Any

from test_support.builders.golden import TEST_DATA

CONTROLS_FILE = TEST_DATA / "faithfulness-controls.json"


def make_control(control_id: str = "mixed") -> dict[str, Any]:
    """A control whose answer has one supported and one unsupported claim, so the expected score is 0.5."""
    claims = [{"pattern": "23", "verdict": 1}, {"pattern": "5000", "verdict": 0}]
    response = "Leave is 23 days. Gym reimbursement is 5000."
    return {"id": control_id, "response": response, "expected_score": 0.5, "claims": claims}


def make_matching_result() -> dict[str, Any]:
    """A faithfulness result that matches make_control(): one claim per pattern, labelled as expected."""
    supported = {"statement": "Leave is 23 days.", "verdict": 1}
    unsupported = {"statement": "Gym reimbursement is 5000.", "verdict": 0}
    return {"value": 0.5, "verdicts": [supported, unsupported]}


def make_catalog(*cases: dict[str, Any], anchors: tuple[str, ...] = ("Policy", )) -> dict[str, Any]:
    """A control catalog that requires the given context fragments; one mixed control by default."""
    return {
            "schema_version": 1,
            "required_context_fragments": list(anchors),
            "cases": list(cases) if cases else [make_control()]}
