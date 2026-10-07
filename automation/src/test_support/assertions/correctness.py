"""Domain expectations for correctness unit scenarios."""

from llm_testkit.evaluation.correctness import validate_control
from test_support.data.correctness import CASE as CASE


def check_control_expectations(cases):
    for control in cases:
        validate_control(control)
        assert control["case_id"] == CASE.id
