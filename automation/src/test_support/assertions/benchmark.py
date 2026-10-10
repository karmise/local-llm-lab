"""Domain expectations for benchmark scenarios."""

import pytest

from llm_testkit.reporting.benchmark import summarize
from test_support.assertions import values
from test_support.builders.benchmark import make_row, prepare_different_configurations_rejected_case
from test_support.data import common as case_data


def check_benchmark_configuration_changes(calibration, definition):
    for change in ("prompt", case_data.MODEL_DIGEST):
        row = make_row("gym_missing", "missing_information")
        prepare_different_configurations_rejected_case(change, row)
        with pytest.raises(ValueError, match="changed"):
            summarize(definition, [make_row(), row], calibration)


def check_answer_timing(summary):
    values.equal(summary["answer_timing"]["measured"], 2)
    values.equal(summary["answer_timing"]["unavailable"], 0)
    values.equal(summary["answer_timing"]["mean_seconds"], 15.0)
    values.equal(summary["answer_timing"]["minimum_seconds"], 10.0)
    values.equal(summary["answer_timing"]["maximum_seconds"], 20.0)
