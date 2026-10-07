"""Domain expectations for benchmark scenarios."""

import pytest

from llm_testkit.reporting.benchmark import summarize
from test_support.builders.benchmark import _row, prepare_different_configurations_rejected_case


def check_benchmark_configuration_changes(calibration, definition):
    for change in ("prompt", "digest"):
        row = _row("gym_missing", "missing_information")
        prepare_different_configurations_rejected_case(change, row)
        with pytest.raises(ValueError, match="changed"):
            summarize(definition, [_row(), row], calibration)
