"""Domain expectations for calibration scenarios."""

import pytest

from llm_testkit.evaluation.calibration import load_controls


def check_invalid_control_catalog(change, path):
    with pytest.raises(ValueError):
        load_controls(path, ["Other document" if change == "context" else "Policy"])
