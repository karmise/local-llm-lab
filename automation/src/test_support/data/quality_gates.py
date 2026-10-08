"""Reviewed catalogs, references and parameter cases."""

import pytest

from llm_testkit.reporting.gates import METRICS
from test_support.paths import AUTOMATION_ROOT

GATES = AUTOMATION_ROOT / "test_data/quality-gates.json"

# Each metric must enforce its threshold. Common invalid-evidence paths need one metric.
FAIL_CLOSED_CASES = [
        *(pytest.param(metric, "low", "failed", id=f"{metric}-below-threshold") for metric in sorted(METRICS)),
        pytest.param("faithfulness", "missing", "error", id="missing-measurement"),
        pytest.param("faithfulness", "invalid", "error", id="invalid-evidence"),
        pytest.param("faithfulness", "not-measured", "error", id="unmeasured-value")]

BOUNDARY_METRIC_CASES = sorted(METRICS)

CONFIGURATION_CHANGE_CASES = ["missing", "unknown", "nan", "boolean", "range", "clinical", "version"]
