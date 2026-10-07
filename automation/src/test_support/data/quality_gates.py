"""Reviewed catalogs, references and parameter cases."""

from llm_testkit.reporting.gates import METRICS
from test_support.paths import AUTOMATION_ROOT

GATES = AUTOMATION_ROOT / "test_data/quality-gates.json"


FAIL_CLOSED_METRIC_CASES = sorted(METRICS)


FAIL_CLOSED_CHANGE_CASES = ["low", "missing", "invalid", "not-measured"]


BOUNDARY_METRIC_CASES = sorted(METRICS)


CONFIGURATION_CHANGE_CASES = [
    "missing",
    "unknown",
    "nan",
    "boolean",
    "range",
    "clinical",
    "version",
]
