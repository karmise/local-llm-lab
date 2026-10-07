"""Scenario data builders and deterministic test doubles."""

from llm_testkit.reporting.gates import METRICS
from test_support.paths import AUTOMATION_ROOT

GATES = AUTOMATION_ROOT / "test_data/quality-gates.json"


def measured_report():
    return {
        "schema_version": 1,
        "status": "checks_passed",
        "dimensions": [
            {"name": "Facts", "status": "passed"},
            {"name": "Sources", "status": "passed"},
            *[
                {
                    "name": metric,
                    "metric": metric,
                    "status": "measured",
                    "details": {"value": 1.0, "threshold": None},
                }
                for metric in sorted(METRICS)
            ],
        ],
    }
