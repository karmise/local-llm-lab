"""Scenario data builders and deterministic test doubles."""

from llm_testkit.reporting.gates import METRICS
from test_support.data.quality_gates import GATES as GATES


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


def prepare_fail_closed_case(change, report, row):
    if change == "low":
        row["details"]["value"] = 0.0
    elif change == "missing":
        report["dimensions"].remove(row)
    elif change == "invalid":
        row.update(status="error", error="Checksum mismatch")
    else:
        row["status"] = "passed"


def prepare_configuration_case(change, config):
    if change == "missing":
        config["minimum_scores"].pop("faithfulness")
    elif change == "unknown":
        config["minimum_scores"]["accuracy"] = 0.8
    elif change in ("nan", "boolean", "range"):
        config["minimum_scores"]["faithfulness"] = {
            "nan": float("nan"),
            "boolean": True,
            "range": 1.1,
        }[change]
    elif change == "clinical":
        config["calibration"] = "clinically validated"
    else:
        config["version"] = ""


def check_fail_closed_step_5(change, gated):
    assert gated["status"] == ("failed" if change == "low" else "error")
