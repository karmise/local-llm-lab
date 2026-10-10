"""Reviewed catalogs, references and parameter cases."""

from test_support.paths import AUTOMATION_ROOT

ROOT = AUTOMATION_ROOT

INVALID_SUMMARY_CHANGE_CASES = [
        "duplicate", "unknown", "category", "missing_metric", "nan", "minimum", "false_pass", "inapplicable"]

ANSWER_DURATION_SECONDS = 12.5

# Input for test_failures_remain_visible
NAME_METRIC_CONTEXT_PRECISION_INPUT = {"name": "context_precision", "metric": "context_precision", "status": "error"}
