"""Reviewed catalogs, references and parameter cases."""

from test_support.paths import AUTOMATION_ROOT

ROOT = AUTOMATION_ROOT

INVALID_PLAN_OPTIONS_CASES = [{
        "case_ids": []}, {
        "case_ids": ["unknown"]}, {
        "case_ids": ["gym_missing"]}, {
        "case_ids": ["paid_leave", "paid_leave"]}, {
        "models": []}, {
        "models": ["x", "x"]}, {
        "models": ["x", "y", "z"]}, {
        "models": ["../bad[model]"]}, {
        "max_model_calls": 42}, {
        "max_model_calls": True}, {
        "max_model_calls": 401}]

INVALID_SUMMARY_CHANGE_CASES = [
        "duplicate", "unknown", "category", "missing_metric", "nan", "minimum", "false_pass", "inapplicable"]

INVALID_ANSWER_DURATIONS = (True, 0, -1, float("nan"), float("inf"), "10")
ANSWER_DURATION_SECONDS = 12.5

# Input for test_failures_remain_visible
NAME_METRIC_CONTEXT_PRECISION_INPUT = {"name": "context_precision", "metric": "context_precision", "status": "error"}
