"""Reviewed catalogs, references and parameter cases."""

from test_support.paths import AUTOMATION_ROOT

ROOT = AUTOMATION_ROOT

SELECTOR = "tests/test_example.py::test_example"

INVALID_PLAN_CHANGE_CASES = [
        "schema", "education", "version", "requirements", "row", "id", "duplicate", "phase", "risk", "acceptance",
        "tests", "selector", "unknown", "escape", "axes", "axis-values", "axis-duplicate", "data", "data-checksum",
        "data-escape"]

TRACE_OUTCOMES_STATUS_CASES = ["passed", "failed", "skipped", "error", "missing"]

UNBOUND_RESULTS_METADATA_CASES = [{
        "qualification_plan_sha256": "stale"}, {
        "test_source_sha256": "stale"}, {
        "framework_source_sha256": "stale"}, {
        "requirement_ids": "not-json"}, {
        "requirement_ids": "null"}, {
        "requirement_ids": '"REQ-EXAMPLE"'}]

TEARDOWN_ENTRIES_CHANGE_CASES = ["error", "metadata"]

PHASE_SCOPE_PHASES_CASES = [[], ["unknown"], ["OQ", "OQ"], ["PQ"]]

INVALID_PACKAGE_INPUTS_CHANGE_CASES = ["empty", "duplicate", "outside-attachment"]

INVALID_MANIFEST_CHANGE_CASES = ["escape", "duplicate", "missing", "schema"]

CHANGED_INPUTS_CHANGE_CASES = ["plan", "junit", "framework", "test", "data"]

CONFLICTING_INLINE_PROVENANCE_CANNOT_PASS_STALE_FIRST_CASES = [True, False]

PACKAGE_VERIFICATION_BINDS_OUTCOMES_TO_INPUTS_CHANGE_CASES = [
        "status", "plan", "resealed-trace", "trace-schema", "unlisted"]

PLAN_REJECTS_AMBIGUOUS_PATHS_AND_PHASE_TYPES_CHANGE_CASES = ["selector-alias", "data-alias", "phase-type"]

# Input for test_matching_requirements
MATCHING_GOLDEN_CASE_METADATA = {"golden_case_id": "first"}

# Input for test_matching_requirements
OTHER_GOLDEN_CASE_METADATA = {"golden_case_id": "other"}

# Input for test_trace_outcomes
FINAL_OUTCOME_STATUSES = {"passed": "passed", "failed": "failed"}

# Input for test_trace_outcomes
INCOMPLETE_OUTCOME_STATUSES = {"error": "incomplete", "skipped": "incomplete"}
