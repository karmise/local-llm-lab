"""Reviewed catalogs, references and parameter cases."""

CAPTURE_ID = "a" * 32


SAMPLE_REJECTS_MISMATCHED_OR_AMBIGUOUS_OBSERVATIONS_CHANGE_CASES = [
    "model",
    "question",
    "history",
    "marker",
    "truncated",
    "indices",
    "empty",
]


INVALID_EVIDENCE_NEVER_LEAVES_PARTIAL_OUTPUT_VALUE_CASES = [float("nan"), float("inf"), object()]


# Input for test_publication_failure_removes_temporary_evidence
SERIALIZABLE_CAPTURE_PAYLOAD = {"value": 1}
