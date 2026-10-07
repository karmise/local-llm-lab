"""Named parameter cases for drift scenarios."""

COMPARISON_CHANGE_STATUS_CASES = [
    ("none", "passed"),
    ("drop", "regression"),
    ("facts", "regression"),
    ("judge", "incomparable"),
    ("prompt", "incomparable"),
    ("config", "incomparable"),
    ("model", "passed"),
]

INVALID_SNAPSHOT_CHANGE_CASES = ["missing", "boolean", "digest", "acceptance"]

SNAPSHOT_ASSEMBLY_CHANGE_CASES = [
    "none",
    "error",
    "policy",
    "configuration",
    "changed-sample",
    "changed-evidence",
    "changed-dataset",
]

HISTORY_ID_CANNOT_ESCAPE_DESTINATION_RUN_ID_CASES = ["../escaped", "/absolute", "nested/file", ".."]

RESEALED_HISTORY_FIELD_CASES = [
    "configuration",
    "judges",
    "evidence_sha256",
]
