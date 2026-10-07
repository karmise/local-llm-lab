"""Named parameter cases for evaluation scenarios."""

QUALITY_SCORE_REJECTS_INVALID_RESULTS_VALUE_CASES = [
    float("nan"),
    float("inf"),
    -0.1,
    1.1,
    True,
    "1",
]

PIPELINE_REJECTS_MISSING_ALTERED_OR_INVALID_VERDICTS_OUTPUT_CASES = [
    {"statements": []},
    {"statements": [{"statement": "Other claim.", "reason": "Wrong.", "verdict": 1}]},
    {"statements": [{"statement": "Claim.", "reason": "Wrong.", "verdict": 2}]},
]
