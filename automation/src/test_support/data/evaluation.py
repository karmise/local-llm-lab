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


SUPPORTED_CLAIM_OUTPUTS = [
    {"statements": ["Supported claim.", "Unsupported claim."]},
    {
        "statements": [
            {"statement": "Supported claim.", "reason": "Found in context.", "verdict": 1},
            {"statement": "Unsupported claim.", "reason": "Absent from context.", "verdict": 0},
        ]
    },
]
CLAIM_EXTRACTION = {"statements": ["Claim."]}


# Input for test_native_judge_request_disables_thinking_and_passes_schema
OBJECT_RESPONSE_SCHEMA = {"type": "object"}


# Input for test_native_judge_request_disables_thinking_and_passes_schema
DETERMINISTIC_JUDGE_OPTIONS = {"temperature": 0}


# Input for test_live_faithfulness_rejects_score_inconsistent_with_verdicts
EXTRACTED_CLAIM_RESPONSE = {"output": {"statements": ["Claim."]}}


# Input for test_live_faithfulness_rejects_score_inconsistent_with_verdicts
UNSUPPORTED_CLAIM_VERDICT = {"output": {"statements": [{"statement": "Claim.", "verdict": 0}]}}
