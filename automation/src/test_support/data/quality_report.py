"""Named parameter cases for quality_report scenarios."""

INVALID_JUDGE_EVIDENCE_CHANGE_CASES = [
    "checksum",
    "score",
    "unfinished",
    "missing_claim",
]


# Input for test_relevance_dimensions_share_one_validated_evidence
ORIGINAL_RELEVANCE_OBSERVATION = {"observation": 1}


# Input for test_saved_faithfulness_rejects_summary_changed_from_raw_judge_calls
INCONSISTENT_FAITHFULNESS_CALL = {
    "output": {"statements": [{"statement": "different claim", "verdict": 0}]}
}
