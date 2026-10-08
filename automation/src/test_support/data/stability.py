"""Named parameter cases for stability scenarios."""

GOLDEN_SUMMARY_SAME_CASE_CASES = [True, False]

GOLDEN_SUMMARY_SAME_CASE_IDS = ["changed-dataset", "different-cases"]

CONFIGURATION_METADATA_CHANGE_CASES = ["capture", "conflict", "invalid-json"]

CONVERSATION_SUMMARY_CHANGE_CASES = ["different-cases", "changed-catalog", "missing-catalog"]

# Input for test_duplicate_call_and_teardown_entries_count_as_one_run
CLASSNAME_NAME_TEST_EXAMPLE_QWEN_RUN_1_INPUT = {"classname": "tests.test_rag", "name": "test_example[qwen-run-1]"}
