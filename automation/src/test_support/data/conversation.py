"""Reviewed catalogs, references and parameter cases."""

from test_support.paths import AUTOMATION_ROOT

DATA = AUTOMATION_ROOT / "test_data"

REVIEWED_REFERENCES_SATISFY_THEIR_INDEPENDENT_ACCEPTANCE_RULES_CASE_ID_CASES = [
        "greeting", "small_talk", "own_policy", "other_company", "mixed_request"]

CONVERSATION_CHECKS_REJECT_POLICY_DUMPS_HALLUCINATIONS_AND_MISSING_INTENTS_CASE_ID_CHANGE_MESSAGE_CASES = [
        ("greeting", "Hi! Employees get 23 working days of paid leave.", "forbidden content"),
        ("greeting", "Hello! Source: company-policy.txt.", "forbidden content"),
        ("greeting", "Hi! " + "word " * 41, "exceeds 40 words"),
        ("small_talk", "A walk sounds nice; it is sunny.", "invented current conditions"),
        ("small_talk", "The uploaded documents have no information about that.", "missing expected fact"),
        ("small_talk", "<think>A walk sounds nice.</think>Hello!", "missing expected fact"),
        ("own_policy", "append wrong units", "wrong day units"),
        ("other_company", "append invented amount", "invented leave entitlement"),
        ("mixed_request", "remove walk", "missing expected fact"),
        ("mixed_request", "remove unknown", "missing expected fact"),
        ("mixed_request", "append transfer", "transferred entitlement")]

POLICY_PARTS_REQUIRE_THE_EXPECTED_DOCUMENT_AND_SUPPORTING_PASSAGE_CASE_ID_CASES = ["own_policy", "mixed_request"]

POLICY_PARTS_REQUIRE_THE_EXPECTED_DOCUMENT_AND_SUPPORTING_PASSAGE_DEFECT_CASES = [
        "wrong document", "unsupported passage"]

WALKING_INTENT_ACCEPTS_FRESH_AIR_WITHOUT_REQUIRING_LITERAL_WALK_CASE_ID_CASES = ["small_talk", "mixed_request"]

INVALID_CATALOGS_FAIL_BEFORE_ANY_GENERATION_DEFECT_MESSAGE_CASES = [("checksum", "checksum mismatch"),
        ("duplicate id", "duplicate conversation case id"), ("word limit type", "max_words"),
        ("absent source", "absent from policy"), ("missing reference rule", "reference is missing"),
        ("forbidden reference", "reference violates")]

# Input for test_chat_request_explicitly_uses_chat_mode_without_changing_query_client_default
CHAT_MODE_CONFIGURATION = {"chatMode": "chat"}

# Input for test_chat_request_explicitly_uses_chat_mode_without_changing_query_client_default
TEMPORARY_WORKSPACE = {"slug": "temporary"}

# Input for test_conversation_fixture_rejects_a_query_profile
QUERY_MODE_CONFIGURATION = {"chatMode": "query"}
