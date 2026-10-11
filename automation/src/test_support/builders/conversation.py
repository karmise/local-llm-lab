"""Test data builders for the reviewed conversational acceptance catalog."""

import json
from typing import Any

from llm_testkit.datasets.conversation import ConversationCase, load_conversation_catalog
from test_support.builders.golden import POLICY_FILE, TEST_DATA

CONVERSATION_FILE = TEST_DATA / "conversation-policy.json"


def catalog_json() -> dict[str, Any]:
    """A fresh, editable copy of the reviewed conversation catalog."""
    return json.loads(CONVERSATION_FILE.read_text())


def conversation_cases() -> dict[str, ConversationCase]:
    """The reviewed conversation cases by id, loaded on each call."""
    return {case.id: case for case in load_conversation_catalog(CONVERSATION_FILE, POLICY_FILE).cases}
