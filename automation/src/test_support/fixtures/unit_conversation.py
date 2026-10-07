"""Fixtures scoped to the associated unit-test module."""

import pytest

from llm_testkit.datasets.conversation import ConversationCatalog, load_conversation_catalog
from test_support.builders.conversation import DATA


@pytest.fixture
def catalog() -> ConversationCatalog:
    return load_conversation_catalog(DATA / "conversation-policy.json", DATA / "company-policy.txt")
