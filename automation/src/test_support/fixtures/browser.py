"""Scoped scenario fixtures."""

from typing import TYPE_CHECKING

import pytest

from test_support.data.browser import CHAT_HTML

if TYPE_CHECKING:
    from playwright.sync_api import Page

    from llm_testkit.pages.workspace_page import WorkspacePage


@pytest.fixture
def workspace(page: "Page") -> "WorkspacePage":
    from llm_testkit.pages.workspace_page import WorkspacePage

    page.set_content(CHAT_HTML)
    return WorkspacePage(page, base_url="http://unused.test", timeout=2)


@pytest.fixture
def chat_simulation(workspace: "WorkspacePage"):
    from test_support.builders.browser import BrowserChatSimulation

    return BrowserChatSimulation(workspace.page)
