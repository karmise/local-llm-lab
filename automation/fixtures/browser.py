"""Scoped scenario fixtures."""

from typing import TYPE_CHECKING

import pytest

from builders.browser import CHAT_HTML

if TYPE_CHECKING:
    from playwright.sync_api import Page

    from pages.workspace_page import WorkspacePage


@pytest.fixture
def workspace(page: "Page") -> "WorkspacePage":
    from pages.workspace_page import WorkspacePage

    page.set_content(CHAT_HTML)
    return WorkspacePage(page, base_url="http://unused.test", timeout=2)


@pytest.fixture
def chat_simulation(workspace: "WorkspacePage"):
    from builders.browser import BrowserChatSimulation

    return BrowserChatSimulation(workspace.page)
