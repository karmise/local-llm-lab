"""Page Object base layer: navigation and timeouts, checked against a mocked browser page."""

from unittest.mock import Mock

import pytest

from pages.base_page import BasePage
from pages.workspace_page import WorkspacePage
from reporting.steps import title

pytestmark = pytest.mark.unit


class ExamplePage(BasePage):
    """The smallest page: one action that opens the given path segments."""
    def open(self, *segments):
        self._goto(*segments)


@pytest.mark.parametrize(("application_url", "segments", "url"), [
        pytest.param(
        "http://localhost:3001", ("workspace", "policy"), "http://localhost:3001/workspace/policy", id="plain"),
        pytest.param(
        "http://localhost:3001/",
        ("workspace", "policy"), "http://localhost:3001/workspace/policy", id="trailing-slash"),
        pytest.param(
        "http://localhost:3001",
        ("workspace", "a/b c?d#e"), "http://localhost:3001/workspace/a%2Fb%20c%3Fd%23e", id="one-segment")])
@title("A page opens the application URL with each segment encoded on its own [{param_id}]")
def test_base_page_opens_encoded_path(application_url, segments, url):
    page = Mock()

    ExamplePage(page, base_url=application_url, timeout=2).open(*segments)

    page.goto.assert_called_once_with(url)


@title("Generated-content waits use the configured timeout in milliseconds")
def test_base_page_answer_timeout():
    assert BasePage(Mock(), base_url="http://localhost", timeout=2.5).answer_timeout_ms == 2500


@title("Reloading keeps the browser on the same page")
def test_base_page_reload():
    page = Mock()

    BasePage(page, base_url="http://localhost", timeout=2).reload()

    page.reload.assert_called_once_with()
    page.goto.assert_not_called()


@title("Opening a workspace navigates to its slug and targets the latest reply again")
def test_workspace_page_open():
    page = Mock()
    workspace = WorkspacePage(page, base_url="http://localhost:3001", timeout=2)
    workspace._reply_index = 3

    workspace.open("policy lab")

    page.goto.assert_called_once_with("http://localhost:3001/workspace/policy%20lab")
    assert workspace.assistant_message is workspace.assistant_messages.last
