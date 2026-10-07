"""Page Object regressions in a real browser with local, deterministic HTML."""

from typing import TYPE_CHECKING

import pytest

from llm_testkit import assertions
from test_support.fixtures.browser import workspace as workspace

if TYPE_CHECKING:
    from llm_testkit.pages.workspace_page import WorkspacePage


pytestmark = pytest.mark.browser


def test_answer_without_sources_is_still_a_completed_answer(workspace: "WorkspacePage") -> None:
    workspace.send_question("Unknown policy?")
    assert assertions.assert_ui_completed_answer(workspace) == "New final answer"


def test_second_question_waits_for_its_own_reply(workspace: "WorkspacePage") -> None:
    workspace.page.evaluate("window.reply('Previous final answer', true)")
    workspace.send_question("Second question?")
    assert assertions.assert_ui_completed_answer(workspace) == "New final answer"


def test_completed_answer_waits_for_nonempty_content(workspace: "WorkspacePage") -> None:
    workspace.page.evaluate("""() => {
        window.reply('');
        setTimeout(() => document.querySelector('.break-words').textContent = 'Ready', 200);
    }""")
    assert assertions.assert_ui_completed_answer(workspace) == "Ready"


def test_completed_answer_allows_disabled_send_with_empty_composer(
    workspace: "WorkspacePage",
) -> None:
    workspace.page.evaluate("""() => {
        window.reply('Complete');
        const send = document.querySelector('button[aria-label="Send prompt message to workspace"]');
        send.disabled = true;
    }""")
    assert assertions.assert_ui_completed_answer(workspace) == "Complete"
