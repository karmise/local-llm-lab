"""Page Object regressions in a real browser with local, deterministic HTML."""

from typing import TYPE_CHECKING

import pytest

from llm_testkit import assertions

if TYPE_CHECKING:
    from playwright.sync_api import Page

    from llm_testkit.pages.workspace_page import WorkspacePage

pytestmark = pytest.mark.browser


@pytest.fixture
def workspace(page: "Page") -> "WorkspacePage":
    from llm_testkit.pages.workspace_page import WorkspacePage

    page.set_content("""
        <div id="chat-history"></div>
        <textarea placeholder="Send a message"></textarea>
        <button aria-label="Send prompt message to workspace">Send</button>
        <script>
          window.reply = (text, sources = false) => {
            const reply = document.createElement('div');
            reply.className = 'group';
            const content = document.createElement('div');
            content.className = 'break-words';
            content.textContent = text;
            reply.appendChild(content);
            const edit = document.createElement('button');
            edit.setAttribute('aria-label', 'Edit Edit response');
            reply.appendChild(edit);
            if (sources) {
              const button = document.createElement('button');
              button.textContent = 'Sources';
              reply.appendChild(button);
            }
            document.querySelector('#chat-history').appendChild(reply);
          };
          document.querySelector('button').onclick = () => {
            setTimeout(() => window.reply('New final answer'), 200);
          };
        </script>
    """)
    return WorkspacePage(page, base_url="http://unused.test", timeout=2)


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
