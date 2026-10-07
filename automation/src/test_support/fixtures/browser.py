"""Scoped scenario fixtures."""

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from playwright.sync_api import Page

    from llm_testkit.pages.workspace_page import WorkspacePage


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
