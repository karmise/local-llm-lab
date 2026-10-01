"""AnythingLLM 1.16.2 workspace actions and observed UI locators."""

import re
from urllib.parse import quote

from playwright.sync_api import Locator, Page


class WorkspacePage:
    def __init__(self, page: Page, *, base_url: str, timeout: float) -> None:
        self.page = page
        self.base_url = base_url.rstrip("/")
        self.answer_timeout_ms = timeout * 1000
        self.composer = page.get_by_placeholder("Send a message", exact=True)
        self.send_button = page.get_by_role("button", name="Send prompt message to workspace", exact=True)
        self.history = page.locator("#chat-history")
        # Upstream has no message test IDs. Scope by its response edit control.
        self.assistant_message = self.history.locator(".group").filter(
            has=page.get_by_role("button", name="Edit Edit response", exact=True),
        ).last
        # The separate Thoughts panel is outside this final Markdown block.
        self.final_answer = self.assistant_message.locator(".break-words")
        self.sources_button = self.assistant_message.get_by_role("button", name="Sources", exact=True)

    def open(self, slug: str) -> None:
        self.page.goto(f"{self.base_url}/workspace/{quote(slug, safe='')}")

    def send_question(self, question: str) -> None:
        self.composer.fill(question)
        self.send_button.click()

    def user_question(self, question: str) -> Locator:
        return self.history.get_by_text(question, exact=True)

    def source_document(self, title: str) -> Locator:
        return self.page.get_by_role("button", name=re.compile(rf"^{re.escape(title)} Document \d+ references?$"))

    def open_sources(self) -> None:
        self.sources_button.click()
