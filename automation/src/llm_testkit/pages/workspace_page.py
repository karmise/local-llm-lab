"""AnythingLLM 1.16.2 workspace actions and observed UI locators."""

import re
from urllib.parse import quote

from playwright.sync_api import Locator, Page

from llm_testkit.reporting.steps import step


class WorkspacePage:
    def __init__(self, page: Page, *, base_url: str, timeout: float) -> None:
        self.page = page
        self.base_url = base_url.rstrip("/")
        self.answer_timeout_ms = timeout * 1000
        self.composer = page.get_by_placeholder("Send a message", exact=True)
        self.send_button = page.get_by_role(
            "button", name="Send prompt message to workspace", exact=True
        )
        self.history = page.locator("#chat-history")
        # Upstream has no message test IDs. Scope by its response edit control.
        self.assistant_messages = self.history.locator(".group").filter(
            has=page.get_by_role("button", name="Edit Edit response", exact=True),
        )
        self._reply_index: int | None = None

    @property
    def assistant_message(self) -> Locator:
        if self._reply_index is None:
            return self.assistant_messages.last
        return self.assistant_messages.nth(self._reply_index)

    @property
    def final_answer(self) -> Locator:
        # The separate Thoughts panel is outside this final Markdown block.
        return self.assistant_message.locator(".break-words")

    @property
    def sources_button(self) -> Locator:
        return self.assistant_message.get_by_role("button", name="Sources", exact=True)

    @step("UI: open workspace")
    def open(self, slug: str) -> None:
        self._reply_index = None
        self.page.goto(f"{self.base_url}/workspace/{quote(slug, safe='')}")

    @step("UI: send question")
    def send_question(self, question: str) -> None:
        self.composer.fill(question)
        self._reply_index = self.assistant_messages.count()
        self.send_button.click()

    def user_question(self, question: str) -> Locator:
        # Markdown typography turns straight apostrophes into smart apostrophes.
        pattern = re.escape(question).replace("'", "['’]")
        return self.history.get_by_text(re.compile(rf"^{pattern}$"))

    def source_document(self, title: str) -> Locator:
        return self.page.get_by_role(
            "button", name=re.compile(rf"^{re.escape(title)} Document \d+ references?$")
        )

    @step("UI: open answer sources")
    def open_sources(self) -> None:
        self.sources_button.click()

    @step("UI: open cited document")
    def open_source_document(self, title: str) -> None:
        self.source_document(title).click()

    def source_heading(self, title: str) -> Locator:
        # The pinned UI truncates source-overlay titles after 45 characters.
        displayed_title = title if len(title) <= 45 else f"{title[:45]}…"
        return self.page.get_by_role("heading", name=displayed_title, exact=True)

    def source_details(self, title: str) -> Locator:
        # Upstream's source overlay lacks a dialog role; scope to its heading.
        return self.page.locator("div.fixed").filter(
            has=self.source_heading(title),
        )

    @step("UI: reload conversation")
    def reload(self) -> None:
        self.page.reload()
