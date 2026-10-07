"""Arrange browser response states without JavaScript in executable scenarios."""

from typing import TYPE_CHECKING

from test_support.data.browser import (
    COMPLETED_REPLY_WITH_DISABLED_SEND,
    EMPTY_THEN_READY_REPLY,
)

if TYPE_CHECKING:
    from playwright.sync_api import Page


class BrowserChatSimulation:
    def __init__(self, page: "Page") -> None:
        self.page = page

    def show_previous_answer(self, text: str) -> None:
        self.page.evaluate("(text) => window.reply(text, true)", text)

    def show_answer_with_delayed_content(self) -> None:
        self.page.evaluate(EMPTY_THEN_READY_REPLY)

    def show_completed_answer_with_disabled_send(self) -> None:
        self.page.evaluate(COMPLETED_REPLY_WITH_DISABLED_SEND)
