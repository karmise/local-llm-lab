"""Base layer for Page Objects: the browser page, the application URL and navigation."""

from typing import TYPE_CHECKING
from urllib.parse import quote

from llm_testkit.reporting.steps import step

if TYPE_CHECKING:
    from playwright.sync_api import Page


class BasePage:
    """Shared navigation for application pages; subclasses own locators and user actions.

    Page Objects never check outcomes: acceptance criteria belong to `assertions.ui`.
    """
    def __init__(self, page: "Page", *, base_url: str, timeout: float) -> None:
        self.page = page
        self.base_url = base_url.rstrip("/")
        # Waits for generated content use the model timeout, not Playwright's default.
        self.answer_timeout_ms = timeout * 1000

    def _goto(self, *segments: str) -> None:
        """Open the application path made of `segments`, each encoded as one URL path segment."""
        self.page.goto("/".join([self.base_url, *(quote(segment, safe="") for segment in segments)]))

    @step("UI: reload page")
    def reload(self) -> None:
        self.page.reload()
