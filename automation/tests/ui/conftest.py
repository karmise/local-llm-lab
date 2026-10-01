from collections.abc import Iterator
import importlib.util
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from llm_testkit.reporting.steps import attach_browser_artifacts, set_metadata

if TYPE_CHECKING:
    from playwright.sync_api import Page
    from llm_testkit.config import Settings
    from llm_testkit.pages.workspace_page import WorkspacePage


@pytest.fixture(scope="session", autouse=True)
def ui_dependencies(request: pytest.FixtureRequest) -> None:
    if request.config.getoption("run_ui"):
        if any(importlib.util.find_spec(name) is None for name in ("playwright", "pytest_playwright", "allure")):
            pytest.fail('Install UI and reporting extras: python -m pip install -e ".[ui,reporting]"', pytrace=False)


@pytest.fixture
def workspace_page(
    page: "Page", settings: "Settings", indexed_workspace: dict[str, Any],
) -> "WorkspacePage":
    from llm_testkit.pages.workspace_page import WorkspacePage

    workspace = WorkspacePage(page, base_url=settings.base_url, timeout=settings.llm_timeout)
    workspace.open(indexed_workspace["slug"])
    return workspace


@pytest.hookimpl(hookwrapper=True, trylast=True)
def pytest_runtest_teardown(item: pytest.Item) -> Iterator[None]:
    yield
    # The Playwright plugin closes contexts and retains failure artifacts first.
    # Attach existing files while the Allure test is still open.
    output = item.funcargs.get("output_path")
    if output and item.config.getoption("allure_report_dir", default=None):
        attach_browser_artifacts(Path(output))


@pytest.fixture
def paid_leave_profile() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[2] / "test_data/quality-paid-leave.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def ui_report_metadata(request: pytest.FixtureRequest) -> None:
    story = request.node.originalname.removeprefix("test_").replace("_", " ").capitalize()
    set_metadata(feature="Workspace UI", story=story)
