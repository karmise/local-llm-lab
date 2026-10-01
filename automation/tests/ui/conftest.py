from collections.abc import Iterator
import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

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
        import allure

        for artifact in sorted(Path(output).glob("*")):
            if artifact.suffix == ".png":
                allure.attach.file(str(artifact), name=artifact.name, attachment_type=allure.attachment_type.PNG)
            elif artifact.suffix == ".zip":
                allure.attach.file(str(artifact), name=artifact.name, attachment_type="application/zip", extension="zip")
