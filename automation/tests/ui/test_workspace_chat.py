import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from llm_testkit import assertions

if TYPE_CHECKING:
    from llm_testkit.pages.workspace_page import WorkspacePage

pytestmark = [pytest.mark.ui, pytest.mark.usefixtures("rag_environment")]


def test_paid_leave_answer_and_source_are_visible(
    workspace_page: "WorkspacePage", uploaded_policy_document: dict[str, Any],
) -> None:
    import allure

    allure.dynamic.feature("Workspace UI")
    allure.dynamic.story("Grounded answer and document citation")
    profile = json.loads((Path(__file__).resolve().parents[2] / "test_data/quality-paid-leave.json").read_text())
    with allure.step("Send the policy question through the browser"):
        workspace_page.send_question(profile["question"])
        assertions.assert_ui_question_visible(workspace_page, profile["question"])
    with allure.step("Wait for a completed answer and check its required facts"):
        answer = assertions.assert_ui_policy_answer(workspace_page, fact_patterns=profile["fact_patterns"])
        allure.attach(answer, name="Displayed final answer", attachment_type=allure.attachment_type.TEXT)
    with allure.step("Open the answer's sources and check the uploaded document"):
        workspace_page.open_sources()
        assertions.assert_ui_document_source(workspace_page, title=uploaded_policy_document["title"])
        allure.attach(workspace_page.page.screenshot(full_page=True, animations="disabled"), name="Answer and source", attachment_type=allure.attachment_type.PNG)
