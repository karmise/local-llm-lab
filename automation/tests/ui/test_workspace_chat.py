from typing import TYPE_CHECKING, Any

import pytest

from llm_testkit import assertions

if TYPE_CHECKING:
    from llm_testkit.pages.workspace_page import WorkspacePage

pytestmark = [pytest.mark.ui, pytest.mark.usefixtures("rag_environment")]


def test_paid_leave_answer_and_source_are_visible(
    workspace_page: "WorkspacePage", uploaded_policy_document: dict[str, Any],
    paid_leave_profile: dict[str, Any],
) -> None:
    import allure

    allure.dynamic.feature("Workspace UI")
    allure.dynamic.story("Grounded answer and document citation")
    profile = paid_leave_profile
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



def test_missing_policy_does_not_display_invented_reimbursement(
    workspace_page: "WorkspacePage",
) -> None:
    import allure

    allure.dynamic.feature("Workspace UI")
    allure.dynamic.story("Missing policy information")
    question = (
        "What is the company's gym membership reimbursement policy, "
        "and how much does it reimburse per month?"
    )
    with allure.step("Ask a question outside the policy document's scope"):
        workspace_page.send_question(question)
        assertions.assert_ui_question_visible(workspace_page, question)
    with allure.step("Check that the final answer acknowledges missing information"):
        answer = assertions.assert_ui_completed_answer(workspace_page)
        allure.attach(answer, name="Displayed final answer", attachment_type=allure.attachment_type.TEXT)
        assertions.assert_missing_policy_answer(answer)


def test_source_details_display_supporting_policy_passages(
    workspace_page: "WorkspacePage", uploaded_policy_document: dict[str, Any],
    paid_leave_profile: dict[str, Any],
) -> None:
    import allure

    allure.dynamic.feature("Workspace UI")
    allure.dynamic.story("Document source details")
    with allure.step("Generate an answer with a policy citation"):
        workspace_page.send_question(paid_leave_profile["question"])
        assertions.assert_ui_completed_answer(workspace_page)
    with allure.step("Open the cited document and check its supporting passages"):
        title = uploaded_policy_document["title"]
        workspace_page.open_sources()
        assertions.assert_ui_document_source(workspace_page, title=title)
        workspace_page.open_source_document(title)
        assertions.assert_ui_source_content(
            workspace_page, title=title, fragments=paid_leave_profile["source_fragments"],
        )
        allure.attach(workspace_page.page.screenshot(full_page=True, animations="disabled"), name="Document source details", attachment_type=allure.attachment_type.PNG)


def test_chat_history_survives_page_reload(
    workspace_page: "WorkspacePage", paid_leave_profile: dict[str, Any],
) -> None:
    import allure

    allure.dynamic.feature("Workspace UI")
    allure.dynamic.story("Chat history persistence")
    question = paid_leave_profile["question"]
    with allure.step("Send a question and remember the completed answer"):
        workspace_page.send_question(question)
        assertions.assert_ui_question_visible(workspace_page, question)
        answer = assertions.assert_ui_completed_answer(workspace_page)
        url = workspace_page.page.url
        allure.attach(answer, name="Answer before reload", attachment_type=allure.attachment_type.TEXT)
    with allure.step("Reload the same conversation and check exact history without duplicates"):
        workspace_page.reload()
        assertions.assert_ui_history_preserved(workspace_page, question=question, answer=answer, url=url)
