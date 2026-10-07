from typing import TYPE_CHECKING, Any

import pytest

from llm_testkit import assertions
from llm_testkit.reporting.steps import title

if TYPE_CHECKING:
    from llm_testkit.pages.workspace_page import WorkspacePage


pytestmark = pytest.mark.ui


@title("UI displays the paid-leave answer and its document source")
def test_paid_leave_answer_and_source_are_visible(
        workspace_page: "WorkspacePage", uploaded_policy_document: dict[str, Any],
        paid_leave_profile: dict[str, Any]) -> None:  # fmt: skip
    workspace_page.send_question(paid_leave_profile["question"])
    assertions.assert_ui_question_visible(workspace_page, paid_leave_profile["question"])
    assertions.assert_ui_policy_answer(
        workspace_page, fact_patterns=paid_leave_profile["fact_patterns"]
    )
    workspace_page.open_sources()
    assertions.assert_ui_document_source(workspace_page, title=uploaded_policy_document["title"])


@title("UI answer acknowledges missing gym policy without inventing reimbursement")
def test_missing_policy_does_not_display_invented_reimbursement(
        workspace_page: "WorkspacePage",
        missing_policy_profile: dict[str, Any]) -> None:  # fmt: skip
    question = missing_policy_profile["question"]
    workspace_page.send_question(question)
    assertions.assert_ui_question_visible(workspace_page, question)
    answer = assertions.assert_ui_completed_answer(workspace_page)
    assertions.assert_missing_policy_answer(answer)


@title("UI source details display the supporting policy passages")
def test_source_details_display_supporting_policy_passages(
        workspace_page: "WorkspacePage", uploaded_policy_document: dict[str, Any],
        paid_leave_profile: dict[str, Any]) -> None:  # fmt: skip
    workspace_page.send_question(paid_leave_profile["question"])
    assertions.assert_ui_completed_answer(workspace_page)
    title = uploaded_policy_document["title"]
    workspace_page.open_sources()
    assertions.assert_ui_document_source(workspace_page, title=title)
    workspace_page.open_source_document(title)
    assertions.assert_ui_source_content(
        workspace_page,
        title=title,
        fragments=paid_leave_profile["source_fragments"],
    )


@title("UI conversation keeps its question and answer after reload")
def test_chat_history_survives_page_reload(workspace_page: "WorkspacePage",
                                           paid_leave_profile: dict[str, Any]) -> None:  # fmt: skip
    question = paid_leave_profile["question"]
    workspace_page.send_question(question)
    assertions.assert_ui_question_visible(workspace_page, question)
    answer = assertions.assert_ui_completed_answer(workspace_page)
    url = workspace_page.page.url
    workspace_page.reload()
    assertions.assert_ui_history_preserved(
        workspace_page, question=question, answer=answer, url=url
    )
