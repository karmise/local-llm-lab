"""Browser checks of the workspace page: question, final answer, sources and preserved history."""

import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from llm_testkit.reporting.steps import attach_screenshot, attach_text, step

if TYPE_CHECKING:
    from llm_testkit.pages.workspace_page import WorkspacePage

from llm_testkit.assertions.answers import assert_required_facts


@step("Check: question is visible")
def assert_ui_question_visible(workspace: "WorkspacePage", question: str) -> None:
    from playwright.sync_api import expect

    expect(workspace.user_question(question)).to_be_visible()


@step("Check: completed final answer is visible")
def assert_ui_completed_answer(workspace: "WorkspacePage", *, expected_text: str | None = None) -> str:
    from playwright.sync_api import expect

    # The response edit control scopes this locator to a persisted assistant reply.
    expect(workspace.final_answer).to_be_visible(timeout=workspace.answer_timeout_ms)
    expect(workspace.send_button).to_be_visible(timeout=workspace.answer_timeout_ms)
    # Send is disabled when the composer is empty, even after a completed reply.
    expect(workspace.final_answer).to_contain_text(re.compile(r"\S"), timeout=workspace.answer_timeout_ms)
    answer = workspace.final_answer.inner_text()
    assert answer.strip(), "Expected a non-empty final answer in the UI"
    attach_text(answer, name="Displayed final answer")
    if expected_text is not None:
        assert answer == expected_text, f"Expected {expected_text!r}, got {answer!r}"
    return answer


@step("Check: final answer contains required policy facts")
def assert_ui_policy_answer(workspace: "WorkspacePage", *, fact_patterns: Mapping[str, str]) -> str:
    answer = assert_ui_completed_answer(workspace)
    assert_required_facts(re.sub(r"\s+", " ", answer), fact_patterns=fact_patterns)
    return answer


@step("Check: uploaded document is listed as a source")
def assert_ui_document_source(workspace: "WorkspacePage", *, title: str) -> None:
    from playwright.sync_api import expect

    expect(workspace.source_document(title)).to_be_visible()
    attach_screenshot(workspace.page, name="Answer and source")


@step("Check: source details contain supporting policy passages")
def assert_ui_source_content(workspace: "WorkspacePage", *, title: str, fragments: Sequence[str]) -> None:
    from playwright.sync_api import expect

    details = workspace.source_details(title)
    expect(details).to_be_visible()
    expect(workspace.source_heading(title)).to_be_visible()
    for fragment in fragments:
        expect(details).to_contain_text(fragment)
    attach_screenshot(workspace.page, name="Document source details")


@step("Check: conversation history survives reload")
def assert_ui_history_preserved(workspace: "WorkspacePage", *, question: str, answer: str, url: str) -> None:
    from playwright.sync_api import expect

    expect(workspace.page).to_have_url(url)
    expect(workspace.user_question(question)).to_have_count(1)
    assert_ui_question_visible(workspace, question)
    expect(workspace.assistant_messages).to_have_count(1)
    expect(workspace.final_answer).to_be_visible()
    expect(workspace.final_answer).to_have_text(answer)
