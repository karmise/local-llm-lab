"""Page Object regressions in a real browser with local, deterministic HTML."""

from typing import TYPE_CHECKING

import pytest

from llm_testkit import assertions
from llm_testkit.reporting.steps import title
from test_support.fixtures.browser import chat_simulation as chat_simulation
from test_support.fixtures.browser import workspace as workspace

if TYPE_CHECKING:
    from llm_testkit.pages.workspace_page import WorkspacePage


pytestmark = pytest.mark.browser


@title("Browser waits for a completed answer even without sources")
def test_answer_without_sources_is_still_a_completed_answer(workspace: "WorkspacePage") -> None:
    workspace.send_question("Unknown policy?")
    assertions.assert_ui_completed_answer(workspace, expected_text="New final answer")


@title("Browser waits for the new reply rather than a previous answer")
def test_second_question_waits_for_its_own_reply(
    workspace: "WorkspacePage", chat_simulation
) -> None:
    chat_simulation.show_previous_answer("Previous final answer")
    workspace.send_question("Second question?")
    assertions.assert_ui_completed_answer(workspace, expected_text="New final answer")


@title("Browser waits for delayed nonempty answer content")
def test_completed_answer_waits_for_nonempty_content(
    workspace: "WorkspacePage", chat_simulation
) -> None:
    chat_simulation.show_answer_with_delayed_content()
    assertions.assert_ui_completed_answer(workspace, expected_text="Ready")


@title("Browser accepts a completed reply with an empty composer")
def test_completed_answer_allows_disabled_send_with_empty_composer(
    workspace: "WorkspacePage",
    chat_simulation,
) -> None:
    chat_simulation.show_completed_answer_with_disabled_send()
    assertions.assert_ui_completed_answer(workspace, expected_text="Complete")
