"""Optional Allure reporting: static titles, argument-free steps and explicit attachments, all no-ops without Allure."""

from unittest.mock import MagicMock, Mock, call

import pytest

from llm_testkit.reporting import steps
from llm_testkit.reporting.live_quality import generate_captured_answer
from llm_testkit.reporting.steps import title

pytestmark = pytest.mark.unit


@pytest.fixture
def allure(monkeypatch):
    """Allure as an installed backend, recording every call."""
    backend = MagicMock(name="allure")
    monkeypatch.setattr(steps, "_backend", lambda: backend)
    return backend


@pytest.fixture
def no_allure(monkeypatch):
    """Allure is not installed."""
    monkeypatch.setattr(steps, "_backend", lambda: None)


def missing(name):
    def import_module(module):
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)

    return import_module


@title("The backend is the imported Allure module when it is installed")
def test_backend_imports_allure(monkeypatch):
    imported = Mock()
    monkeypatch.setattr(steps, "import_module", imported)

    assert steps._backend() is imported.return_value
    imported.assert_called_once_with("allure")


@title("A missing Allure package means no backend")
def test_backend_is_none_without_allure(monkeypatch):
    monkeypatch.setattr(steps, "import_module", missing("allure"))

    assert steps._backend() is None


@title("A broken dependency inside an installed Allure package is not hidden")
def test_backend_propagates_other_missing_modules(monkeypatch):
    monkeypatch.setattr(steps, "import_module", missing("pluggy"))

    with pytest.raises(ModuleNotFoundError, match="pluggy"):
        steps._backend()


@title("A title uses Allure's native title decorator")
def test_title_uses_allure(allure):
    def function():
        pass

    decorated = steps.title("Readable title")(function)

    allure.title.assert_called_once_with("Readable title")
    allure.title.return_value.assert_called_once_with(function)
    assert decorated is allure.title.return_value.return_value


@title("Without Allure a title leaves the function unchanged")
def test_title_without_allure(no_allure):
    def function():
        pass

    assert steps.title("Readable title")(function) is function


def operation(secret, *, fail=False):
    """An operation whose arguments must never appear in the report."""
    if fail:
        raise RuntimeError("Operation failed")
    return {"secret": secret}


@title("A step reports its static title only and returns the operation's result")
def test_step_reports_title_without_arguments(allure):
    reported = steps.step("Ask the assistant")(operation)

    result = reported("private-key")

    assert result == {"secret": "private-key"}
    assert allure.mock_calls == [
            call.step("Ask the assistant"),
            call.step().__enter__(),
            call.step().__exit__(None, None, None)]
    assert reported.__name__ == "operation" and reported.__wrapped__ is operation


@title("A failing step re-raises the original exception after closing the Allure step")
def test_step_propagates_failure(allure):
    reported = steps.step("Ask the assistant")(operation)

    with pytest.raises(RuntimeError, match="Operation failed") as failure:
        reported("private-key", fail=True)

    exit_args = allure.step.return_value.__exit__.call_args.args
    assert exit_args[0] is RuntimeError and exit_args[1] is failure.value


@title("Without Allure a step still runs the operation and propagates failures")
def test_step_without_allure(no_allure):
    reported = steps.step("Ask the assistant")(operation)

    assert reported("private-key") == {"secret": "private-key"}
    with pytest.raises(RuntimeError, match="Operation failed"):
        reported("private-key", fail=True)


@title("The backend is looked up when a step runs, not when it is decorated")
def test_step_resolves_backend_at_call_time(monkeypatch):
    lookups = []
    monkeypatch.setattr(steps, "_backend", lambda: lookups.append(1))
    reported = steps.step("Ask the assistant")(operation)

    reported("private-key")

    assert lookups == [1]


@title("Feature and story metadata are sent to Allure")
def test_set_metadata(allure):
    steps.set_metadata(feature="RAG quality", story="Saved answer")

    assert allure.mock_calls == [call.dynamic.feature("RAG quality"), call.dynamic.story("Saved answer")]


@title("Text is attached as plain text")
def test_attach_text(allure):
    steps.attach_text("answer", name="Final answer")

    allure.attach.assert_called_once_with("answer", name="Final answer", attachment_type="text/plain")


@title("A file is attached by path with its media type and extension")
def test_attach_file(allure, tmp_path):
    steps.attach_file(tmp_path / "sample.json", name="Sample", media_type="application/json", extension="json")

    allure.attach.file.assert_called_once_with(
            str(tmp_path / "sample.json"), name="Sample", attachment_type="application/json", extension="json")


@title("A screenshot is a full-page capture with animations disabled, attached as PNG")
def test_attach_screenshot(allure):
    page = Mock()

    steps.attach_screenshot(page, name="Workspace")

    page.screenshot.assert_called_once_with(full_page=True, animations="disabled")
    allure.attach.assert_called_once_with(
            page.screenshot.return_value, name="Workspace", attachment_type="image/png", extension="png")


@title("Browser artifacts attach screenshots and traces in name order and ignore other files")
def test_attach_browser_artifacts(allure, tmp_path):
    for name in ("b.png", "a.zip", "notes.txt", "c.png"):
        (tmp_path / name).write_text("")

    steps.attach_browser_artifacts(tmp_path)

    assert allure.attach.file.call_args_list == [
            call(str(tmp_path / "a.zip"), name="a.zip", attachment_type="application/zip", extension="zip"),
            call(str(tmp_path / "b.png"), name="b.png", attachment_type="image/png", extension="png"),
            call(str(tmp_path / "c.png"), name="c.png", attachment_type="image/png", extension="png")]


@title("Requirements and qualification phases become Allure labels")
def test_traceability_labels(allure):
    steps.traceability_labels(["REQ-1", "REQ-2"], ["OQ"])

    assert allure.dynamic.label.call_args_list == [
            call("requirement", "REQ-1"),
            call("requirement", "REQ-2"),
            call("qualification_phase", "OQ")]


@pytest.mark.parametrize(
        "report", [
        pytest.param(lambda tmp_path: steps.set_metadata(feature="f", story="s"), id="metadata"),
        pytest.param(lambda tmp_path: steps.attach_text("t", name="n"), id="text"),
        pytest.param(
        lambda tmp_path: steps.attach_file(tmp_path / "f", name="n", media_type="m", extension="e"), id="file"),
        pytest.param(lambda tmp_path: steps.attach_screenshot(Mock(), name="n"), id="screenshot"),
        pytest.param(lambda tmp_path: steps.traceability_labels(["REQ-1"], ["OQ"]), id="labels")])
@title("Without Allure every reporting call is a no-op [{param_id}]")
def test_reporting_without_allure_is_a_no_op(no_allure, tmp_path, report):
    assert report(tmp_path) is None


@title("Without Allure a screenshot is not even taken")
def test_screenshot_without_allure_is_not_taken(no_allure):
    page = Mock()

    steps.attach_screenshot(page, name="Workspace")

    page.screenshot.assert_not_called()


@title("A live capture asks the profile's question with its reference and attaches the sample")
def test_live_capture_uses_the_profile(allure, tmp_path):
    chat = Mock()
    profile = {"question": "A different policy question?", "reference": "Its expected answer."}

    generate_captured_answer(chat, profile, tmp_path / "sample.json")

    chat.assert_called_once_with("A different policy question?", "Its expected answer.")
    allure.attach.file.assert_called_once_with(
            str(tmp_path / "sample.json"), name="Captured evaluation sample", attachment_type="application/json",
            extension="json")
