from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock

import pytest

from llm_testkit.reporting import steps
from llm_testkit.reporting.live_quality import generate_captured_answer
from llm_testkit.reporting.steps import title

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("reporting_installed", [False, True])
@title("Reported operation preserves results and failures without exposing arguments [{param_id}]")
def test_reported_operation_preserves_result_and_failure_without_exposing_arguments(
    monkeypatch: pytest.MonkeyPatch,
    reporting_installed: bool,
) -> None:
    backend = Mock() if reporting_installed else None
    events = []

    @contextmanager
    def reported_step(title: str):
        events.append(title)
        try:
            yield
        except RuntimeError:
            events.append("failure recorded")
            raise

    if backend is not None:
        backend.step.side_effect = reported_step
    monkeypatch.setattr(steps, "_backend", lambda: backend)
    result = object()
    failure = RuntimeError("Operation failed")

    @steps.step("API: upload document")
    def operation(secret: str, *, fail: bool = False) -> object:
        if fail:
            raise failure
        return result

    assert operation("private-key") is result
    with pytest.raises(RuntimeError, match="Operation failed") as caught:
        operation("private-key", fail=True)
    assert caught.value is failure
    expected = (
        ["API: upload document", "API: upload document", "failure recorded"]
        if reporting_installed
        else []
    )
    assert events == expected
    assert operation.__wrapped__.__name__ == "operation"
    if backend is not None:
        assert backend.attach.call_count == 0
        assert backend.dynamic.parameter.call_count == 0


@title("Optional reporter tolerates missing Allure but propagates other dependency errors")
def test_optional_reporter_handles_only_the_missing_allure_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_allure(name: str):
        raise ModuleNotFoundError("Allure is not installed", name="allure")

    monkeypatch.setattr(steps, "import_module", missing_allure)
    assert steps._backend() is None

    def missing_transitive_dependency(name: str):
        raise ModuleNotFoundError("A dependency is broken", name="other_dependency")

    monkeypatch.setattr(steps, "import_module", missing_transitive_dependency)
    with pytest.raises(ModuleNotFoundError, match="dependency is broken"):
        steps._backend()


def test_live_capture_uses_the_shared_profile_reference(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(steps, "_backend", lambda: None)
    chat = Mock()
    profile = {"question": "A different policy question?", "reference": "Its expected answer."}

    generate_captured_answer(chat, profile, tmp_path / "sample.json")

    chat.assert_called_once_with(profile["question"], profile["reference"])
