from contextlib import contextmanager
from unittest.mock import Mock

import pytest

from llm_testkit.reporting.steps import title
from llm_testkit import assertions
from llm_testkit.reporting import steps

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("reporting_installed", [False, True])
@title('Reported operation preserves results and failures without exposing arguments [{param_id}]')
def test_reported_operation_preserves_result_and_failure_without_exposing_arguments(
    monkeypatch: pytest.MonkeyPatch, reporting_installed: bool,
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

    assertions.assert_field_equals({"same_result": operation("private-key") is result}, "same_result", True)
    with pytest.raises(RuntimeError, match="Operation failed") as caught:
        operation("private-key", fail=True)
    assertions.assert_field_equals({"same_exception": caught.value is failure}, "same_exception", True)
    expected = ["API: upload document", "API: upload document", "failure recorded"] if reporting_installed else []
    assertions.assert_field_equals({"events": events}, "events", expected)
    assertions.assert_field_equals({"wrapped": operation.__wrapped__.__name__}, "wrapped", "operation")
    if backend is not None:
        assertions.assert_field_equals({"attachments": backend.attach.call_count}, "attachments", 0)
        assertions.assert_field_equals({"parameters": backend.dynamic.parameter.call_count}, "parameters", 0)


@title('Optional reporter tolerates missing Allure but propagates other dependency errors')
def test_optional_reporter_handles_only_the_missing_allure_package(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_allure(name: str):
        raise ModuleNotFoundError("Allure is not installed", name="allure")

    monkeypatch.setattr(steps, "import_module", missing_allure)
    assertions.assert_field_equals({"backend": steps._backend()}, "backend", None)

    def missing_transitive_dependency(name: str):
        raise ModuleNotFoundError("A dependency is broken", name="other_dependency")

    monkeypatch.setattr(steps, "import_module", missing_transitive_dependency)
    with pytest.raises(ModuleNotFoundError, match="dependency is broken"):
        steps._backend()
