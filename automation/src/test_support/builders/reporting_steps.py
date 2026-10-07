"""Named scenario preparation and test doubles."""

from contextlib import contextmanager
from unittest.mock import Mock

from llm_testkit.reporting import steps


def make_reported_step_stub(events):
    @contextmanager
    def reported_step(title: str):
        events.append(title)
        try:
            yield
        except RuntimeError:
            events.append("failure recorded")
            raise

    return reported_step


def prepare_reported_operation_case(backend, reported_step):
    if backend is not None:
        backend.step.side_effect = reported_step


def make_operation_stub(failure, result):
    @steps.step("API: upload document")
    def operation(secret: str, *, fail: bool = False) -> object:
        if fail:
            raise failure
        return result

    return operation


def check_reported_operation_outcome(
    backend,
):
    if backend is not None:
        assert backend.attach.call_count == 0
        assert backend.dynamic.parameter.call_count == 0


def make_missing_allure_stub():
    def missing_allure(name: str):
        raise ModuleNotFoundError("Allure is not installed", name="allure")

    return missing_allure


def make_missing_transitive_dependency_stub():
    def missing_transitive_dependency(name: str):
        raise ModuleNotFoundError("A dependency is broken", name="other_dependency")

    return missing_transitive_dependency


def make_reporting_backend(
    reporting_installed,
):
    backend = Mock() if reporting_installed else None

    return backend


def expected_reporting_events(
    reporting_installed,
):
    expected = (
        ["API: upload document", "API: upload document", "failure recorded"]
        if reporting_installed
        else []
    )

    return expected
