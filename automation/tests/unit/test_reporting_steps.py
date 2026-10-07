from pathlib import Path

import pytest

from llm_testkit.reporting import steps
from llm_testkit.reporting.live_quality import generate_captured_answer
from llm_testkit.reporting.steps import title
from test_support.assertions import errors as errors
from test_support.assertions import mocks as mock_checks
from test_support.assertions import values as value_checks
from test_support.assertions.reporting_steps import check_reported_operation_outcome
from test_support.builders.reporting_steps import (
    expected_reporting_events,
    make_missing_allure_stub,
    make_missing_transitive_dependency_stub,
    make_operation_stub,
    make_reported_step_stub,
    make_reporting_backend,
    prepare_reported_operation_case,
)
from test_support.data.reporting_steps import (
    REPORTED_OPERATION_REPORTING_INSTALLED_CASES,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "reporting_installed",
    REPORTED_OPERATION_REPORTING_INSTALLED_CASES,
)
@title("Reported operation preserves results and failures without exposing arguments [{param_id}]")
def test_reported_operation_preserves_result_and_failure_without_exposing_arguments(
    monkeypatch: pytest.MonkeyPatch,
    reporting_installed: bool,
) -> None:
    backend = make_reporting_backend(reporting_installed)
    events = []

    reported_step = make_reported_step_stub(events)

    prepare_reported_operation_case(backend, reported_step)
    monkeypatch.setattr(steps, "_backend", lambda: backend)
    result = object()
    failure = RuntimeError("Operation failed")

    operation = make_operation_stub(failure, result)

    value_checks.identical(operation("private-key"), result)
    caught = errors.rejects(
        lambda: operation("private-key", fail=True), expected=RuntimeError, match="Operation failed"
    )
    value_checks.identical(caught.value, failure)
    expected = expected_reporting_events(reporting_installed)
    value_checks.equal(events, expected)
    value_checks.equal(operation.__wrapped__.__name__, "operation")
    check_reported_operation_outcome(backend)


@title("Optional reporter tolerates missing Allure but propagates other dependency errors")
def test_optional_reporter_handles_only_the_missing_allure_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_allure = make_missing_allure_stub()

    monkeypatch.setattr(steps, "import_module", missing_allure)
    value_checks.identical(steps._backend(), None)

    missing_transitive_dependency = make_missing_transitive_dependency_stub()

    monkeypatch.setattr(steps, "import_module", missing_transitive_dependency)
    errors.rejects(
        lambda: steps._backend(), expected=ModuleNotFoundError, match="dependency is broken"
    )


def test_live_capture_uses_the_shared_profile_reference(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mock_factory
) -> None:
    monkeypatch.setattr(steps, "_backend", lambda: None)
    chat = mock_factory()
    profile = {"question": "A different policy question?", "reference": "Its expected answer."}

    generate_captured_answer(chat, profile, tmp_path / "sample.json")

    mock_checks.called_once_with(chat, profile["question"], profile["reference"])
