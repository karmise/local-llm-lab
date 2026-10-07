import shutil

import pytest

from llm_testkit.reporting.steps import title
from test_support.data.pytest_options import (
    DESELECTED_LIVE_TESTS_DO_NOT_VALIDATE_LIVE_OPTIONS_SELECTOR_CASES,
    INVALID_MATRIX_OPTIONS_ARE_USAGE_ERRORS_ARGUMENTS_CASES,
)
from test_support.data.scripts.pytest_options import (
    DESELECTED_LIVE_TESTS_DO_NOT_VALIDATE_LIVE_OPTIONS_MAKEPYFILE_SOURCE,
    GOLDEN_COLLECTION_FORMS_CASE_MODEL_REPEAT_MATRIX_MAKEPYFILE_SOURCE,
    LIVE_QUALITY_BUDGET_APPLIES_TO_SELECTED_MATRIX_MAKEPYFILE_SOURCE,
    MODEL_MATRIX_DEDUPLICATES_NAMES_AND_RETAINS_INDEPENDENT_REPETITIONS_MAKEPYFILE_SOURCE,
    MODEL_MATRIX_DOES_NOT_REQUIRE_AN_UNUSED_ITERATION_FIXTURE_MAKEPYFILE_SOURCE,
    OPT_IN_SCENARIOS_SKIP_BEFORE_RESOLVING_EXTERNAL_FIXTURES_MAKEPYFILE_SOURCE,
)
from test_support.fixtures.unit_pytest_options import runner as runner
from test_support.paths import AUTOMATION_ROOT

pytestmark = pytest.mark.unit


def test_opt_in_scenarios_skip_before_resolving_external_fixtures(runner: pytest.Pytester) -> None:
    runner.makepyfile(OPT_IN_SCENARIOS_SKIP_BEFORE_RESOLVING_EXTERNAL_FIXTURES_MAKEPYFILE_SOURCE)
    runner.runpytest_subprocess("-q").assert_outcomes(skipped=5)


@title("Explicit golden flag enables selected acceptance scenarios")
def test_explicit_golden_flag_enables_selected_cases(runner: pytest.Pytester) -> None:
    runner.makepyfile("""
        import pytest
        @pytest.mark.golden
        def test_golden(): pass
    """)
    runner.runpytest_subprocess("--run-golden", "-q").assert_outcomes(passed=1)


@title("Golden collection combines case, model and independent repetition selection")
def test_golden_collection_forms_case_model_repeat_matrix(runner: pytest.Pytester) -> None:

    source = AUTOMATION_ROOT / "test_data"
    shutil.copytree(source, runner.path / "test_data")
    runner.makepyfile(GOLDEN_COLLECTION_FORMS_CASE_MODEL_REPEAT_MATRIX_MAKEPYFILE_SOURCE)
    runner.runpytest_subprocess(
        "--run-golden",
        "--rag-model",
        "model-a",
        "--rag-model",
        "model-b",
        "--rag-repeat",
        "2",
        "-k",
        "travel_allowance",
        "-q",
    ).assert_outcomes(passed=4, deselected=60)


@pytest.mark.parametrize(
    "selector", DESELECTED_LIVE_TESTS_DO_NOT_VALIDATE_LIVE_OPTIONS_SELECTOR_CASES
)
def test_deselected_live_tests_do_not_validate_live_options(
    runner: pytest.Pytester, selector: tuple[str, str]
) -> None:
    runner.makepyfile(DESELECTED_LIVE_TESTS_DO_NOT_VALIDATE_LIVE_OPTIONS_MAKEPYFILE_SOURCE)
    runner.runpytest_subprocess("-q", *selector, "--run-ui", "--run-live-quality").assert_outcomes(
        passed=1, deselected=2
    )


def test_model_matrix_deduplicates_names_and_retains_independent_repetitions(
    runner: pytest.Pytester,
) -> None:
    runner.makepyfile(
        MODEL_MATRIX_DEDUPLICATES_NAMES_AND_RETAINS_INDEPENDENT_REPETITIONS_MAKEPYFILE_SOURCE
    )
    result = runner.runpytest_subprocess(
        "-q",
        "--rag-model",
        "model-a",
        "--rag-model",
        " model-a ",
        "--rag-model",
        "model-b",
        "--rag-repeat",
        "2",
    )
    result.assert_outcomes(passed=4)


@pytest.mark.parametrize(
    "arguments",
    INVALID_MATRIX_OPTIONS_ARE_USAGE_ERRORS_ARGUMENTS_CASES,
)
def test_invalid_matrix_options_are_usage_errors(
    runner: pytest.Pytester, arguments: tuple[str, str]
) -> None:
    runner.makepyfile("def test_noop(): pass")
    result = runner.runpytest_subprocess(*arguments)
    assert result.ret == pytest.ExitCode.USAGE_ERROR


def test_live_quality_budget_applies_to_selected_matrix(runner: pytest.Pytester) -> None:
    runner.makepyfile(LIVE_QUALITY_BUDGET_APPLIES_TO_SELECTED_MATRIX_MAKEPYFILE_SOURCE)
    result = runner.runpytest_subprocess("--run-live-quality", "--capture-rag", "--rag-repeat", "2")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*limited to one scenario/model/repetition*"])


def test_explicit_ui_run_requires_active_browser_plugin(runner: pytest.Pytester) -> None:
    runner.makepyfile("""
        import pytest
        @pytest.mark.ui
        def test_ui(page): pass
    """)
    result = runner.runpytest_subprocess("--run-ui")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*active pytest-playwright plugin*"])


def test_model_matrix_does_not_require_an_unused_iteration_fixture(runner: pytest.Pytester) -> None:
    runner.makepyfile(MODEL_MATRIX_DOES_NOT_REQUIRE_AN_UNUSED_ITERATION_FIXTURE_MAKEPYFILE_SOURCE)
    runner.runpytest_subprocess(
        "--rag-model", "model-a", "--rag-repeat", "2", "-q"
    ).assert_outcomes(passed=2)
