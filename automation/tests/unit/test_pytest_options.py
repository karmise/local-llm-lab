import pytest

from llm_testkit.reporting.steps import title

pytestmark = pytest.mark.unit


@pytest.fixture
def runner(framework_pytester: pytest.Pytester) -> pytest.Pytester:
    framework_pytester.makeconftest('pytest_plugins = ["llm_testkit.pytest_support.options"]')
    return framework_pytester


def test_opt_in_scenarios_skip_before_resolving_external_fixtures(runner: pytest.Pytester) -> None:
    runner.makepyfile("""
        import pytest
        @pytest.mark.ui
        def test_ui(missing_browser): pass
        @pytest.mark.browser
        def test_browser(missing_browser): pass
        @pytest.mark.live_quality
        def test_live(missing_judge): pass
        @pytest.mark.golden
        def test_golden(missing_model): pass
        @pytest.mark.conversation
        def test_conversation(missing_model): pass
    """)
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
    import shutil
    from pathlib import Path

    source = Path(__file__).resolve().parents[2] / "test_data"
    shutil.copytree(source, runner.path / "test_data")
    runner.makepyfile("""
        import pytest
        @pytest.mark.rag
        @pytest.mark.golden
        def test_golden(golden_case, generation_model, rag_iteration):
            assert golden_case.id == 'travel_allowance'
            assert generation_model in {'model-a', 'model-b'}
            assert rag_iteration in {1, 2}
    """)
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


@pytest.mark.parametrize("selector", [("-m", "unit"), ("-k", "offline")])
def test_deselected_live_tests_do_not_validate_live_options(
    runner: pytest.Pytester, selector: tuple[str, str]
) -> None:
    runner.makepyfile("""
        import pytest
        @pytest.mark.unit
        def test_offline(): pass
        @pytest.mark.ui
        def test_ui(missing_browser): pass
        @pytest.mark.live_quality
        def test_live(missing_judge): pass
    """)
    runner.runpytest_subprocess("-q", *selector, "--run-ui", "--run-live-quality").assert_outcomes(
        passed=1, deselected=2
    )


def test_model_matrix_deduplicates_names_and_retains_independent_repetitions(
    runner: pytest.Pytester,
) -> None:
    runner.makepyfile("""
        import pytest
        @pytest.mark.rag
        def test_rag(generation_model, rag_iteration):
            assert generation_model in {"model-a", "model-b"}
            assert rag_iteration in {1, 2}
    """)
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
    [
        ("--rag-repeat", "0"),
        ("--rag-repeat", "oops"),
        ("--rag-model", " "),
    ],
)
def test_invalid_matrix_options_are_usage_errors(
    runner: pytest.Pytester, arguments: tuple[str, str]
) -> None:
    runner.makepyfile("def test_noop(): pass")
    result = runner.runpytest_subprocess(*arguments)
    assert result.ret == pytest.ExitCode.USAGE_ERROR


def test_live_quality_budget_applies_to_selected_matrix(runner: pytest.Pytester) -> None:
    runner.makepyfile("""
        import pytest
        @pytest.mark.rag
        @pytest.mark.live_quality
        def test_live(generation_model, rag_iteration): pass
    """)
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
    runner.makepyfile("""
        import pytest
        @pytest.mark.rag
        def test_model_only(generation_model):
            assert generation_model == "model-a"
    """)
    runner.runpytest_subprocess(
        "--rag-model", "model-a", "--rag-repeat", "2", "-q"
    ).assert_outcomes(passed=2)
