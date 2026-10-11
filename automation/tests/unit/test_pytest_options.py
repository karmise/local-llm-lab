"""Pytest options: costly scenarios are opt-in, matrices are explicit and invalid selections are usage errors."""

import shutil

import pytest

from llm_testkit.reporting.steps import title
from test_support.paths import AUTOMATION_ROOT

pytestmark = pytest.mark.unit


@pytest.fixture
def runner(framework_pytester):
    """A child pytest session with the options plugin and a copy of the reviewed test data."""
    framework_pytester.makeconftest('pytest_plugins = ["llm_testkit.pytest_support.options"]')
    shutil.copytree(AUTOMATION_ROOT / "test_data", framework_pytester.path / "test_data")
    return framework_pytester


def run(runner, source, *args):
    runner.makepyfile(source)
    return runner.runpytest_subprocess("-q", "-rs", *args)


def collected(runner, source, *args) -> list[str]:
    """Node ids collected for ``source``, without running it."""
    result = run(runner, source, "--collect-only", *args)
    return [line.split("::", 1)[1] for line in result.stdout.lines if "::" in line]


def usage_error(runner, source, *args) -> pytest.RunResult:
    result = run(runner, source, *args)
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    return result


@pytest.mark.parametrize(("marker", "flag"), [
        pytest.param("ui", "--run-ui", id="ui"),
        pytest.param("browser", "--run-ui", id="browser"),
        pytest.param("live_quality", "--run-live-quality", id="live-quality"),
        pytest.param("golden", "--run-golden", id="golden"),
        pytest.param("conversation", "--run-conversation", id="conversation"),
        pytest.param("bias", "--run-bias", id="bias"),
        pytest.param("adversarial", "--run-adversarial", id="adversarial"),
        pytest.param("performance", "--run-performance", id="performance")])
@title("A costly scenario is skipped before its fixtures resolve unless its flag is given [{param_id}]")
def test_opt_in_scenario_is_skipped(runner, marker, flag):
    result = run(
            runner, f"""
        import pytest
        @pytest.mark.{marker}
        def test_costly(missing_fixture): pass
    """)

    result.assert_outcomes(skipped=1)
    result.stdout.fnmatch_lines([f"*{flag}*"])


@pytest.mark.parametrize(("marker", "flag"), [
        pytest.param("golden", "--run-golden", id="golden"),
        pytest.param("conversation", "--run-conversation", id="conversation"),
        pytest.param("bias", "--run-bias", id="bias"),
        pytest.param("adversarial", "--run-adversarial", id="adversarial")])
@title("Its flag enables a costly scenario [{param_id}]")
def test_opt_in_scenario_runs_with_flag(runner, marker, flag):
    result = run(
            runner, f"""
        import pytest
        @pytest.mark.{marker}
        def test_costly(): pass
    """, flag)

    result.assert_outcomes(passed=1)


@title("A performance batch runs only in its selected mode")
def test_performance_mode_selection(runner):
    source = """
        import pytest
        @pytest.mark.performance
        def test_health(): pass
        @pytest.mark.performance
        @pytest.mark.rag
        def test_rag(): pass
    """

    health = run(runner, source, "--run-performance")
    rag = run(runner, source, "--run-performance", "--performance-mode", "rag")

    assert (health.parseoutcomes(), rag.parseoutcomes()) == ({"passed": 1, "skipped": 1}, {"passed": 1, "skipped": 1})
    rag.stdout.fnmatch_lines(["*Select explicitly with --run-performance and --performance-mode*"])


PERFORMANCE = """
    import pytest
    @pytest.mark.performance
    @pytest.mark.rag
    def test_rag(): pass
"""


@pytest.mark.parametrize(("arguments", "message"), [
        pytest.param(["--performance-requests", "21"], "Performance budget: Performance request budget must be between",
        id="too-many-requests"),
        pytest.param(["--performance-requests", "2", "--performance-users", "3"],
        "Performance budget: Use one to four users", id="more-users-than-requests"),
        pytest.param(["--capture-rag"], "Performance batches do not support the single-answer capture mode",
        id="capture"),
        pytest.param(["--performance-p95", "0"], "Performance p95 must be finite and positive", id="zero-p95"),
        pytest.param(["--performance-p95", "nan"], "Performance p95 must be finite and positive", id="nan-p95")])
@title("An invalid selected performance batch is a usage error [{param_id}]")
def test_performance_usage_errors(runner, arguments, message):
    result = usage_error(runner, PERFORMANCE, "--run-performance", "--performance-mode", "rag", *arguments)

    result.stderr.fnmatch_lines([f"*{message}*"])


@title("A valid performance budget on the selected batch is accepted")
def test_performance_budget_accepted(runner):
    result = run(
            runner, PERFORMANCE, "--run-performance", "--performance-mode", "rag", "--performance-requests", "20",
            "--performance-users", "4", "--performance-p95", "0.5")

    result.assert_outcomes(passed=1)


@title("A skipped performance batch does not validate the performance budget")
def test_unselected_performance_budget_is_not_validated(runner):
    result = run(runner, PERFORMANCE, "--performance-requests", "0")

    result.assert_outcomes(skipped=1)


@title("Conversation checks refuse capture mode")
def test_conversation_refuses_capture(runner):
    result = usage_error(
            runner, """
        import pytest
        @pytest.mark.conversation
        def test_chat(): pass
    """, "--run-conversation", "--capture-rag")

    result.stderr.fnmatch_lines(["*Conversation checks do not support --capture-rag*"])


@title("Every reviewed conversation, bias and adversarial case becomes one test")
def test_catalog_cases_are_parametrized(runner):
    ids = collected(
            runner, """
        def test_conversation(conversation_case): pass
        def test_bias(bias_case): pass
        def test_adversarial(adversarial_case): pass
    """)

    assert ids == [
            *(
            f"test_conversation[{c}]"
            for c in ("greeting", "small_talk", "own_policy", "other_company", "mixed_request")), *(
            f"test_bias[{p}-{v}]" for p in ("gender_carryover", "age_carryover", "nationality_carryover")
            for v in ("1", "2")), *(
            f"test_adversarial[{c}]" for c in (
            "user_override", "role_spoof", "forged_source", "fabricated_benefit", "document_instruction",
            "conflicting_note"))]


@pytest.mark.parametrize(("catalog", "fixture", "message"), [
        pytest.param(
        "conversation-policy.json", "conversation_case", "Invalid conversation catalog", id="conversation"),
        pytest.param("bias-policy.json", "bias_case", "Invalid bias catalog", id="bias"),
        pytest.param("adversarial-policy.json", "adversarial_case", "Invalid adversarial catalog", id="adversarial"),
        pytest.param("golden-policy.json", "golden_case", "Invalid golden dataset", id="golden"),
        pytest.param("golden-policy.json", "bias_case", "Invalid bias catalog", id="bias-golden-dependency"),
        pytest.param(
        "golden-policy.json", "adversarial_case", "Invalid adversarial catalog", id="adversarial-golden-dependency")])
@title("An invalid reviewed catalog is a usage error naming the catalog [{param_id}]")
def test_invalid_catalog_is_usage_error(runner, catalog, fixture, message):
    (runner.path / "test_data" / catalog).write_text("{}")

    result = usage_error(runner, f"def test_case({fixture}): pass")

    result.stderr.fnmatch_lines([f"*{message}: *"])


@title("A missing catalog file is a usage error too")
def test_missing_catalog_is_usage_error(runner):
    (runner.path / "test_data/conversation-policy.json").unlink()

    result = usage_error(runner, "def test_case(conversation_case): pass")

    result.stderr.fnmatch_lines(["*Invalid conversation catalog: *"])


@title("Document attacks require capture mode to verify actual exposure; user attacks do not")
def test_document_attacks_require_capture(runner):
    source = "import pytest\n@pytest.mark.adversarial\ndef test_attack(adversarial_case): pass\n"

    refused = usage_error(runner, source, "--run-adversarial")
    user_attacks = run(runner, source, "--run-adversarial", "-k", "not document_instruction and not conflicting_note")
    captured = run(runner, source, "--run-adversarial", "--capture-rag")

    refused.stderr.fnmatch_lines(["*Document attacks require --capture-rag*"])
    user_attacks.assert_outcomes(passed=4, deselected=2)
    captured.assert_outcomes(passed=6)


PROMPTS = """
import pytest
@pytest.mark.prompt_regression
def test_prompt(prompt_variant): pass
"""


@pytest.mark.parametrize(("arguments", "ids"), [
        pytest.param([], ["test_prompt[baseline]", "test_prompt[grounded_v2]"], id="all-by-default"),
        pytest.param(["--rag-prompt", "grounded_v2", "--rag-prompt", "grounded_v2"], ["test_prompt[grounded_v2]"],
        id="deduplicated-selection")])
@title("Prompt regression runs the selected prompt variants once each [{param_id}]")
def test_prompt_variants(runner, arguments, ids):
    assert collected(runner, PROMPTS, *arguments) == ids


@title("Prompt regression is skipped unless enabled, and runs every variant when enabled")
def test_prompt_regression_is_opt_in(runner):
    skipped = run(runner, PROMPTS)
    enabled = run(runner, PROMPTS, "--run-prompt-regression")

    skipped.assert_outcomes(skipped=2)
    skipped.stdout.fnmatch_lines(["*--run-prompt-regression*"])
    enabled.assert_outcomes(passed=2)


@title("An unknown prompt variant is a usage error")
def test_unknown_prompt_variant(runner):
    result = usage_error(runner, PROMPTS, "--rag-prompt", "missing")

    result.stderr.fnmatch_lines(["*Invalid prompt selection: Unknown prompt variant*"])


@title("An invalid prompt catalog is a usage error")
def test_invalid_prompt_catalog(runner):
    (runner.path / "test_data/prompt-variants.json").write_text("{}")

    result = usage_error(runner, PROMPTS)

    result.stderr.fnmatch_lines(["*Invalid prompt selection: *"])


@title("Golden collection combines case, model and independent repetition")
def test_golden_matrix(runner):
    ids = collected(
            runner, """
        import pytest
        @pytest.mark.rag
        def test_golden(golden_case, generation_model, rag_iteration): pass
    """, "--rag-model", "model-a", "--rag-model", "model-b", "--rag-repeat", "2", "-k", "travel_allowance")

    assert ids == [f"test_golden[travel_allowance-{m}-run-{i}]" for m in ("model-a", "model-b") for i in (1, 2)]


RAG = """
import pytest
@pytest.mark.rag
def test_rag(generation_model, rag_iteration): pass
"""


@pytest.mark.parametrize(("arguments", "ids"), [
        pytest.param([], ["test_rag[qwen3.5:4b]", "test_rag[qwen2.5:7b]"], id="default-models"),
        pytest.param(["--rag-model", "model-a", "--rag-model", " model-a ", "--rag-model", "model-b"],
        ["test_rag[model-a]", "test_rag[model-b]"], id="deduplicated-models"),
        pytest.param(["--rag-model", "model-a", "--rag-repeat", "2"],
        ["test_rag[model-a-run-1]", "test_rag[model-a-run-2]"], id="repetitions")])
@title("RAG tests run each selected model once per independent repetition [{param_id}]")
def test_rag_matrix(runner, arguments, ids):
    assert collected(runner, RAG, *arguments) == ids


@pytest.mark.parametrize("marker", ["live_quality", "performance", "conversation"])
@title("Single-answer scenarios default to the primary model only [{marker}]")
def test_single_answer_scenarios_default_to_primary_model(runner, marker):
    ids = collected(
            runner, f"""
        import pytest
        @pytest.mark.rag
        @pytest.mark.{marker}
        def test_single(generation_model): pass
    """)

    assert ids == ["test_single[qwen3.5:4b]"]


@title("Repetitions are passed to tests that ask for the iteration")
def test_rag_iteration_values(runner):
    result = run(
            runner, """
        import pytest
        @pytest.mark.rag
        def test_rag(generation_model, rag_iteration):
            assert (generation_model, rag_iteration) in {("model-a", 1), ("model-a", 2)}
    """, "--rag-model", "model-a", "--rag-repeat", "2")

    result.assert_outcomes(passed=2)


@title("A model-only test does not need the iteration fixture")
def test_model_matrix_without_iteration(runner):
    result = run(
            runner, """
        import pytest
        @pytest.mark.rag
        def test_model_only(generation_model):
            assert generation_model == "model-a"
    """, "--rag-model", "model-a", "--rag-repeat", "2")

    result.assert_outcomes(passed=2)


@title("Tests without the rag marker are not given a model matrix")
def test_model_matrix_requires_rag_marker(runner):
    result = run(runner, "def test_plain(generation_model): pass")

    result.assert_outcomes(errors=1)


@pytest.mark.parametrize(("arguments", "message"), [
        pytest.param(["--rag-repeat", "0"], "--rag-repeat must be a positive integer", id="zero-repeat"),
        pytest.param(["--rag-repeat", "oops"], "--rag-repeat must be a positive integer", id="text-repeat"),
        pytest.param(["--rag-model", " "], "Model name must not be empty", id="blank-model"),
        pytest.param(["--judge-model", ""], "Model name must not be empty", id="blank-judge")])
@title("Invalid matrix options are usage errors [{param_id}]")
def test_invalid_matrix_options(runner, arguments, message):
    result = usage_error(runner, "def test_noop(): pass", *arguments)

    result.stderr.fnmatch_lines([f"*{message}*"])


@title("Deselected live tests do not validate live options")
@pytest.mark.parametrize("selector", [("-m", "unit"), ("-k", "offline")])
def test_deselected_live_tests_are_not_validated(runner, selector):
    result = run(
            runner, """
        import pytest
        @pytest.mark.unit
        def test_offline(): pass
        @pytest.mark.ui
        def test_ui(missing_browser): pass
        @pytest.mark.live_quality
        def test_live(missing_judge): pass
    """, *selector, "--run-ui", "--run-live-quality")

    result.assert_outcomes(passed=1, deselected=2)


@title("An explicit UI run requires the active Playwright plugin")
def test_ui_requires_playwright(runner):
    result = usage_error(runner, "import pytest\n@pytest.mark.ui\ndef test_ui(page): pass\n", "--run-ui")

    result.stderr.fnmatch_lines(["*UI checks require the UI extra and active pytest-playwright plugin*"])


LIVE = """
import pytest
@pytest.mark.rag
@pytest.mark.live_quality
def test_live(generation_model, rag_iteration): pass
"""


@pytest.mark.parametrize(("arguments", "message"), [
        pytest.param([], "Live quality requires --capture-rag and the capture Compose overlay", id="no-capture"),
        pytest.param(["--capture-rag", "--rag-repeat", "2"], "limited to one scenario/model/repetition per run",
        id="two-repetitions"),
        pytest.param(["--capture-rag", "--rag-model", "a", "--rag-model", "b"], "limited to one scenario/model",
        id="two-models"),
        pytest.param(["--capture-rag"], "Live quality requires the reporting extra and --alluredir", id="no-allure")])
@title("A live quality run is limited to one captured, reported scenario [{param_id}]")
def test_live_quality_usage_errors(runner, arguments, message):
    result = usage_error(runner, LIVE, "--run-live-quality", *arguments, "-p", "no:allure_pytest")

    result.stderr.fnmatch_lines([f"*{message}*"])


@title("The live quality options are not checked when no live scenario is selected")
def test_live_quality_without_live_items(runner):
    result = run(runner, "def test_plain(): pass", "--run-live-quality")

    result.assert_outcomes(passed=1)


@title("The CI and report options are registered with their documented choices")
def test_report_options_are_registered(runner):
    result = run(
            runner, """
        def test_options(pytestconfig):
            values = {name: pytestconfig.getoption(name) for name in (
                "ci_profile", "ci_models", "judge_model", "performance_mode", "performance_requests",
                "performance_users", "performance_p95", "rag_repeat", "capture_rag")}
            assert values == {"ci_profile": "smoke", "ci_models": "comparison", "judge_model": "qwen3.5:4b",
                "performance_mode": "health", "performance_requests": 1, "performance_users": 1,
                "performance_p95": 5.0, "rag_repeat": 1, "capture_rag": False}
    """, "--ci-profile", "smoke", "--ci-models", "comparison")

    result.assert_outcomes(passed=1)
