"""CLI options and selection rules for costly integration scenarios."""

import importlib.util
import math
from pathlib import Path

import pytest

from llm_testkit.datasets.adversarial import load_adversarial_cases
from llm_testkit.datasets.bias import load_bias_cases
from llm_testkit.datasets.conversation import load_conversation_catalog
from llm_testkit.datasets.golden import load_golden_dataset
from llm_testkit.datasets.prompts import load_prompt_catalog
from llm_testkit.performance.runner import validate_budget

DEFAULT_RAG_MODELS = ("qwen3.5:4b", "qwen2.5:7b")


def _positive_repeat(value: str) -> int:
    try:
        count = int(value)
    except ValueError:
        raise pytest.UsageError("--rag-repeat must be a positive integer") from None
    if count < 1:
        raise pytest.UsageError("--rag-repeat must be a positive integer")
    return count


def _model_name(value: str) -> str:
    name = value.strip()
    if not name:
        raise pytest.UsageError("Model name must not be empty")
    return name


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--ci-profile", choices=("smoke", "curated"), help="Expected live CI benchmark scope")
    parser.addoption("--ci-models", choices=("primary", "comparison"), help="Expected live CI generation models")
    parser.addoption("--ci-revision", help="Exact commit SHA tested by the live CI producer")
    parser.addoption("--benchmark-report", type=Path, help="Saved dataset benchmark for offline Allure reporting")
    parser.addoption(
            "--run-conversation", action="store_true",
            help="Enable Chat-mode conversational acceptance checks (one generation per case).")
    parser.addoption("--run-bias", action="store_true", help="Enable paired counterfactual policy acceptance checks")
    parser.addoption("--run-performance", action="store_true", help="Enable bounded performance batches")
    parser.addoption("--performance-mode", choices=("health", "rag"), default="health")
    parser.addoption("--performance-requests", type=int, default=1)
    parser.addoption("--performance-users", type=int, default=1)
    parser.addoption(
            "--performance-p95", type=float, default=5.0,
            help="Explicit maximum p95 seconds; use a model-appropriate threshold for RAG")
    parser.addoption("--quality-gates", type=Path, help="Explicit experimental quality gates for saved evidence")
    parser.addoption("--run-adversarial", action="store_true", help="Enable curated adversarial policy scenarios")
    parser.addoption(
            "--run-prompt-regression", action="store_true",
            help="Enable versioned prompt comparisons against golden cases")
    parser.addoption(
            "--rag-prompt", action="append", dest="rag_prompts", help="Select prompt variant IDs for prompt regression")
    parser.addoption("--relevance-report", type=Path, help="Existing precision/recall evidence")
    parser.addoption(
            "--run-golden", action="store_true",
            help="Enable source-bound golden RAG scenarios (one generation per selected case).")
    parser.addoption(
            "--run-ui", action="store_true",
            help="Enable browser checks (application UI scenarios generate one answer each).")
    parser.addoption(
            "--run-live-quality", action="store_true",
            help="Enable one live RAG-to-Allure scenario (three model calls maximum).")
    parser.addoption(
            "--judge-model", type=_model_name, default="qwen3.5:4b",
            help="Local judge model for the live quality scenario.")
    parser.addoption("--quality-sample", type=Path, help="Captured sample for an offline quality report.")
    parser.addoption("--correctness-report", type=Path, help="Optional existing golden-reference correctness evidence.")
    parser.addoption("--faithfulness-report", type=Path, help="Existing judge report bound to the sample checksum.")
    parser.addoption(
            "--capture-rag", action="store_true", default=False,
            help="Save actual model messages and evaluation samples; requires compose.capture.yaml.")
    parser.addoption(
            "--rag-model", action="append", type=_model_name, dest="rag_models", default=None,
            help="Generation model for RAG tests; repeat to select multiple models.")
    parser.addoption(
            "--rag-repeat", type=_positive_repeat, default=1,
            help="Independent repetitions per RAG scenario and model (default: 1).")


_SELECTION_ERRORS = pytest.StashKey[list[str]]()


def _catalog(metafunc: pytest.Metafunc, label: str, load):
    """Load a reviewed catalog for parametrization; record a failure to report as a usage error after collection."""
    try:
        return load(metafunc.config.rootpath / "test_data")
    except (ValueError, OSError) as error:
        metafunc.config.stash.setdefault(_SELECTION_ERRORS, []).append(f"{label}: {error}")
        return None


def _golden(root: Path):
    return load_golden_dataset(root / "golden-policy.json", root / "company-policy.txt")


def _parametrize(metafunc: pytest.Metafunc, name: str, cases, ids) -> None:
    metafunc.parametrize(name, cases or [], ids=ids if cases else [])


def _parametrize_catalogs(metafunc: pytest.Metafunc) -> None:
    names = metafunc.fixturenames
    if "conversation_case" in names:
        catalog = _catalog(
                metafunc, "Invalid conversation catalog",
                lambda root: load_conversation_catalog(root / "conversation-policy.json", root / "company-policy.txt"))
        cases = catalog.cases if catalog else None
        _parametrize(metafunc, "conversation_case", cases, [c.id for c in cases or []])
    if "bias_case" in names:
        cases = _catalog(
                metafunc, "Invalid bias catalog",
                lambda root: load_bias_cases(root / "bias-policy.json", _golden(root)))
        _parametrize(metafunc, "bias_case", cases, [f"{c.pair_id}-{c.variant_id}" for c in cases or []])
    if "adversarial_case" in names:
        cases = _catalog(
                metafunc, "Invalid adversarial catalog",
                lambda root: load_adversarial_cases(root / "adversarial-policy.json", _golden(root)))
        _parametrize(metafunc, "adversarial_case", cases, [c.id for c in cases or []])
    if metafunc.definition.get_closest_marker("prompt_regression"):
        selected = metafunc.config.getoption("rag_prompts")

        def load_variants(root: Path):
            catalog = load_prompt_catalog(root / "prompt-variants.json")
            variants = {v.id: v for v in catalog.variants}
            chosen = list(dict.fromkeys(selected or variants))
            if any(identifier not in variants for identifier in chosen):
                raise ValueError("Unknown prompt variant")
            return [variants[identifier] for identifier in chosen]

        variants = _catalog(metafunc, "Invalid prompt selection", load_variants)
        _parametrize(metafunc, "prompt_variant", variants, [v.id for v in variants or []])
    if "golden_case" in names:
        dataset = _catalog(metafunc, "Invalid golden dataset", _golden)
        cases = dataset.cases if dataset else None
        _parametrize(metafunc, "golden_case", cases, [case.id for case in cases or []])


def _parametrize_models(metafunc: pytest.Metafunc) -> None:
    definition = metafunc.definition
    if not definition.get_closest_marker("rag") or "generation_model" not in metafunc.fixturenames:
        return
    single_answer = any(definition.get_closest_marker(m) for m in ("live_quality", "performance", "conversation"))
    defaults = ("qwen3.5:4b", ) if single_answer else DEFAULT_RAG_MODELS
    models = list(dict.fromkeys(metafunc.config.getoption("rag_models") or defaults))
    count = metafunc.config.getoption("rag_repeat")
    cases = [(model, iteration) for model in models for iteration in range(1, count + 1)]
    ids = [model if count == 1 else f"{model}-run-{iteration}" for model, iteration in cases]
    if "rag_iteration" in metafunc.fixturenames:
        metafunc.parametrize(("generation_model", "rag_iteration"), cases, ids=ids)
    else:
        metafunc.parametrize("generation_model", [model for model, _ in cases], ids=ids)


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    _parametrize_catalogs(metafunc)
    _parametrize_models(metafunc)


def _skip_unless_enabled(
        config: pytest.Config, items: list[pytest.Item], markers: tuple[str, ...], option: str,
        flag: str) -> list[pytest.Item]:
    """Skip items with any of ``markers`` unless ``option`` is set; return the selected items with them."""
    marked = [item for item in items if any(item.get_closest_marker(m) for m in markers)]
    if not config.getoption(option):
        for item in marked:
            item.add_marker(pytest.mark.skip(reason=f"Enable explicitly with {flag}"))
        return []
    return marked


def _check_conversation(config: pytest.Config, selected: list[pytest.Item]) -> None:
    if selected and config.getoption("capture_rag"):
        raise pytest.UsageError("Conversation checks do not support --capture-rag: small talk is not a RAG sample")


def _check_performance(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        if not item.get_closest_marker("performance"):
            continue
        mode = "rag" if item.get_closest_marker("rag") else "health"
        if not config.getoption("run_performance") or mode != config.getoption("performance_mode"):
            item.add_marker(pytest.mark.skip(reason="Select explicitly with --run-performance and --performance-mode"))
            continue
        try:
            validate_budget(config.getoption("performance_requests"), config.getoption("performance_users"))
        except ValueError as error:
            raise pytest.UsageError(f"Performance budget: {error}") from error
        if mode == "rag" and config.getoption("capture_rag"):
            raise pytest.UsageError("Performance batches do not support the single-answer capture mode")
        p95 = config.getoption("performance_p95")
        if not math.isfinite(p95) or p95 <= 0:
            raise pytest.UsageError("Performance p95 must be finite and positive")


def _check_document_attacks(config: pytest.Config, selected: list[pytest.Item]) -> None:
    for item in selected:
        case = item.callspec.params.get("adversarial_case") if hasattr(item, "callspec") else None
        if case is not None and case.document_appendix and not config.getoption("capture_rag"):
            raise pytest.UsageError("Document attacks require --capture-rag to verify actual retrieved attack exposure")


def _check_browser(config: pytest.Config, selected: list[pytest.Item]) -> None:
    if selected and not config.pluginmanager.hasplugin("playwright"):
        raise pytest.UsageError("UI checks require the UI extra and active pytest-playwright plugin")


def _check_live_quality(config: pytest.Config, selected: list[pytest.Item]) -> None:
    if not selected:
        return
    if not config.getoption("capture_rag"):
        raise pytest.UsageError("Live quality requires --capture-rag and the capture Compose overlay")
    if len(selected) != 1:
        raise pytest.UsageError("Live quality is limited to one scenario/model/repetition per run")
    if not config.getoption("allure_report_dir", default=None):
        raise pytest.UsageError("Live quality requires the reporting extra and --alluredir")
    if importlib.util.find_spec("ragas") is None:
        raise pytest.UsageError("Live quality requires the evaluation extra")


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    # Validate only selected tests, after pytest applies -k and -m filters.
    errors = config.stash.get(_SELECTION_ERRORS, [])
    if errors:
        raise pytest.UsageError(errors[0])
    _check_conversation(
            config, _skip_unless_enabled(config, items, ("conversation", ), "run_conversation", "--run-conversation"))
    _skip_unless_enabled(config, items, ("bias", ), "run_bias", "--run-bias")
    _check_performance(config, items)
    _check_document_attacks(
            config, _skip_unless_enabled(config, items, ("adversarial", ), "run_adversarial", "--run-adversarial"))
    _skip_unless_enabled(config, items, ("prompt_regression", ), "run_prompt_regression", "--run-prompt-regression")
    _skip_unless_enabled(config, items, ("golden", ), "run_golden", "--run-golden")
    _check_browser(config, _skip_unless_enabled(config, items, ("ui", "browser"), "run_ui", "--run-ui"))
    _check_live_quality(
            config, _skip_unless_enabled(config, items, ("live_quality", ), "run_live_quality", "--run-live-quality"))
