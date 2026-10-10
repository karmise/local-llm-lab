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


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "conversation_case" in metafunc.fixturenames:
        root = metafunc.config.rootpath / "test_data"
        try:
            catalog = load_conversation_catalog(root / "conversation-policy.json", root / "company-policy.txt")
        except (ValueError, OSError) as error:
            raise pytest.UsageError(f"Invalid conversation catalog: {error}") from error
        metafunc.parametrize("conversation_case", catalog.cases, ids=[c.id for c in catalog.cases])
    if "bias_case" in metafunc.fixturenames:
        root = metafunc.config.rootpath / "test_data"
        try:
            dataset = load_golden_dataset(root / "golden-policy.json", root / "company-policy.txt")
            cases = load_bias_cases(root / "bias-policy.json", dataset)
        except (ValueError, OSError) as error:
            raise pytest.UsageError(f"Invalid bias catalog: {error}") from error
        metafunc.parametrize("bias_case", cases, ids=[f"{c.pair_id}-{c.variant_id}" for c in cases])
    if "adversarial_case" in metafunc.fixturenames:
        root = metafunc.config.rootpath / "test_data"
        try:
            dataset = load_golden_dataset(root / "golden-policy.json", root / "company-policy.txt")
            cases = load_adversarial_cases(root / "adversarial-policy.json", dataset)
        except (ValueError, OSError) as error:
            raise pytest.UsageError(f"Invalid adversarial catalog: {error}") from error
        metafunc.parametrize("adversarial_case", cases, ids=[c.id for c in cases])
    if metafunc.definition.get_closest_marker("prompt_regression"):
        try:
            catalog = load_prompt_catalog(metafunc.config.rootpath / "test_data/prompt-variants.json")
            selected = metafunc.config.getoption("rag_prompts") or [v.id for v in catalog.variants]
            variants = {v.id: v for v in catalog.variants}
            if any(identifier not in variants for identifier in selected):
                raise ValueError("Unknown prompt variant")
        except (ValueError, OSError) as error:
            raise pytest.UsageError(f"Invalid prompt selection: {error}") from error
        metafunc.parametrize(
                "prompt_variant", [variants[i] for i in dict.fromkeys(selected)], ids=list(dict.fromkeys(selected)))
    if "golden_case" in metafunc.fixturenames:
        root = metafunc.config.rootpath / "test_data"
        try:
            dataset = load_golden_dataset(root / "golden-policy.json", root / "company-policy.txt")
        except (ValueError, OSError) as error:
            raise pytest.UsageError(f"Invalid golden dataset: {error}") from error
        metafunc.parametrize("golden_case", dataset.cases, ids=[case.id for case in dataset.cases])
    if (metafunc.definition.get_closest_marker("rag") and "generation_model" in metafunc.fixturenames):
        defaults = (("qwen3.5:4b", ) if metafunc.definition.get_closest_marker("live_quality")
                or metafunc.definition.get_closest_marker("performance")
                or metafunc.definition.get_closest_marker("conversation") else DEFAULT_RAG_MODELS)
        models = list(dict.fromkeys(metafunc.config.getoption("rag_models") or defaults))
        count = metafunc.config.getoption("rag_repeat")
        cases = [(model, iteration) for model in models for iteration in range(1, count + 1)]
        ids = [model if count == 1 else f"{model}-run-{iteration}" for model, iteration in cases]
        if "rag_iteration" in metafunc.fixturenames:
            metafunc.parametrize(("generation_model", "rag_iteration"), cases, ids=ids)
        else:
            metafunc.parametrize("generation_model", [model for model, _ in cases], ids=ids)


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    # Validate only selected tests, after pytest applies -k and -m filters.
    for item in items:
        if item.get_closest_marker("conversation"):
            if not config.getoption("run_conversation"):
                item.add_marker(pytest.mark.skip(reason="Enable explicitly with --run-conversation"))
            elif config.getoption("capture_rag"):
                raise pytest.UsageError(
                        "Conversation checks do not support --capture-rag: small talk is not a RAG sample")
    if not config.getoption("run_bias"):
        for item in items:
            if item.get_closest_marker("bias"):
                item.add_marker(pytest.mark.skip(reason="Enable explicitly with --run-bias"))
    for item in items:
        if item.get_closest_marker("performance"):
            mode = "rag" if item.get_closest_marker("rag") else "health"
            if not config.getoption("run_performance") or mode != config.getoption("performance_mode"):
                item.add_marker(
                        pytest.mark.skip(reason="Select explicitly with --run-performance and --performance-mode"))
            else:
                count, users = (config.getoption("performance_requests"), config.getoption("performance_users"))
                try:
                    validate_budget(count, users)
                except ValueError as error:
                    raise pytest.UsageError(f"Performance budget: {error}") from error
                if mode == "rag" and config.getoption("capture_rag"):
                    raise pytest.UsageError("Performance batches do not support the single-answer capture mode")

                p95 = config.getoption("performance_p95")
                if not math.isfinite(p95) or p95 <= 0:
                    raise pytest.UsageError("Performance p95 must be finite and positive")
    if not config.getoption("run_adversarial"):
        for item in items:
            if item.get_closest_marker("adversarial"):
                item.add_marker(pytest.mark.skip(reason="Enable explicitly with --run-adversarial"))
    else:
        for item in items:
            if item.get_closest_marker("adversarial") and hasattr(item, "callspec"):
                case = item.callspec.params.get("adversarial_case")
                if (case is not None and case.document_appendix and not config.getoption("capture_rag")):
                    raise pytest.UsageError(
                            "Document attacks require --capture-rag to verify actual retrieved attack exposure")
    if not config.getoption("run_prompt_regression"):
        for item in items:
            if item.get_closest_marker("prompt_regression"):
                item.add_marker(pytest.mark.skip(reason="Enable explicitly with --run-prompt-regression"))
    if not config.getoption("run_golden"):
        for item in items:
            if item.get_closest_marker("golden"):
                item.add_marker(pytest.mark.skip(reason="Enable explicitly with --run-golden"))
    browser_items = [item for item in items if item.get_closest_marker("ui") or item.get_closest_marker("browser")]
    if not config.getoption("run_ui"):
        for item in browser_items:
            item.add_marker(pytest.mark.skip(reason="Enable explicitly with --run-ui"))
    elif browser_items and not config.pluginmanager.hasplugin("playwright"):
        raise pytest.UsageError("UI checks require the UI extra and active pytest-playwright plugin")
    live_items = [item for item in items if item.get_closest_marker("live_quality")]
    if not config.getoption("run_live_quality"):
        for item in live_items:
            item.add_marker(pytest.mark.skip(reason="Enable explicitly with --run-live-quality"))
        return
    if live_items:
        if not config.getoption("capture_rag"):
            raise pytest.UsageError("Live quality requires --capture-rag and the capture Compose overlay")
        if len(live_items) != 1:
            raise pytest.UsageError("Live quality is limited to one scenario/model/repetition per run")
        if not config.getoption("allure_report_dir", default=None):
            raise pytest.UsageError("Live quality requires the reporting extra and --alluredir")
        if importlib.util.find_spec("ragas") is None:
            raise pytest.UsageError("Live quality requires the evaluation extra")
