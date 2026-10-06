"""CLI options and selection rules for costly integration scenarios."""

import importlib.util
from pathlib import Path

import pytest

from llm_testkit.datasets.golden import load_golden_dataset
from llm_testkit.datasets.prompts import load_prompt_catalog

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
    parser.addoption(
        "--run-prompt-regression",
        action="store_true",
        help="Enable versioned prompt comparisons against golden cases",
    )
    parser.addoption(
        "--rag-prompt",
        action="append",
        dest="rag_prompts",
        help="Select prompt variant IDs for prompt regression",
    )
    parser.addoption("--relevance-report", type=Path, help="Existing precision/recall evidence")
    parser.addoption(
        "--run-golden",
        action="store_true",
        help="Enable source-bound golden RAG scenarios (one generation per selected case).",
    )
    parser.addoption(
        "--run-ui",
        action="store_true",
        help="Enable browser checks (application UI scenarios generate one answer each).",
    )
    parser.addoption(
        "--run-live-quality",
        action="store_true",
        help="Enable one live RAG-to-Allure scenario (three model calls maximum).",
    )
    parser.addoption(
        "--judge-model",
        type=_model_name,
        default="qwen3.5:4b",
        help="Local judge model for the live quality scenario.",
    )
    parser.addoption(
        "--quality-sample", type=Path, help="Captured sample for an offline quality report."
    )
    parser.addoption(
        "--correctness-report",
        type=Path,
        help="Optional existing golden-reference correctness evidence.",
    )
    parser.addoption(
        "--faithfulness-report",
        type=Path,
        help="Existing judge report bound to the sample checksum.",
    )
    parser.addoption(
        "--capture-rag",
        action="store_true",
        default=False,
        help="Save actual model messages and evaluation samples; requires compose.capture.yaml.",
    )
    parser.addoption(
        "--rag-model",
        action="append",
        type=_model_name,
        dest="rag_models",
        default=None,
        help="Generation model for RAG tests; repeat to select multiple models.",
    )
    parser.addoption(
        "--rag-repeat",
        type=_positive_repeat,
        default=1,
        help="Independent repetitions per RAG scenario and model (default: 1).",
    )


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if metafunc.definition.get_closest_marker("prompt_regression"):
        try:
            catalog = load_prompt_catalog(
                metafunc.config.rootpath / "test_data/prompt-variants.json"
            )
            selected = metafunc.config.getoption("rag_prompts") or [v.id for v in catalog.variants]
            variants = {v.id: v for v in catalog.variants}
            if any(identifier not in variants for identifier in selected):
                raise ValueError("Unknown prompt variant")
        except (ValueError, OSError) as error:
            raise pytest.UsageError(f"Invalid prompt selection: {error}") from error
        metafunc.parametrize(
            "prompt_variant",
            [variants[i] for i in dict.fromkeys(selected)],
            ids=list(dict.fromkeys(selected)),
        )
    if "golden_case" in metafunc.fixturenames:
        root = metafunc.config.rootpath / "test_data"
        try:
            dataset = load_golden_dataset(root / "golden-policy.json", root / "company-policy.txt")
        except (ValueError, OSError) as error:
            raise pytest.UsageError(f"Invalid golden dataset: {error}") from error
        metafunc.parametrize("golden_case", dataset.cases, ids=[case.id for case in dataset.cases])
    if (
        metafunc.definition.get_closest_marker("rag")
        and "generation_model" in metafunc.fixturenames
    ):
        defaults = (
            ("qwen3.5:4b",)
            if metafunc.definition.get_closest_marker("live_quality")
            else DEFAULT_RAG_MODELS
        )
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
    if not config.getoption("run_prompt_regression"):
        for item in items:
            if item.get_closest_marker("prompt_regression"):
                item.add_marker(
                    pytest.mark.skip(reason="Enable explicitly with --run-prompt-regression")
                )
    if not config.getoption("run_golden"):
        for item in items:
            if item.get_closest_marker("golden"):
                item.add_marker(pytest.mark.skip(reason="Enable explicitly with --run-golden"))
    browser_items = [
        item
        for item in items
        if item.get_closest_marker("ui") or item.get_closest_marker("browser")
    ]
    if not config.getoption("run_ui"):
        for item in browser_items:
            item.add_marker(pytest.mark.skip(reason="Enable explicitly with --run-ui"))
    elif browser_items and not config.pluginmanager.hasplugin("playwright"):
        raise pytest.UsageError(
            "UI checks require the UI extra and active pytest-playwright plugin"
        )
    live_items = [item for item in items if item.get_closest_marker("live_quality")]
    if not config.getoption("run_live_quality"):
        for item in live_items:
            item.add_marker(pytest.mark.skip(reason="Enable explicitly with --run-live-quality"))
        return
    if live_items:
        if not config.getoption("capture_rag"):
            raise pytest.UsageError(
                "Live quality requires --capture-rag and the capture Compose overlay"
            )
        if len(live_items) != 1:
            raise pytest.UsageError(
                "Live quality is limited to one scenario/model/repetition per run"
            )
        if not config.getoption("allure_report_dir", default=None):
            raise pytest.UsageError("Live quality requires the reporting extra and --alluredir")
        if importlib.util.find_spec("ragas") is None:
            raise pytest.UsageError("Live quality requires the evaluation extra")
