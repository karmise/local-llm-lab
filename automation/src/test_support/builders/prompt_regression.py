"""Test data builders for versioned prompt variants and their JUnit evidence."""

import json
from typing import Any

from llm_testkit.datasets.prompts import PromptCatalog, PromptVariant, load_prompt_catalog
from test_support.builders.golden import GOLDEN_DATASET, TEST_DATA
from test_support.builders.identities import MODEL_DIGEST

PROMPTS_FILE = TEST_DATA / "prompt-variants.json"
CLASSNAME = "tests.test_prompt_regression"


def catalog_json() -> dict[str, Any]:
    """A fresh, editable copy of the reviewed prompt catalog."""
    return json.loads(PROMPTS_FILE.read_text())


def prompt_catalog() -> PromptCatalog:
    return load_prompt_catalog(PROMPTS_FILE)


def prompt_run(variant: PromptVariant, *, outcome: str | None = None, name: str | None = None, **changes) -> dict:
    """One JUnit test case of a paid-leave run with this prompt on 'model'; values can be changed or set to None."""
    catalog = prompt_catalog()
    properties = {
            "golden_case_id": "paid_leave",
            "generation_model": "model",
            "rag_iteration": "1",
            "golden_dataset_sha256": GOLDEN_DATASET.sha256,
            "policy_sha256": GOLDEN_DATASET.policy_sha256,
            "prompt_id": variant.id,
            "prompt_version": variant.version,
            "prompt_sha256": variant.sha256,
            "prompt_catalog_sha256": catalog.sha256,
            "model_digest": MODEL_DIGEST,
            "thinking_mode": "default",
            "workspace_configuration": json.dumps({
            "chatModel": "model",
            "openAiPrompt": variant.prompt,
            "topN": 4})}
    properties.update(changes)
    return {
            "name": name or f"test_prompt[{variant.id}]",
            "properties": {
            key: value
            for key, value in properties.items() if value is not None},
            "outcome": outcome}
