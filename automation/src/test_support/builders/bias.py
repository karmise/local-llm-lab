"""Test data builders for counterfactual bias pairs and their JUnit evidence."""

import json
from pathlib import Path
from typing import Any

from llm_testkit.datasets.bias import BiasCase, load_bias_cases
from test_support.builders.golden import GOLDEN_DATASET, TEST_DATA
from test_support.builders.identities import MODEL_DIGEST
from test_support.builders.junit_xml import write_junit as write_bias_junit

BIAS_FILE = TEST_DATA / "bias-policy.json"
SETTINGS = {"chatModel": "model", "openAiPrompt": "Policy", "topN": 4}


def catalog_json() -> dict[str, Any]:
    """A fresh, editable copy of the reviewed counterfactual catalog."""
    return json.loads(BIAS_FILE.read_text())


def bias_cases() -> dict[tuple[str, str], BiasCase]:
    """Reviewed variants by (pair id, variant id), loaded on each call."""
    return {(case.pair_id, case.variant_id): case for case in load_bias_cases(BIAS_FILE, GOLDEN_DATASET)}


def bias_run(case: BiasCase, *, outcome: str | None = None, name: str | None = None, **changes: str) -> dict:
    """One JUnit test case of a bias variant run on 'model'; property values can be changed or set to None."""
    properties = {
            "bias_pair_id": case.pair_id,
            "bias_variant_id": case.variant_id,
            "generation_model": "model",
            "rag_iteration": "1",
            "bias_catalog_sha256": case.catalog_sha256,
            "golden_dataset_sha256": GOLDEN_DATASET.sha256,
            "policy_sha256": GOLDEN_DATASET.policy_sha256,
            "golden_case_id": case.golden_case.id,
            "model_digest": MODEL_DIGEST,
            "thinking_mode": "default",
            "workspace_configuration": json.dumps(SETTINGS)}
    properties.update(changes)
    return {
            "name": name or f"test_bias[{case.pair_id}-{case.variant_id}]",
            "properties": {
            key: value
            for key, value in properties.items() if value is not None},
            "outcome": outcome}


def write_junit(path: Path, runs: list[dict], *, classname: str = "tests.test_bias") -> Path:
    return write_bias_junit(path, runs, classname=classname)
