"""Select a source-bound matrix and reject expensive runs before generation."""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from llm_testkit.datasets.golden import GoldenDataset

CONTROL_IDS = ("supported_paraphrase", "wrong_numbers", "unsupported_gym_amount")
DEFAULT_CASES = ("paid_leave", "leave_approver", "hotel_receipt_condition", "gym_missing")


@dataclass(frozen=True)
class BenchmarkPlan:
    case_ids: tuple[str, ...]
    models: tuple[str, ...]
    judge_model: str
    maximum_calls: int


def make_plan(
    dataset: GoldenDataset,
    *,
    case_ids: list[str] | None = None,
    models: list[str] | None = None,
    judge_model: str = "qwen3.5:4b",
    max_model_calls: int = 50,
) -> BenchmarkPlan:
    identifiers = tuple(case_ids if case_ids is not None else DEFAULT_CASES)
    selected_models = tuple(models if models is not None else ("qwen3.5:4b",))
    cases = {case.id: case for case in dataset.cases}
    if (
        not identifiers
        or len(set(identifiers)) != len(identifiers)
        or any(i not in cases for i in identifiers)
    ):
        raise ValueError("Select unique, known golden cases")
    if "paid_leave" not in identifiers:
        raise ValueError("Include paid_leave to anchor the hand-labelled judge controls")
    for model in (*selected_models, judge_model):
        if not isinstance(model, str) or not re.fullmatch(r"[A-Za-z0-9_.:/-]+", model):
            raise ValueError("Invalid model name")
    if not 1 <= len(selected_models) <= 2 or len(set(selected_models)) != len(selected_models):
        raise ValueError("Select one or two unique generation models")
    # One generation + 2 faithfulness + 4 correctness + up to 5 relevance calls.
    # Missing-information cases use only reviewed answer/source checks.
    calls = 6 + len(selected_models) * sum(
        1 if cases[i].category == "missing_information" else 12 for i in identifiers
    )
    if type(max_model_calls) is not int or not 1 <= max_model_calls <= 400:
        raise ValueError("Model-call budget must be an integer between 1 and 400")
    if calls > max_model_calls:
        raise ValueError(f"Plan needs at most {calls} model calls; budget is {max_model_calls}")
    return BenchmarkPlan(identifiers, selected_models, judge_model, calls)


def manifest(plan: BenchmarkPlan, dataset: GoldenDataset, gates: dict, controls: Path) -> dict:
    cases = {case.id: case for case in dataset.cases}
    return {
        "schema_version": 1,
        "scope": "Curated policy benchmark; not a population accuracy estimate",
        "golden_dataset_sha256": dataset.sha256,
        "golden_dataset_version": dataset.version,
        "policy_sha256": dataset.policy_sha256,
        "quality_gates": gates,
        "controls_sha256": hashlib.sha256(controls.read_bytes()).hexdigest(),
        "control_ids": list(CONTROL_IDS),
        "judge_model": plan.judge_model,
        "maximum_model_calls": plan.maximum_calls,
        "execution": {"concurrency": 1, "retries": 0, "repetitions": 1},
        "expected_rows": [
            {"case_id": i, "category": cases[i].category, "model": model}
            for model in plan.models
            for i in plan.case_ids
        ],
        "human_review": "pending",
    }


def configuration_hash(configuration: dict) -> str:
    """Ignore only the per-request observation marker when comparing settings."""
    configuration = dict(configuration)
    configuration["openAiPrompt"] = re.sub(
        r"\n\[LLM_TESTKIT_CAPTURE:[0-9a-f]{32}\]$", "", configuration["openAiPrompt"]
    )
    configuration.pop("chatModel", None)
    return hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
