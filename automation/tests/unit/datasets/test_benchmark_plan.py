"""Benchmark planning: a bounded, source-bound matrix, its manifest and comparable workspace settings."""

import hashlib

import pytest

from llm_testkit.datasets.benchmark import CONTROL_IDS, DEFAULT_CASES, configuration_hash, make_plan, manifest
from llm_testkit.reporting.steps import title
from test_support.builders.benchmark import QUALITY_GATES
from test_support.builders.calibration import CONTROLS_FILE
from test_support.builders.golden import GOLDEN_DATASET

pytestmark = pytest.mark.unit

# Budget: 6 fixed calls plus, per model, 12 for each answered case and 1 for each missing-information case.


@title("The default plan covers four curated cases on one model within 43 serial calls")
def test_default_plan():
    plan = make_plan(GOLDEN_DATASET)

    assert plan.case_ids == DEFAULT_CASES
    assert (plan.models, plan.judge_model) == (("qwen3.5:4b", ), "qwen3.5:4b")
    assert plan.maximum_calls == 6 + 3 * 12 + 1


@title("A plan keeps the selected case order, models and judge model")
def test_plan_keeps_selection():
    plan = make_plan(
            GOLDEN_DATASET, case_ids=["gym_missing", "paid_leave"], models=["qwen2.5:7b", "qwen3.5:4b"],
            judge_model="judge:1b")

    assert plan.case_ids == ("gym_missing", "paid_leave")
    assert (plan.models, plan.judge_model) == (("qwen2.5:7b", "qwen3.5:4b"), "judge:1b")
    assert plan.maximum_calls == 6 + 2 * (12 + 1)


@title("A plan whose call count equals the budget is accepted; one call less is rejected")
def test_plan_budget_boundary():
    two_models = dict(models=["qwen3.5:4b", "qwen2.5:7b"])

    assert make_plan(GOLDEN_DATASET, **two_models, max_model_calls=80).maximum_calls == 80
    with pytest.raises(ValueError, match="needs at most 80 model calls; budget is 79"):
        make_plan(GOLDEN_DATASET, **two_models, max_model_calls=79)


@title("Without an explicit budget a plan may use at most 50 calls")
def test_plan_default_budget():
    five_cases = ["paid_leave", "leave_approver", "carryover_limit", "travel_allowance"]

    with pytest.raises(ValueError, match="needs at most 54 model calls; budget is 50"):
        make_plan(GOLDEN_DATASET, case_ids=five_cases)


@title("Model names may use upper-case letters, digits and the separators of Ollama tags")
def test_plan_accepts_ollama_model_names():
    plan = make_plan(
            GOLDEN_DATASET, case_ids=["paid_leave"], models=["Qwen3.5:4B", "org/model_v2.1-q4"],
            judge_model="Judge:Latest")

    assert plan.models == ("Qwen3.5:4B", "org/model_v2.1-q4")


@title("The largest valid budget of 400 calls is accepted")
def test_plan_accepts_largest_budget():
    assert make_plan(GOLDEN_DATASET, max_model_calls=400).maximum_calls == 43


@title("The smallest valid budget of one call is compared with the plan rather than rejected as invalid")
def test_plan_compares_smallest_budget_with_calls():
    with pytest.raises(ValueError, match="needs at most 43 model calls; budget is 1"):
        make_plan(GOLDEN_DATASET, max_model_calls=1)


@pytest.mark.parametrize(("options", "message"), [
        pytest.param({"case_ids": []}, "unique, known golden cases", id="no-cases"),
        pytest.param({"case_ids": ["paid_leave", "unknown"]}, "unique, known golden cases", id="unknown-case"),
        pytest.param({"case_ids": ["paid_leave", "paid_leave"]}, "unique, known golden cases", id="duplicate-case"),
        pytest.param({"case_ids": ["gym_missing"]}, "Include paid_leave", id="without-control-anchor"),
        pytest.param({"models": []}, "one or two unique generation models", id="no-models"),
        pytest.param({"models": ["x", "x"]}, "one or two unique generation models", id="duplicate-model"),
        pytest.param({"models": ["x", "y", "z"]}, "one or two unique generation models", id="three-models"),
        pytest.param({"models": ["../bad[model]"]}, "Invalid model name", id="unsafe-model-name"),
        pytest.param({"models": [7]}, "Invalid model name", id="model-not-a-string"),
        pytest.param({"judge_model": "bad judge"}, "Invalid model name", id="unsafe-judge-name"),
        pytest.param({"max_model_calls": 0}, "integer between 1 and 400", id="zero-budget"),
        pytest.param({"max_model_calls": 401}, "integer between 1 and 400", id="budget-above-limit"),
        pytest.param({"max_model_calls": True}, "integer between 1 and 400", id="boolean-budget"),
        pytest.param({"max_model_calls": 50.0}, "integer between 1 and 400", id="float-budget")])
@title("Invalid or unsafe matrices are rejected before any model call [{param_id}]")
def test_plan_rejects_invalid_matrix(options, message):
    with pytest.raises(ValueError, match=message):
        make_plan(GOLDEN_DATASET, **options)


@title("The manifest records the reviewed inputs, bounded execution and the planned model-major matrix")
def test_manifest_records_plan_and_inputs():
    plan = make_plan(GOLDEN_DATASET, case_ids=["paid_leave", "gym_missing"], models=["qwen3.5:4b", "qwen2.5:7b"])

    definition = manifest(plan, GOLDEN_DATASET, QUALITY_GATES, CONTROLS_FILE)

    assert definition["schema_version"] == 1
    assert "not a population accuracy estimate" in definition["scope"]
    assert definition["golden_dataset_sha256"] == GOLDEN_DATASET.sha256
    assert definition["golden_dataset_version"] == GOLDEN_DATASET.version
    assert definition["policy_sha256"] == GOLDEN_DATASET.policy_sha256
    assert definition["quality_gates"] == QUALITY_GATES
    assert definition["controls_sha256"] == hashlib.sha256(CONTROLS_FILE.read_bytes()).hexdigest()
    assert definition["control_ids"] == list(CONTROL_IDS)
    assert (definition["judge_model"], definition["maximum_model_calls"]) == ("qwen3.5:4b", plan.maximum_calls)
    assert definition["execution"] == {"concurrency": 1, "retries": 0, "repetitions": 1}
    assert definition["expected_rows"] == [{
            "case_id": "paid_leave",
            "category": "multi_fact",
            "model": "qwen3.5:4b"}, {
            "case_id": "gym_missing",
            "category": "missing_information",
            "model": "qwen3.5:4b"}, {
            "case_id": "paid_leave",
            "category": "multi_fact",
            "model": "qwen2.5:7b"}, {
            "case_id": "gym_missing",
            "category": "missing_information",
            "model": "qwen2.5:7b"}]
    assert definition["human_review"] == "pending"


SETTINGS = {"openAiPrompt": "Answer from policy.", "chatModel": "qwen3.5:4b", "topN": 4}


@pytest.mark.parametrize(
        "changed", [
        pytest.param({
        **SETTINGS, "chatModel": "qwen2.5:7b"}, id="generation-model"),
        pytest.param({
        **SETTINGS, "openAiPrompt": "Answer from policy.\n[LLM_TESTKIT_CAPTURE:" + "f" * 32 + "]"},
        id="trailing-capture-marker"),
        pytest.param(dict(reversed(SETTINGS.items())), id="key-order"),
        pytest.param({
        k: v
        for k, v in SETTINGS.items() if k != "chatModel"}, id="without-generation-model")])
@title("Settings compare equal across generation models, per-request capture markers and key order [{param_id}]")
def test_configuration_hash_ignores_model_and_marker(changed):
    assert configuration_hash(changed) == configuration_hash(SETTINGS)


@pytest.mark.parametrize(
        "changed", [
        pytest.param({
        **SETTINGS, "openAiPrompt": "Answer freely."}, id="prompt"),
        pytest.param({
        **SETTINGS, "topN": 8}, id="retrieval-setting"),
        pytest.param({
        **SETTINGS, "openAiPrompt": "[LLM_TESTKIT_CAPTURE:" + "f" * 32 + "]\nAnswer from policy."},
        id="marker-not-at-end"),
        pytest.param({
        **SETTINGS, "openAiPrompt": "Answer from policy.\n[LLM_TESTKIT_CAPTURE:short]"}, id="malformed-marker")])
@title("Any other change of prompt or retrieval settings changes the configuration hash [{param_id}]")
def test_configuration_hash_detects_changed_settings(changed):
    assert configuration_hash(changed) != configuration_hash(SETTINGS)


@title("The configuration hash is the SHA-256 of the sorted settings and leaves its input unchanged")
def test_configuration_hash_value():
    settings = dict(SETTINGS)

    digest = configuration_hash(settings)

    assert digest == hashlib.sha256(b'{"openAiPrompt": "Answer from policy.", "topN": 4}').hexdigest()
    assert settings == SETTINGS
