"""Judge suitability controls: a source-bound, engineering-labelled catalog and budgeted control selection."""

import hashlib
import json

import pytest

from llm_testkit.datasets.judge_controls import load_judge_controls, maximum_calls, select_judge_controls
from llm_testkit.reporting.steps import title
from test_support.builders.golden import GOLDEN_DATASET, POLICY_FILE
from test_support.builders.judge_validation import JUDGE_CONTROLS_FILE, catalog_json, control, curated_controls

pytestmark = pytest.mark.unit

CURATED_IDS = [
        "receipt_condition_supported", "receipt_condition_contradicted", "manager_policy_paraphrase",
        "manager_wrong_role", "relevant_context_first", "relevant_context_second", "all_reference_facts_retrieved",
        "notice_fact_not_retrieved"]
POLICY_LINES = [
        "Each employee receives 23 working days of paid leave per year.",
        "A leave request must be submitted at least 12 calendar days before leave starts.",
        "The employee's direct manager approves the request.",
        "Up to 6 unused working days may be carried over to the following year."]


def load(catalog: dict, tmp_path, *, policy_file=POLICY_FILE):
    path = tmp_path / "controls.json"
    path.write_text(json.dumps(catalog))
    return load_judge_controls(path, GOLDEN_DATASET, policy_file)


def with_cases(*cases: dict) -> dict:
    return {**catalog_json(), "cases": list(cases)}


@title("The curated catalog loads all eight controls bound to the policy and golden dataset")
def test_curated_catalog_loads():
    catalog = curated_controls()

    assert [case["id"] for case in catalog.cases] == CURATED_IDS
    assert catalog.version == "policy-judge-suitability-v1"
    assert catalog.sha256 == hashlib.sha256(JUDGE_CONTROLS_FILE.read_bytes()).hexdigest()
    assert (catalog.policy_sha256, catalog.dataset_sha256) == (GOLDEN_DATASET.policy_sha256, GOLDEN_DATASET.sha256)


@pytest.mark.parametrize(("control_id", "calls"), [("receipt_condition_supported", 2), ("manager_policy_paraphrase", 4),
        ("relevant_context_first", 2), ("notice_fact_not_retrieved", 1)])
@title("A control's judge budget follows its metric; precision needs one call per context [{param_id}]")
def test_maximum_calls_per_metric(control_id, calls):
    assert maximum_calls(control(control_id)) == calls


@pytest.mark.parametrize(("corrupt", "message"), [
        pytest.param(lambda c: c.update(schema_version=2), "Unsupported judge suitability catalog schema", id="schema"),
        pytest.param(lambda c: c.update(policy_sha256="stale"), "bound policy and golden dataset", id="policy-hash"),
        pytest.param(
        lambda c: c.update(golden_dataset_sha256="stale"), "bound policy and golden dataset", id="dataset-hash"),
        pytest.param(lambda c: c.pop("version"), "Judge control version must be a nonempty string", id="no-version"),
        pytest.param(
        lambda c: c.update(label_origin="human_reviewed"), "disclose engineering labels", id="label-origin"),
        pytest.param(lambda c: c.update(human_review="approved"), "pending human review", id="human-review"),
        pytest.param(lambda c: c.update(cases=[]), "between one and twelve", id="no-controls"),
        pytest.param(lambda c: c.update(cases={}), "between one and twelve", id="controls-not-a-list"),
        pytest.param(
        lambda c: c.update(cases=[{
        **c["cases"][0], "id": f"control_{i}"} for i in range(13)]), "between one and twelve", id="thirteen-controls")])
@title("A catalog not bound to the reviewed sources or not disclosing pending review is rejected [{param_id}]")
def test_catalog_rejects_unbound_catalog(tmp_path, corrupt, message):
    catalog = catalog_json()
    corrupt(catalog)

    with pytest.raises(ValueError, match=message):
        load(catalog, tmp_path)


@title("Controls are rejected when the policy file differs from the one they were labelled against")
def test_catalog_rejects_other_policy_file(tmp_path):
    policy = tmp_path / "policy.txt"
    policy.write_text(POLICY_FILE.read_text() + "\nAmended clause.")

    with pytest.raises(ValueError, match="bound policy and golden dataset"):
        load(catalog_json(), tmp_path, policy_file=policy)


@title("Duplicate JSON fields are rejected instead of silently keeping the last value")
def test_catalog_rejects_duplicate_fields(tmp_path):
    path = tmp_path / "controls.json"
    path.write_text(JUDGE_CONTROLS_FILE.read_text().replace('"version":', '"version": "x", "version":', 1))

    with pytest.raises(ValueError, match="duplicate JSON field: version"):
        load_judge_controls(path, GOLDEN_DATASET, POLICY_FILE)


def change(control_id: str, **fields):
    """Edit one control in a catalog."""
    return lambda catalog: next(c for c in catalog["cases"] if c["id"] == control_id).update(fields)


def change_label(control_id: str, field: str, **rule):
    """Edit the first claim label of one control."""
    return lambda catalog: next(c for c in catalog["cases"] if c["id"] == control_id)["labels"][field][0].update(rule)


FAITHFUL = "receipt_condition_supported"
PRECISION = "relevant_context_first"
RECALL = "notice_fact_not_retrieved"
CORRECTNESS = "manager_policy_paraphrase"


@pytest.mark.parametrize(("corrupt", "message"), [
        pytest.param(
        lambda c: c["cases"].__setitem__(0, "control"), "Judge control must be an object", id="not-object"),
        pytest.param(change(FAITHFUL, id=""), "Judge control id must be a nonempty string", id="blank-id"),
        pytest.param(change(FAITHFUL, id="Bad-Id"), "valid and unique", id="invalid-id"),
        pytest.param(change(FAITHFUL, id="receipt_condition_contradicted"), "valid and unique", id="duplicate-id"),
        pytest.param(change(FAITHFUL, metric="answer_relevancy"), "known metric and golden case", id="unknown-metric"),
        pytest.param(change(FAITHFUL, case_id="unknown"), "known metric and golden case", id="unknown-case"),
        pytest.param(change(FAITHFUL, user_input="Edited"), "question/reference differs", id="edited-question"),
        pytest.param(change(FAITHFUL, reference="Edited"), "question/reference differs", id="edited-reference"),
        pytest.param(change(FAITHFUL, response=" "), "Synthetic control response must be", id="blank-response"),
        pytest.param(change(FAITHFUL, rationale=None), "Judge control rationale must be", id="no-rationale"),
        pytest.param(change(FAITHFUL, retrieved_contexts=[]), "one to four exact policy excerpts", id="no-contexts"),
        pytest.param(
        change(FAITHFUL, retrieved_contexts=POLICY_LINES + POLICY_LINES[:1]), "one to four exact policy excerpts",
        id="five-contexts"),
        pytest.param(change(FAITHFUL, retrieved_contexts=[1]), "one to four exact policy excerpts", id="context-type"),
        pytest.param(
        change(FAITHFUL, retrieved_contexts=[" "]), "one to four exact policy excerpts", id="blank-context"),
        pytest.param(
        change(FAITHFUL, retrieved_contexts=["Invented policy."]), "one to four exact policy excerpts",
        id="invented-context"),
        pytest.param(
        change(FAITHFUL, retrieved_contexts="Each employee"), "one to four exact policy excerpts",
        id="contexts-not-a-list"),
        pytest.param(change(FAITHFUL, expected_score=True), "between zero and one", id="boolean-score"),
        pytest.param(change(FAITHFUL, expected_score=1.5), "between zero and one", id="score-above-one"),
        pytest.param(change(FAITHFUL, expected_score=-0.5), "between zero and one", id="negative-score"),
        pytest.param(change(FAITHFUL, labels=[]), "Judge control labels must be an object", id="labels-not-object"),
        pytest.param(
        change(FAITHFUL, labels={
        "response": [{
        "pattern": "receipt",
        "verdict": 1}],
        "ignored": []}), "exactly the fields required", id="extra-label-field"),
        pytest.param(
        change(CORRECTNESS, labels={"response": [{
        "pattern": "manager",
        "verdict": 1}]}), "exactly the fields required", id="missing-label-field"),
        pytest.param(
        change(RECALL, labels={"response": [{
        "pattern": "23",
        "verdict": 1}]}), "exactly the fields required", id="recall-labels-response"),
        pytest.param(change(PRECISION, labels={"verdicts": [1]}), "ordered binary label", id="too-few-verdicts"),
        pytest.param(change(PRECISION, labels={"verdicts": [1, 2]}), "ordered binary label", id="verdict-two"),
        pytest.param(change(PRECISION, labels={"verdicts": [True, 0]}), "ordered binary label", id="boolean-verdict"),
        pytest.param(change(PRECISION, labels={"verdicts": "10"}), "ordered binary label", id="verdicts-not-a-list"),
        pytest.param(
        change(FAITHFUL, labels={"response": []}), "requires response claim labels", id="no-response-labels"),
        pytest.param(
        change(RECALL, labels={"reference": []}), "requires reference claim labels", id="no-reference-labels"),
        pytest.param(
        change(FAITHFUL, labels={"response": ["receipt"]}), "Judge claim label must be an object",
        id="label-not-object"),
        pytest.param(change_label(FAITHFUL, "response", pattern=""), "Judge claim pattern must be", id="blank-pattern"),
        pytest.param(
        change_label(FAITHFUL, "response", pattern="x*"), "must not match empty text", id="pattern-matches-empty-text"),
        pytest.param(change_label(FAITHFUL, "response", verdict=True), "binary integers", id="boolean-claim-verdict"),
        pytest.param(change_label(FAITHFUL, "response", verdict=2), "binary integers", id="claim-verdict-two"),
        pytest.param(change(FAITHFUL, expected_score=0), "disagrees with labelled verdicts", id="faithfulness-score"),
        pytest.param(
        change(CORRECTNESS, expected_score=0.5), "disagrees with labelled verdicts", id="correctness-score"),
        pytest.param(change(PRECISION, expected_score=0.5), "disagrees with labelled verdicts", id="precision-score"),
        pytest.param(change(RECALL, expected_score=1), "disagrees with labelled verdicts", id="recall-score")])
@title("Malformed controls or labels that disagree with their expected score are rejected [{param_id}]")
def test_catalog_rejects_invalid_control(tmp_path, corrupt, message):
    catalog = catalog_json()
    corrupt(catalog)

    with pytest.raises(ValueError, match=message):
        load(catalog, tmp_path)


@title("A catalog of one control and a catalog of twelve controls are both accepted")
def test_catalog_accepts_size_bounds(tmp_path):
    single = load(with_cases(control(FAITHFUL)), tmp_path)
    twelve = load(with_cases(*[{**control(FAITHFUL), "id": f"control_{i}"} for i in range(12)]), tmp_path)

    assert (len(single.cases), len(twelve.cases)) == (1, 12)


@pytest.mark.parametrize(("labels", "expected"), [
        pytest.param([1, 0, 0, 1], 0.75, id="useful-first-and-last"),
        pytest.param([0, 0, 0, 0], 0, id="nothing-useful")])
@title("Precision controls accept four exact excerpts; a control without useful context expects zero [{param_id}]")
def test_catalog_accepts_four_context_precision(tmp_path, labels, expected):
    case = {
            **control(PRECISION), "retrieved_contexts": POLICY_LINES,
            "labels": {
            "verdicts": labels},
            "expected_score": expected}

    assert load(with_cases(case), tmp_path).cases[0]["expected_score"] == expected


@title("Correctness controls expect F1 rounded to two decimals")
def test_catalog_rounds_correctness_f1(tmp_path):
    labels = {
            "response": [{
            "pattern": "manager",
            "verdict": 1}],
            "reference": [{
            "pattern": "manager",
            "verdict": 1}, {
            "pattern": "approv",
            "verdict": 0}]}
    case = {**control(CORRECTNESS), "labels": labels, "expected_score": 0.67}

    assert load(with_cases(case), tmp_path).cases[0]["expected_score"] == 0.67


@title("Without a selection every control is chosen when the 18-call budget covers them all")
def test_select_all_within_budget():
    assert [case["id"] for case in select_judge_controls(curated_controls(), None, 18)] == CURATED_IDS


@title("A selection keeps catalog order and may use exactly its call budget")
def test_select_subset_in_catalog_order():
    chosen = select_judge_controls(curated_controls(), [RECALL, FAITHFUL], 3)

    assert [case["id"] for case in chosen] == [FAITHFUL, RECALL]


@pytest.mark.parametrize(("identifiers", "budget"), [
        pytest.param(None, 17, id="one-call-short"),
        pytest.param([RECALL], 0, id="zero-budget"),
        pytest.param([RECALL], 33, id="budget-above-maximum"),
        pytest.param([RECALL], True, id="boolean-budget")])
@title("Selections over budget or with an invalid budget are rejected before any judge call [{param_id}]")
def test_select_rejects_budget(identifiers, budget):
    with pytest.raises(ValueError, match="exceed the explicit call budget"):
        select_judge_controls(curated_controls(), identifiers, budget)


@title("The largest budget of 32 calls is accepted")
def test_select_accepts_largest_budget():
    assert len(select_judge_controls(curated_controls(), None, 32)) == 8


@pytest.mark.parametrize("identifiers", [[], ["unknown"], [FAITHFUL, "unknown"]], ids=["empty", "unknown", "mixed"])
@title("Empty or unknown selections are rejected [{param_id}]")
def test_select_rejects_unknown_controls(identifiers):
    with pytest.raises(ValueError, match="nonempty, known judge control ids"):
        select_judge_controls(curated_controls(), identifiers, 32)
