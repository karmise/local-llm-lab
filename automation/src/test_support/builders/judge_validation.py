"""Test data builders for judge suitability controls: the reviewed catalog and scripted judge replies."""

import json
from typing import Any

from llm_testkit.datasets.judge_controls import JudgeControls, load_judge_controls
from test_support.builders.golden import GOLDEN_DATASET, POLICY_FILE, TEST_DATA

JUDGE_CONTROLS_FILE = TEST_DATA / "judge-validation.json"
RECALL_STATEMENTS = (
        "Each employee receives 23 working days of paid leave per year.",
        "A leave request must be submitted at least 12 calendar days before leave starts.")


def catalog_json() -> dict[str, Any]:
    """A fresh, editable copy of the reviewed judge control catalog."""
    return json.loads(JUDGE_CONTROLS_FILE.read_text())


def curated_controls() -> JudgeControls:
    return load_judge_controls(JUDGE_CONTROLS_FILE, GOLDEN_DATASET, POLICY_FILE)


def control(control_id: str) -> dict[str, Any]:
    """A fresh copy of one curated control."""
    return next(case for case in catalog_json()["cases"] if case["id"] == control_id)


def verdict(statement: str, value: int) -> dict[str, Any]:
    return {"statement": statement, "verdict": value, "reason": "Independent engineering test label"}


def control_outputs(case: dict[str, Any]) -> list[dict[str, Any]]:
    """The structured judge replies that reproduce a control's labels exactly, in RAGAS call order."""
    labels = case["labels"]
    if case["metric"] == "faithfulness":
        return [{
                "statements": [case["response"]]}, {
                "statements": [verdict(case["response"], labels["response"][0]["verdict"])]}]
    if case["metric"] == "factual_correctness":
        return [{
                "claims": [case["response"]]}, {
                "statements": [verdict(case["response"], labels["response"][0]["verdict"])]}, {
                "claims": [case["reference"]]}, {
                "statements": [verdict(case["reference"], labels["reference"][0]["verdict"])]}]
    if case["metric"] == "context_precision":
        return [{"verdict": value, "reason": "Independent ranked-context label"} for value in labels["verdicts"]]
    classifications = [{
            "statement": statement,
            "attributed": rule["verdict"],
            "reason": "Source label"} for statement, rule in zip(RECALL_STATEMENTS, labels["reference"], strict=True)]
    return [{"classifications": classifications}]
