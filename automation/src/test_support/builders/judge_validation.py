"""Prepared judge responses keep unit scenarios independent of model inference."""

import asyncio
from dataclasses import dataclass
from typing import Any

from llm_testkit.evaluation.judge_validation import evaluate_judge_controls
from test_support.data.judge_validation import RECALL_STATEMENTS


def verdict(statement: str, value: int) -> dict:
    return {"statement": statement, "verdict": value, "reason": "Independent engineering test label"}


def control_outputs(control: dict) -> list[dict]:
    labels = control["labels"]
    if control["metric"] == "faithfulness":
        return [{
                "statements": [control["response"]]}, {
                "statements": [verdict(control["response"], labels["response"][0]["verdict"])]}]
    if control["metric"] == "factual_correctness":
        return [{
                "claims": [control["response"]]}, {
                "statements": [verdict(control["response"], labels["response"][0]["verdict"])]}, {
                "claims": [control["reference"]]}, {
                "statements": [verdict(control["reference"], labels["reference"][0]["verdict"])]}]
    if control["metric"] == "context_precision":
        return [{"verdict": value, "reason": "Independent ranked-context label"} for value in labels["verdicts"]]
    return [{
            "classifications": [{
            "statement": statement,
            "attributed": rule["verdict"],
            "reason": "Source label"} for statement, rule in zip(RECALL_STATEMENTS, labels["reference"], strict=True)]}]


def mutate_catalog(data: dict, change: str) -> None:
    mutations = {
            "policy_hash": lambda: data.update(policy_sha256="stale"),
            "dataset_hash": lambda: data.update(golden_dataset_sha256="stale"),
            "duplicate_id": lambda: data["cases"].append(data["cases"][0]),
            "reference": lambda: data["cases"][0].update(reference="Invented reference"),
            "invented_context": lambda: data["cases"][0].update(retrieved_contexts=["Invented policy"]),
            "boolean_score": lambda: data["cases"][0].update(expected_score=True),
            "boolean_verdict": lambda: data["cases"][0]["labels"]["response"][0].update(verdict=True),
            "score_label_disagreement": lambda: data["cases"][0].update(expected_score=0),
            "empty_pattern": lambda: data["cases"][0]["labels"]["response"][0].update(pattern=".*"),
            "invalid_metric": lambda: data["cases"][0].update(metric="unknown"),
            "extra_label_field": lambda: data["cases"][0]["labels"].update(ignored=[])}
    mutations[change]()


@dataclass
class ValidationBatch:
    controls: list[dict]
    factory: Any
    scorer: Any

    def run(self) -> list[dict]:
        return asyncio.run(evaluate_judge_controls(self.controls, self.factory, self.scorer))


@dataclass
class SavedReviewScenario:
    path: Any
    evidence_path: Any

    def review(self) -> dict:
        from llm_testkit.reporting.judge_review import review_benchmark

        return review_benchmark(self.path)

    def corrupt_evidence(self) -> None:
        self.evidence_path.write_text("{}")
