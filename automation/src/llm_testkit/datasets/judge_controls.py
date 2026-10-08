"""Source-bound engineering labels for small judge suitability experiments."""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_testkit.datasets.golden import GoldenDataset
from llm_testkit.datasets.validation import load_catalog, require_object, require_text

METRICS = frozenset({"faithfulness", "factual_correctness", "context_precision", "context_recall"})


@dataclass(frozen=True)
class JudgeControls:
    version: str
    sha256: str
    policy_sha256: str
    dataset_sha256: str
    cases: tuple[dict[str, Any], ...]


def maximum_calls(control: dict[str, Any]) -> int:
    return {
            "faithfulness": 2,
            "factual_correctness": 4,
            "context_precision": len(control["retrieved_contexts"]),
            "context_recall": 1}[control["metric"]]


def load_judge_controls(path: Path, dataset: GoldenDataset, policy_file: Path) -> JudgeControls:
    raw = path.read_bytes()
    catalog = load_catalog(raw, "judge suitability catalog")
    policy = policy_file.read_bytes()
    if (catalog.get("policy_sha256") != hashlib.sha256(policy).hexdigest()
                or catalog.get("policy_sha256") != dataset.policy_sha256
                or catalog.get("golden_dataset_sha256") != dataset.sha256):
        raise ValueError("Judge controls require the bound policy and golden dataset")
    version = require_text(catalog.get("version"), "Judge control version")
    if catalog.get("label_origin") != "engineering_authored" or catalog.get("human_review") != "pending":
        raise ValueError("Judge controls must disclose engineering labels and pending human review")
    cases = catalog.get("cases")
    if not isinstance(cases, list) or not 1 <= len(cases) <= 12:
        raise ValueError("Judge catalog requires between one and twelve controls")
    golden = {case.id: case for case in dataset.cases}
    identifiers = set()
    for control in cases:
        require_object(control, "Judge control")
        identifier = require_text(control.get("id"), "Judge control id")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", identifier) or identifier in identifiers:
            raise ValueError("Judge control ids must be valid and unique")
        identifiers.add(identifier)
        metric = control.get("metric")
        if metric not in METRICS or control.get("case_id") not in golden:
            raise ValueError("Judge control requires a known metric and golden case")
        case = golden[control["case_id"]]
        if control.get("user_input") != case.question or control.get("reference") != case.reference:
            raise ValueError("Judge control question/reference differs from its golden case")
        require_text(control.get("response"), "Synthetic control response")
        require_text(control.get("rationale"), "Judge control rationale")
        contexts = control.get("retrieved_contexts")
        if (not isinstance(contexts, list) or not 1 <= len(contexts) <= 4
                    or any(not isinstance(c, str) or not c.strip() or c not in policy.decode("utf-8")
                for c in contexts)):
            raise ValueError("Judge contexts must be one to four exact policy excerpts")
        value = control.get("expected_score")
        if type(value) not in (int, float) or not 0 <= value <= 1:
            raise ValueError("Judge control expected score must be between zero and one")
        labels = require_object(control.get("labels"), "Judge control labels")
        expected_fields = {"response", "reference"} if metric == "factual_correctness" else {
                "verdicts"
        } if metric == "context_precision" else {"reference"} if metric == "context_recall" else {"response"}
        if set(labels) != expected_fields:
            raise ValueError("Judge control labels must contain exactly the fields required by their metric")
        if metric == "context_precision":
            verdicts = labels.get("verdicts")
            if (not isinstance(verdicts, list) or len(verdicts) != len(contexts)
                        or any(type(v) is not int or v not in (0, 1) for v in verdicts)):
                raise ValueError("Precision controls require an ordered binary label for every context")
            expected = sum(sum(verdicts[:i + 1]) / (i + 1) * v for i, v in enumerate(verdicts)) / (sum(verdicts) or 1)
        else:
            fields = ("response", "reference") if metric == "factual_correctness" else (
                    "reference", ) if metric == "context_recall" else ("response", )
            for field in fields:
                rules = labels.get(field)
                if not isinstance(rules, list) or not rules:
                    raise ValueError(f"Judge control requires {field} claim labels")
                for rule in rules:
                    require_object(rule, "Judge claim label")
                    pattern = require_text(rule.get("pattern"), "Judge claim pattern")
                    if re.compile(pattern, re.IGNORECASE).search(""):
                        raise ValueError("Judge claim patterns must not match empty text")
                    if type(rule.get("verdict")) is not int or rule["verdict"] not in (0, 1):
                        raise ValueError("Judge claim labels must be binary integers")
            if metric == "factual_correctness":
                tp = sum(rule["verdict"] for rule in labels["response"])
                fp = len(labels["response"]) - tp
                fn = sum(1 - rule["verdict"] for rule in labels["reference"])
                expected = round(2 * tp / (2 * tp + fp + fn), 2)
            else:
                rules = labels[fields[0]]
                expected = sum(rule["verdict"] for rule in rules) / len(rules)
        if abs(value - expected) > 1e-9:
            raise ValueError("Judge expected score disagrees with labelled verdicts")
    return JudgeControls(version, hashlib.sha256(raw).hexdigest(), dataset.policy_sha256, dataset.sha256, tuple(cases))


def select_judge_controls(catalog: JudgeControls, identifiers: list[str] | None, budget: int) -> list[dict[str, Any]]:
    selected = set(identifiers) if identifiers is not None else {case["id"] for case in catalog.cases}
    if not selected or selected - {case["id"] for case in catalog.cases}:
        raise ValueError("Select nonempty, known judge control ids")
    cases = [case for case in catalog.cases if case["id"] in selected]
    if type(budget) is not int or not 1 <= budget <= 32 or sum(maximum_calls(case) for case in cases) > budget:
        raise ValueError("Judge controls exceed the explicit call budget (maximum 32)")
    return cases
