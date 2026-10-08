"""Counterfactual employee descriptors with identical reviewed policy expectations."""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from llm_testkit.datasets.golden import GoldenCase, GoldenDataset
from llm_testkit.datasets.validation import load_catalog, require_object, require_text, validate_patterns


@dataclass(frozen=True)
class BiasCase:
    pair_id: str
    attribute: str
    variant_id: str
    descriptor: str
    question: str
    golden_case: GoldenCase
    forbidden_patterns: tuple[tuple[str, str], ...]
    catalog_sha256: str
    catalog_version: str


def load_bias_cases(path: Path, dataset: GoldenDataset) -> tuple[BiasCase, ...]:
    raw = path.read_bytes()
    data = load_catalog(raw, "bias catalog")
    if data.get("golden_dataset_sha256") != dataset.sha256:
        raise ValueError("Bias catalog must bind to the current golden dataset")
    require_text(data.get("version"), "bias catalog version")
    if not isinstance(data.get("pairs"), list) or not data["pairs"]:
        raise ValueError("Bias catalog requires pairs")
    golden = {c.id: c for c in dataset.cases}
    result = []
    seen = set()
    for pair in data["pairs"]:
        require_object(pair, "bias catalog row")
        identifier = pair.get("id")
        if (not isinstance(identifier, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", identifier) or identifier in seen):
            raise ValueError("Invalid or duplicate bias pair")
        seen.add(identifier)
        if (pair.get("attribute") not in ("gender", "age", "nationality")
                    or not isinstance(pair.get("golden_case_id"), str) or pair.get("golden_case_id") not in golden):
            raise ValueError("Unknown bias attribute or golden expectation")
        if not isinstance(pair.get("rationale"), str) or not pair["rationale"].strip():
            raise ValueError("Counterfactual pair requires reviewed rationale")
        case = golden[pair["golden_case_id"]]
        variants = pair.get("variants")
        if (not isinstance(variants, list) or len(variants) != 2 or any(not isinstance(v, dict) for v in variants)
                    or any(not isinstance(v.get("id"), str) for v in variants)
                    or {v["id"] for v in variants} != {"1", "2"}):
            raise ValueError("Bias pair requires exactly two distinct variants")
        if variants[0].get("descriptor") == variants[1].get("descriptor"):
            raise ValueError("Counterfactual descriptors must differ")
        patterns = validate_patterns(pair.get("forbidden_patterns"), f"{identifier}: forbidden_patterns", required=True)
        for variant in variants:
            descriptor, question = variant.get("descriptor"), variant.get("question")
            if (not isinstance(descriptor, str) or not descriptor.strip()
                        or question != f"I am an employee and I am {descriptor}. " + case.question):
                raise ValueError("Only the declared descriptor may change between questions")
            result.append(
                    BiasCase(
                    identifier, pair["attribute"], variant["id"], descriptor, question, case, patterns,
                    hashlib.sha256(raw).hexdigest(), data["version"]))
    return tuple(result)
