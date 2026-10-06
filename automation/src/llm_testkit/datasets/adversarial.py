"""Source-bound adversarial inputs reuse reviewed golden expectations."""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from llm_testkit.datasets.golden import GoldenCase, GoldenDataset

CATEGORIES = frozenset(
    {
        "direct_injection",
        "role_spoofing",
        "source_spoofing",
        "unsupported_fact",
        "retrieved_injection",
        "conflicting_context",
    }
)
DOCUMENT_CATEGORIES = frozenset({"retrieved_injection", "conflicting_context"})


@dataclass(frozen=True)
class AdversarialCase:
    id: str
    category: str
    golden_case: GoldenCase
    question: str
    document_appendix: str
    forbidden_patterns: tuple[tuple[str, str], ...]
    catalog_sha256: str
    catalog_version: str


def load_adversarial_cases(path: Path, dataset: GoldenDataset) -> tuple[AdversarialCase, ...]:
    raw = path.read_bytes()
    data = json.loads(raw)
    if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise ValueError("Unsupported adversarial catalog schema")
    if data.get("golden_dataset_sha256") != dataset.sha256:
        raise ValueError("Adversarial expectations require the current golden dataset")
    if not isinstance(data.get("version"), str) or not data["version"].strip():
        raise ValueError("Adversarial catalog requires a version")
    if not isinstance(data.get("cases"), list) or not data["cases"]:
        raise ValueError("Adversarial catalog requires cases")
    golden = {c.id: c for c in dataset.cases}
    cases = []
    for row in data["cases"]:
        identifier = row.get("id")
        if (
            not isinstance(identifier, str)
            or not re.fullmatch(r"[a-z][a-z0-9_]*", identifier)
            or any(c.id == identifier for c in cases)
        ):
            raise ValueError("Invalid or duplicated attack id")
        if row.get("category") not in CATEGORIES or row.get("golden_case_id") not in golden:
            raise ValueError("Unknown attack category or golden case")
        case = golden[row["golden_case_id"]]
        question, appendix = row.get("question"), row.get("document_appendix")
        if not isinstance(question, str) or case.question not in question:
            raise ValueError("Attack must retain the original policy question")
        if not isinstance(appendix, str) or bool(appendix.strip()) != (
            row["category"] in DOCUMENT_CATEGORIES
        ):
            raise ValueError(
                "Document attacks require an appendix; user attacks must not modify the policy"
            )
        patterns = row.get("forbidden_patterns")
        if not isinstance(patterns, dict) or not patterns:
            raise ValueError("Attack requires explicit forbidden outputs")
        for label, pattern in patterns.items():
            if (
                not isinstance(label, str)
                or not label.strip()
                or not isinstance(pattern, str)
                or not pattern.strip()
            ):
                raise ValueError("Forbidden rules must have nonempty labels and patterns")
            if re.compile(pattern, re.IGNORECASE).search(""):
                raise ValueError("Forbidden regex must not match empty text")
        cases.append(
            AdversarialCase(
                identifier,
                row["category"],
                case,
                question,
                appendix,
                tuple(patterns.items()),
                hashlib.sha256(raw).hexdigest(),
                data["version"],
            )
        )
    return tuple(cases)


def materialize_policy(base_policy: Path, case: AdversarialCase, destination: Path) -> Path:
    if not case.document_appendix:
        return base_policy
    destination.write_text(
        base_policy.read_text(encoding="utf-8") + case.document_appendix, encoding="utf-8"
    )
    return destination
