"""Source-bound adversarial inputs reuse reviewed golden expectations."""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from llm_testkit.datasets.golden import GoldenCase, GoldenDataset
from llm_testkit.datasets.validation import (
    load_catalog,
    require_object,
    require_text,
    validate_patterns,
)

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
    data = load_catalog(raw, "adversarial catalog")
    if data.get("golden_dataset_sha256") != dataset.sha256:
        raise ValueError("Adversarial expectations require the current golden dataset")
    require_text(data.get("version"), "adversarial catalog version")
    if not isinstance(data.get("cases"), list) or not data["cases"]:
        raise ValueError("Adversarial catalog requires cases")
    golden = {c.id: c for c in dataset.cases}
    cases = []
    for row in data["cases"]:
        require_object(row, "adversarial catalog row")
        identifier = row.get("id")
        if (
            not isinstance(identifier, str)
            or not re.fullmatch(r"[a-z][a-z0-9_]*", identifier)
            or any(c.id == identifier for c in cases)
        ):
            raise ValueError("Invalid or duplicated attack id")
        if (
            not isinstance(row.get("category"), str)
            or row["category"] not in CATEGORIES
            or not isinstance(row.get("golden_case_id"), str)
            or row["golden_case_id"] not in golden
        ):
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
        patterns = validate_patterns(
            row.get("forbidden_patterns"), f"{identifier}: forbidden_patterns", required=True
        )
        cases.append(
            AdversarialCase(
                identifier,
                row["category"],
                case,
                question,
                appendix,
                patterns,
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
