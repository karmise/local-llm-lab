"""Load reviewed expectations without deriving them from generated answers."""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from llm_testkit.datasets.validation import load_catalog, require_text, validate_patterns

CATEGORIES = frozenset({"fact_lookup", "multi_fact", "boundary", "missing_information"})


@dataclass(frozen=True)
class GoldenCase:
    id: str
    category: str
    question: str
    reference: str
    rationale: str
    required_patterns: tuple[tuple[str, str], ...]
    forbidden_patterns: tuple[tuple[str, str], ...]
    source_fragments: tuple[str, ...]


@dataclass(frozen=True)
class GoldenDataset:
    version: str
    sha256: str
    policy_sha256: str
    cases: tuple[GoldenCase, ...]


def load_golden_dataset(path: Path, policy_file: Path) -> GoldenDataset:
    raw = path.read_bytes()
    data = load_catalog(raw, "golden dataset")
    version = require_text(data.get("version"), "Dataset version")
    policy = policy_file.read_bytes()
    policy_text = policy.decode("utf-8")
    policy_sha256 = hashlib.sha256(policy).hexdigest()
    if data.get("policy_sha256") != policy_sha256:
        raise ValueError(
            "Golden dataset policy checksum mismatch; review expectations before updating"
        )
    rows = data.get("cases")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Golden dataset must contain cases")
    cases = []
    identifiers = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Golden case must be an object")
        identifier = require_text(row.get("id"), "Case id")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", identifier) or identifier in identifiers:
            raise ValueError(f"Invalid or duplicate golden case id: {identifier}")
        identifiers.add(identifier)
        category = row.get("category")
        if not isinstance(category, str) or category not in CATEGORIES:
            raise ValueError(f"{identifier}: unknown category")
        question = require_text(row.get("question"), f"{identifier}: question")
        reference = require_text(row.get("reference"), f"{identifier}: reference")
        rationale = require_text(row.get("rationale"), f"{identifier}: rationale")
        required = validate_patterns(row.get("required_patterns"), identifier, required=True)
        forbidden = validate_patterns(row.get("forbidden_patterns"), identifier, required=False)
        fragments = row.get("source_fragments")
        if not isinstance(fragments, list) or not fragments:
            raise ValueError(f"{identifier}: source_fragments must be a nonempty list")
        for fragment in fragments:
            require_text(fragment, f"{identifier}: source fragment")
            if fragment not in policy_text:
                raise ValueError(f"{identifier}: source fragment is absent from policy: {fragment}")
        normalized = " ".join(reference.split())
        for label, pattern in required:
            if not re.search(pattern, normalized, flags=re.IGNORECASE):
                raise ValueError(f"{identifier}: reference does not satisfy required rule: {label}")
        for label, pattern in forbidden:
            if re.search(pattern, normalized, flags=re.IGNORECASE):
                raise ValueError(f"{identifier}: reference violates forbidden rule: {label}")
        cases.append(
            GoldenCase(
                identifier,
                category,
                question,
                reference,
                rationale,
                required,
                forbidden,
                tuple(fragments),
            )
        )
    return GoldenDataset(version, hashlib.sha256(raw).hexdigest(), policy_sha256, tuple(cases))
