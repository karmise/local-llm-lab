"""Reviewed conversational acceptance rules, independent of generated answers."""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from llm_testkit.datasets.validation import load_catalog, require_text, validate_patterns


@dataclass(frozen=True)
class ConversationCase:
    id: str
    title: str
    question: str
    reference: str
    rationale: str
    required_patterns: tuple[tuple[str, str], ...]
    forbidden_patterns: tuple[tuple[str, str], ...]
    source_fragments: tuple[str, ...]
    max_words: int


@dataclass(frozen=True)
class ConversationCatalog:
    version: str
    sha256: str
    policy_sha256: str
    cases: tuple[ConversationCase, ...]


def load_conversation_catalog(path: Path, policy_file: Path) -> ConversationCatalog:
    raw = path.read_bytes()
    data = load_catalog(raw, "conversation catalog")
    version = require_text(data.get("version"), "Conversation catalog version")
    policy = policy_file.read_bytes()
    policy_sha256 = hashlib.sha256(policy).hexdigest()
    if data.get("policy_sha256") != policy_sha256:
        raise ValueError("Conversation policy checksum mismatch; review acceptance rules")
    policy_text = policy.decode("utf-8")
    rows = data.get("cases")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Conversation catalog must contain cases")
    cases = []
    identifiers = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Conversation case must be an object")
        identifier = require_text(row.get("id"), "Conversation case id")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", identifier) or identifier in identifiers:
            raise ValueError(f"Invalid or duplicate conversation case id: {identifier}")
        identifiers.add(identifier)
        texts = {
            name: require_text(row.get(name), f"{identifier}: {name}")
            for name in ("title", "question", "reference", "rationale")
        }
        required = validate_patterns(row.get("required_patterns"), identifier, required=True)
        forbidden = validate_patterns(row.get("forbidden_patterns"), identifier, required=False)
        max_words = row.get("max_words")
        if type(max_words) is not int or not 1 <= max_words <= 500:
            raise ValueError(f"{identifier}: max_words must be an integer from 1 to 500")
        fragments = row.get("source_fragments")
        if not isinstance(fragments, list):
            raise ValueError(f"{identifier}: source_fragments must be a list")
        for fragment in fragments:
            require_text(fragment, f"{identifier}: source fragment")
            if fragment not in policy_text:
                raise ValueError(f"{identifier}: source fragment is absent from policy")
        reference = " ".join(texts["reference"].split())
        if len(reference.split()) > max_words:
            raise ValueError(f"{identifier}: reference exceeds max_words")
        for label, pattern in required:
            if not re.search(pattern, reference, flags=re.IGNORECASE):
                raise ValueError(f"{identifier}: reference is missing required rule: {label}")
        for label, pattern in forbidden:
            if re.search(pattern, reference, flags=re.IGNORECASE):
                raise ValueError(f"{identifier}: reference violates forbidden rule: {label}")
        cases.append(
            ConversationCase(
                identifier,
                **texts,
                required_patterns=required,
                forbidden_patterns=forbidden,
                source_fragments=tuple(fragments),
                max_words=max_words,
            )
        )
    return ConversationCatalog(
        version, hashlib.sha256(raw).hexdigest(), policy_sha256, tuple(cases)
    )
