"""Versioned system prompts for controlled regression comparisons."""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PromptVariant:
    id: str
    version: str
    prompt: str
    sha256: str


@dataclass(frozen=True)
class PromptCatalog:
    version: str
    baseline: str
    variants: tuple[PromptVariant, ...]
    sha256: str


def load_prompt_catalog(path: Path) -> PromptCatalog:
    raw = path.read_bytes()
    data = json.loads(raw)
    if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise ValueError("Unsupported prompt catalog schema")
    if not isinstance(data.get("version"), str) or not data["version"].strip():
        raise ValueError("Prompt catalog requires a version")
    rows = data.get("variants")
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError("Prompt regression requires at least two variants")
    variants = []
    for row in rows:
        identifier, version, prompt = (row.get(k) for k in ("id", "version", "prompt"))
        if not isinstance(identifier, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", identifier):
            raise ValueError("Invalid prompt id")
        if any(v.id == identifier for v in variants):
            raise ValueError("Duplicate prompt id")
        if any(not isinstance(v, str) or not v.strip() for v in (version, prompt)):
            raise ValueError("Prompt text and version must be nonempty")
        if "[LLM_TESTKIT_CAPTURE:" in prompt:
            raise ValueError("Prompt catalog must not contain runtime capture markers")
        variants.append(
            PromptVariant(identifier, version, prompt, hashlib.sha256(prompt.encode()).hexdigest())
        )
    if data.get("baseline") not in {v.id for v in variants}:
        raise ValueError("Baseline prompt is absent")
    return PromptCatalog(
        data["version"], data["baseline"], tuple(variants), hashlib.sha256(raw).hexdigest()
    )
