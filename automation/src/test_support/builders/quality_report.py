"""Test data builders for the saved-answer quality report: a captured sample, its judge evidence and a profile."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_testkit.observation.evaluation_sample import build_sample

CAPTURE_ID = "a" * 32
DOCUMENT = f"automation-{CAPTURE_ID}-company-policy.txt"
QUESTION = "Leave?"
FULL_ANSWER = "23 working days; 12 calendar days"
GENERATION_MODEL = "test-model"
JUDGE_DIGEST = "digest"


def policy_context(document: str = DOCUMENT) -> str:
    """A retrieved context whose metadata names ``document`` as its source."""
    return f"<document_metadata>\nsourceDocument: {document}\n</document_metadata>\n{FULL_ANSWER}"


def faithfulness_evidence(sample_path: Path, answer: str) -> dict[str, Any]:
    """Completed faithfulness evidence for ``sample_path`` that supports the single statement ``answer``."""
    return {
            "schema_version": 1,
            "metric": "faithfulness",
            "status": "completed",
            "sample_sha256": hashlib.sha256(sample_path.read_bytes()).hexdigest(),
            "result": {
            "value": 1.0,
            "statements": [answer],
            "verdicts": [{
            "statement": answer,
            "verdict": 1}]},
            "judge_model": "test-judge",
            "judge_model_digest": JUDGE_DIGEST,
            "judge_configuration": {
            "think": False},
            "ragas_version": "test-version",
            "created_at": "test-time"}


@dataclass(frozen=True)
class QualityInputs:
    """Paths to the three files a quality report is built from."""

    sample: Path
    evidence: Path
    profile: Path

    def paths(self) -> tuple[Path, Path, Path]:
        return self.sample, self.evidence, self.profile

    def edit(self, path: Path, **changes: Any) -> None:
        """Replace top-level fields of the JSON file at ``path``."""
        path.write_text(json.dumps(json.loads(path.read_text()) | changes))

    def edit_result(self, **changes: Any) -> None:
        """Replace fields of the saved faithfulness result."""
        result = json.loads(self.evidence.read_text())["result"]
        self.edit(self.evidence, result=result | changes)

    def rebind_evidence(self) -> None:
        """Point the faithfulness evidence at the sample as it is now."""
        self.edit(self.evidence, sample_sha256=hashlib.sha256(self.sample.read_bytes()).hexdigest())


def quality_inputs(
        tmp_path: Path, *, answer: str = FULL_ANSWER, contexts: tuple[str, ...] | None = None,
        cited_document: str = DOCUMENT) -> QualityInputs:
    """A captured answer, faithfulness evidence bound to it and a profile requiring two facts from one document."""
    contexts = (policy_context(), ) if contexts is None else contexts
    blocks = "".join(f"[CONTEXT {n}]:\n{context}\n[END CONTEXT {n}]\n" for n, context in enumerate(contexts))
    capture = {
            "schema_version": 1,
            "boundary": "ollama-sdk-chat",
            "request": {
            "model":
            GENERATION_MODEL,
            "stream":
            False,
            "messages": [{
            "role": "system",
            "content": f"[LLM_TESTKIT_CAPTURE:{CAPTURE_ID}]\n{blocks}".rstrip()}, {
            "role": "user",
            "content": QUESTION}]}}
    sample = build_sample(
            capture, question=QUESTION, answer=answer, reference="Expected facts", expected_model=GENERATION_MODEL,
            capture_id=CAPTURE_ID)
    sample["response_sources"] = [{"title": cited_document, "text": FULL_ANSWER}]
    sample_path = tmp_path / "sample.json"
    sample_path.write_text(json.dumps(sample))
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(json.dumps(faithfulness_evidence(sample_path, answer)))
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
            json.dumps({
            "schema_version": 1,
            "id": "leave",
            "question": QUESTION,
            "fact_patterns": {
            "leave allowance": "23 working days",
            "notice period": "12 calendar days"},
            "document_title_pattern": r"automation-[a-f0-9]{32}-company-policy\.txt",
            "source_fragments": ["23 working days", "12 calendar days"]}))
    return QualityInputs(sample_path, evidence_path, profile_path)
