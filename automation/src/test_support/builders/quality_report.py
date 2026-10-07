"""Scenario data builders and deterministic test doubles."""

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock

from llm_testkit.observation.evaluation_sample import build_sample
from test_support.data import common as case_data


def _files(tmp_path: Path, *, incomplete: bool = False) -> tuple[Path, Path, Path]:
    identifier = "a" * 32
    title = f"automation-{identifier}-company-policy.txt"
    content = f"<document_metadata>\nsourceDocument: {title}\n</document_metadata>\n23 working days; 12 calendar days"
    capture = {
        "schema_version": 1,
        "boundary": "ollama-sdk-chat",
        "request": {
            "model": case_data.TEST_MODEL,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": f"[LLM_TESTKIT_CAPTURE:{identifier}]\n[CONTEXT 0]:\n{content}\n[END CONTEXT 0]",
                },
                {"role": "user", "content": "Leave?"},
            ],
        },
    }
    answer = "23 working days" if incomplete else "23 working days; 12 calendar days"
    sample = build_sample(
        capture,
        question="Leave?",
        answer=answer,
        reference="Expected facts",
        expected_model=case_data.TEST_MODEL,
        capture_id=identifier,
    )
    sample["response_sources"] = [{"title": title, "text": "23 working days; 12 calendar days"}]
    sample_path = tmp_path / case_data.SAMPLE_FILE_NAME
    sample_path.write_text(json.dumps(sample))
    evidence = {
        "schema_version": 1,
        "metric": "faithfulness",
        "status": "completed",
        "sample_sha256": hashlib.sha256(sample_path.read_bytes()).hexdigest(),
        "result": {
            "value": 1.0,
            "statements": [answer],
            "verdicts": [{"statement": answer, "verdict": 1}],
        },
        "judge_model": "test-judge",
        "judge_model_digest": case_data.MODEL_DIGEST,
        "judge_configuration": {"think": False},
        "ragas_version": "test-version",
        "created_at": "test-time",
    }
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(json.dumps(evidence))
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "id": "leave",
                "question": "Leave?",
                "fact_patterns": {
                    "leave allowance": "23 working days",
                    "notice period": "12 calendar days",
                },
                "document_title_pattern": r"automation-[a-f0-9]{32}-company-policy\.txt",
                "source_fragments": ["23 working days", "12 calendar days"],
            }
        )
    )
    return sample_path, evidence_path, profile_path


def prepare_invalid_judge_evidence_case(change, evidence):
    if change == "checksum":
        evidence["sample_sha256"] = "another-sample"
    elif change == "score":
        evidence["result"]["value"] = 0.5
    elif change == "unfinished":
        evidence["status"] = "error"
    else:
        evidence["result"]["verdicts"] = []


def make_step_stub(visited):
    @contextmanager
    def step(name: str):
        visited.append(name)
        yield

    return step


def make_load_dataset_stub(loads):
    def load_dataset(*args):
        loads.append(1)
        return Mock(sha256="a" * 64)

    return load_dataset


def make_check_stub(checks, evidence_path):
    def check(evidence, *args):
        checks.append(evidence)
        evidence_path.write_text('{"observation": 2}')
        return {
            "context_precision": evidence["observation"] / 2,
            "context_recall": evidence["observation"] / 2,
        }

    return check


def make_claim_extraction_call(evidence):
    """Build input for test_saved_faithfulness_rejects_summary_changed_from_raw_judge_calls."""
    return {"output": {"statements": evidence["result"]["statements"]}}
