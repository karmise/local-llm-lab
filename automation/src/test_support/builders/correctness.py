"""Scenario data builders and deterministic test doubles."""

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from requests import Response

from llm_testkit.evaluation.correctness import METRIC_CONFIGURATION
from llm_testkit.observation.evaluation_sample import build_sample
from llm_testkit.reporting.quality import build_quality_report
from test_support.data import common as case_data
from test_support.data.correctness import CASE as CASE
from test_support.data.correctness import DATASET as DATASET
from test_support.data.correctness import (
    EDITED_RAW_CLAIMS,
    EVIDENCE_MUTATION_FIELDS,
    INCOMPLETE_REFERENCE_CLAIMS,
    MODIFIED_EVIDENCE_VALUE,
)
from test_support.data.correctness import ROOT as ROOT


def _verdicts(claims, values):
    return [
        {"statement": claim, "verdict": value, "reason": "Test label"}
        for claim, value in zip(claims, values, strict=True)
    ]


def _result(response_values=(1, 1), reference_values=(1, 1), value=1.0):
    response = [f"Response claim {i}" for i in range(len(response_values))]
    reference = [f"Reference claim {i}" for i in range(len(reference_values))]
    return {
        "value": value,
        "response_claims": response,
        "reference_claims": reference,
        "response_verdicts": _verdicts(response, response_values),
        "reference_verdicts": _verdicts(reference, reference_values),
    }


def _sample():
    identifier = "a" * 32
    name = f"automation-{identifier}-company-policy.txt"
    context = (
        f"<document_metadata>\nsourceDocument: {name}\n</document_metadata>\n"
        + (ROOT / "company-policy.txt").read_text()
    )
    observation = {
        "schema_version": 1,
        "boundary": "ollama-sdk-chat",
        "request": {
            "model": case_data.TEST_MODEL,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": f"[LLM_TESTKIT_CAPTURE:{identifier}]\n[CONTEXT 0]:\n{context}\n[END CONTEXT 0]",
                },
                {"role": "user", "content": CASE.question},
            ],
        },
    }
    sample = build_sample(
        observation,
        question=CASE.question,
        answer=CASE.reference,
        reference=CASE.reference,
        expected_model=case_data.TEST_MODEL,
        capture_id=identifier,
    )
    sample["response_sources"] = [{"title": name, "text": context}]
    return sample


def _calls(result):
    return [
        {"output": {"claims": result["response_claims"]}},
        {"output": {"statements": result["response_verdicts"]}},
        {"output": {"claims": result["reference_claims"]}},
        {"output": {"statements": result["reference_verdicts"]}},
    ]


def _evidence(sample, checksum="sample"):
    return {
        "schema_version": 1,
        "metric": "factual_correctness",
        "status": "completed",
        "sample_sha256": checksum,
        "golden_dataset_sha256": DATASET.sha256,
        "golden_case_id": CASE.id,
        "reference_sha256": hashlib.sha256(CASE.reference.encode()).hexdigest(),
        "response_origin": "application_sample",
        "question": CASE.question,
        "reference": CASE.reference,
        "response": sample["response"],
        "metric_configuration": deepcopy(METRIC_CONFIGURATION),
        "result": _result(),
        "judge_calls": _calls(_result()),
        "judge_model": case_data.TEST_MODEL,
        "judge_model_digest": case_data.MODEL_DIGEST,
    }


def prepare_invalid_claim_evidence_is_rejected_case(change, result):
    if change == "missing":
        result["reference_verdicts"].pop()
    elif change == "duplicate":
        result["response_claims"][1] = result["response_claims"][0]
    elif change == "invalid-verdict":
        result["response_verdicts"][0]["verdict"] = True
    elif change == "score":
        result["value"] = 0.5
    elif change == "counts":
        result["counts"] = {"tp": 9, "fp": 0, "fn": 0}
    elif change == "empty":
        result["response_claims"] = []
    else:
        result["reference_verdicts"][0]["reason"] = ""


def make_failed_score_stub():
    async def failed_score(sample, judge):
        raise ValueError("Truncated judge response")

    return failed_score


def append_judge_responses(outputs, responses):
    for output in outputs:
        response = Response()
        response.status_code = 200
        response._content = json.dumps(
            {
                "model": case_data.TEST_MODEL,
                "done": True,
                "done_reason": "stop",
                "message": {"content": json.dumps(output)},
            }
        ).encode()
        responses.append(response)


def make_valid_faithfulness_evidence(checksum):
    """Build input for test_quality_report_adds_independent_correctness_measurement."""
    return {
        "schema_version": 1,
        "metric": "faithfulness",
        "status": "completed",
        "sample_sha256": checksum,
        "result": {
            "value": 1.0,
            "statements": ["Claim"],
            "verdicts": [{"statement": "Claim", "verdict": 1}],
        },
        "judge_model": "judge",
        "judge_model_digest": case_data.MODEL_DIGEST,
        "judge_configuration": {},
        "ragas_version": "test",
        "created_at": "now",
    }


@dataclass
class CorrectnessEvidenceScenario:
    sample: dict[str, Any]
    evidence: dict[str, Any]

    def invalidate(self, change: str) -> None:
        self.evidence[EVIDENCE_MUTATION_FIELDS[change]] = MODIFIED_EVIDENCE_VALUE

    def corrupt_raw_claims(self) -> None:
        self.evidence["judge_calls"][0]["output"]["claims"] = list(EDITED_RAW_CLAIMS)


@dataclass
class CorrectnessQualityScenario:
    sample_path: Path
    faithfulness_path: Path
    correctness_path: Path
    correctness: dict[str, Any]

    def build_report(self) -> dict[str, Any]:
        return build_quality_report(
            self.sample_path,
            self.faithfulness_path,
            ROOT / "quality-paid-leave.json",
            correctness_path=self.correctness_path,
        )

    def invalidate_sample_checksum(self) -> None:
        self.correctness["sample_sha256"] = "other"
        self.correctness_path.write_text(json.dumps(self.correctness))


@dataclass
class IncompleteControlScenario:
    control: dict[str, Any]
    result: dict[str, Any]

    def attribute_missing_claim(self) -> None:
        self.result["reference_verdicts"] = _verdicts(INCOMPLETE_REFERENCE_CLAIMS, (1, 1))
        self.result["value"] = 1.0


def make_quality_scenario(tmp_path: Path) -> CorrectnessQualityScenario:
    sample = _sample()
    source = tmp_path / case_data.SAMPLE_FILE_NAME
    source.write_text(json.dumps(sample))
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    faith_path = tmp_path / "faith.json"
    faith_path.write_text(json.dumps(make_valid_faithfulness_evidence(checksum)))
    correctness = _evidence(sample, checksum)
    correctness["result"] = _result((0, 0), (0, 0), 0.0)
    correctness["judge_calls"] = _calls(correctness["result"])
    correct_path = tmp_path / "correct.json"
    correct_path.write_text(json.dumps(correctness))
    return CorrectnessQualityScenario(source, faith_path, correct_path, correctness)


def make_incomplete_control(controls) -> IncompleteControlScenario:
    control = next(c for c in controls if c["id"] == "incomplete")
    result = _result((1, 1), (1, 0), 0.8)
    result["reference_claims"] = list(INCOMPLETE_REFERENCE_CLAIMS)
    result["reference_verdicts"] = _verdicts(INCOMPLETE_REFERENCE_CLAIMS, (1, 0))
    return IncompleteControlScenario(control, result)
