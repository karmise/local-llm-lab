"""Scenario data builders and deterministic test doubles."""

import hashlib
from copy import deepcopy

from llm_testkit.datasets.golden import load_golden_dataset
from llm_testkit.evaluation.correctness import METRIC_CONFIGURATION
from llm_testkit.observation.evaluation_sample import build_sample
from test_support.paths import AUTOMATION_ROOT

ROOT = AUTOMATION_ROOT / "test_data"


DATASET = load_golden_dataset(ROOT / "golden-policy.json", ROOT / "company-policy.txt")


CASE = next(case for case in DATASET.cases if case.id == "paid_leave")


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
            "model": "test-model",
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
        expected_model="test-model",
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
        "judge_model": "test-model",
        "judge_model_digest": "digest",
    }
