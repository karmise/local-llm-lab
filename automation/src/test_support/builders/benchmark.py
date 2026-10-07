"""Scenario data builders and deterministic test doubles."""

import hashlib
from unittest.mock import Mock

from llm_testkit.evaluation import benchmark as evaluation
from llm_testkit.observation.evaluation_sample import build_sample
from llm_testkit.reporting.gates import METRICS
from test_support.paths import AUTOMATION_ROOT

ROOT = AUTOMATION_ROOT


def _sample(case, dataset, model="qwen3.5:4b"):
    identifier = "a" * 32
    document = f"automation-{identifier}-company-policy.txt"
    context = (
        f"<document_metadata>\nsourceDocument: {document}\n</document_metadata>\n"
        + (ROOT / "test_data/company-policy.txt").read_text()
    )
    capture = {
        "schema_version": 1,
        "boundary": "ollama-sdk-chat",
        "request": {
            "model": model,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": f"[LLM_TESTKIT_CAPTURE:{identifier}]\n[CONTEXT 0]:\n{context}\n[END CONTEXT 0]",
                },
                {"role": "user", "content": case.question},
            ],
        },
    }
    sample = build_sample(
        capture,
        question=case.question,
        answer=case.reference,
        reference=case.reference,
        expected_model=model,
        capture_id=identifier,
    )
    sample["metadata"] = {
        "golden_case_id": case.id,
        "golden_category": case.category,
        "golden_dataset_sha256": dataset.sha256,
        "policy_sha256": dataset.policy_sha256,
        "model_digest": "generation-digest",
        "workspace_configuration": {"openAiPrompt": "Policy only", "chatModel": model},
    }
    sample["response_sources"] = [{"title": document, "text": context}]
    return sample


def _row(case_id="paid_leave", category="multi_fact"):
    refusal = category == "missing_information"
    return {
        "case_id": case_id,
        "category": category,
        "model": "qwen3.5:4b",
        "model_digest": "generation-digest",
        "workspace_configuration": {"openAiPrompt": "Policy only"},
        "generation_status": "passed",
        "dimensions": [
            {"name": "Reviewed answer rules", "status": "passed"},
            {"name": "Document sources", "status": "passed"},
            *[
                {
                    "name": m,
                    "metric": m,
                    "status": "not_applicable" if refusal else "passed",
                    "value": 1.0,
                    "minimum": {
                        "faithfulness": 0.9,
                        "factual_correctness": 0.8,
                        "context_precision": 0.8,
                        "context_recall": 0.9,
                    }[m],
                }
                for m in sorted(METRICS)
            ],
        ],
    }


def _mock_metrics(monkeypatch, case, dataset, sample_path):
    checksum = hashlib.sha256(sample_path.read_bytes()).hexdigest()
    verdict = {"statement": case.reference, "verdict": 1, "reason": "Supported"}
    faith = {
        "schema_version": 1,
        "metric": "faithfulness",
        "status": "completed",
        "sample_sha256": checksum,
        "judge_model": "qwen3.5:4b",
        "judge_model_digest": "judge-digest",
        "judge_configuration": {"think": False},
        "ragas_version": "0.4.3",
        "created_at": "now",
        "result": {"value": 1.0, "statements": [case.reference], "verdicts": [verdict]},
        "judge_calls": [
            {"output": {"statements": [case.reference]}},
            {"output": {"statements": [verdict]}},
        ],
    }
    from llm_testkit.evaluation.correctness import METRIC_CONFIGURATION

    correct = {
        **faith,
        "metric": "factual_correctness",
        "response_origin": "application_sample",
        "golden_dataset_sha256": dataset.sha256,
        "golden_case_id": case.id,
        "metric_configuration": METRIC_CONFIGURATION,
        "question": case.question,
        "reference": case.reference,
        "response": case.reference,
        "reference_sha256": hashlib.sha256(case.reference.encode()).hexdigest(),
        "result": {
            "value": 1.0,
            "response_claims": [case.reference],
            "reference_claims": [case.reference],
            "response_verdicts": [verdict],
            "reference_verdicts": [verdict],
        },
        "judge_calls": [
            {"output": {"claims": [case.reference]}},
            {"output": {"statements": [verdict]}},
            {"output": {"claims": [case.reference]}},
            {"output": {"statements": [verdict]}},
        ],
    }
    precision = {"verdict": 1, "reason": "Relevant"}
    recall = {"statement": case.reference, "attributed": 1, "reason": "Supported"}
    relevance = {
        **correct,
        "metric": "context_relevance",
        "result": {
            "context_precision": 1 / (1 + 1e-10),
            "context_recall": 1.0,
            "precision_verdicts": [precision],
            "recall_classifications": [recall],
        },
        "precision_calls": [{"output": precision}],
        "recall_calls": [{"output": {"classifications": [recall]}}],
    }
    relevance.pop("judge_calls")
    for function, result in (
        ("evaluate_sample_report", faith),
        ("evaluate_correctness_report", correct),
        ("evaluate_relevance_report", relevance),
    ):
        monkeypatch.setattr(evaluation, function, Mock(return_value=result))
