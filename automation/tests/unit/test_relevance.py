import asyncio
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import pytest
from pydantic import BaseModel

from llm_testkit.datasets.golden import load_golden_dataset
from llm_testkit.evaluation.relevance import (
    check_relevance_evidence,
    score_relevance,
    validate_relevance,
)
from llm_testkit.reporting.steps import title

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2] / "test_data"
DATASET = load_golden_dataset(ROOT / "golden-policy.json", ROOT / "company-policy.txt")
CASE = next(c for c in DATASET.cases if c.id == "paid_leave")


def result(values=(1, 0, 1)):
    return {
        "context_precision": sum(sum(values[: i + 1]) / (i + 1) * v for i, v in enumerate(values))
        / (sum(values) + 1e-10),
        "context_recall": 0.5,
        "precision_verdicts": [{"verdict": v, "reason": "Labelled context"} for v in values],
        "recall_classifications": [
            {
                "statement": "Employees receive 23 working days of paid leave.",
                "attributed": 1,
                "reason": "Present",
            },
            {
                "statement": "Request at least 12 calendar days before leave.",
                "attributed": 0,
                "reason": "Absent",
            },
        ],
    }


@title("Real RAGAS context metrics preserve retrieval order and detect missing reference facts")
def test_real_ragas_metrics_with_mocked_judge():
    pytest.importorskip("ragas")
    from ragas.llms.base import InstructorBaseRagasLLM

    class Judge(InstructorBaseRagasLLM):
        def __init__(self, outputs):
            self.outputs = iter(outputs)
            self.calls = []

        def generate(self, prompt: str, response_model: type[BaseModel]):
            parsed = response_model.model_validate(next(self.outputs), strict=True)
            self.calls.append({"output": parsed.model_dump()})
            return parsed

        async def agenerate(self, prompt, response_model):
            return self.generate(prompt, response_model)

    expected = result()
    precision = Judge(expected["precision_verdicts"])
    recall = Judge([{"classifications": expected["recall_classifications"]}])
    sample = {
        "user_input": CASE.question,
        "reference": CASE.reference,
        "retrieved_contexts": ["Relevant", "Unrelated", "Relevant"],
    }
    assert asyncio.run(score_relevance(sample, CASE, precision, recall)) == expected
    assert len(precision.calls) == 3
    assert len(recall.calls) == 1
    assert expected["context_precision"] == pytest.approx(5 / 6)


@pytest.mark.parametrize(
    "change", ["binary", "reason", "score", "duplicate", "missing_fact", "count", "empty"]
)
@title("Context metrics reject invalid verdicts and incomplete reference coverage [{param_id}]")
def test_invalid_relevance(change):
    data = result()
    if change == "binary":
        data["precision_verdicts"][0]["verdict"] = True
    elif change == "reason":
        data["recall_classifications"][0]["reason"] = ""
    elif change == "score":
        data["context_recall"] = 1.0
    elif change == "duplicate":
        data["recall_classifications"][1] = deepcopy(data["recall_classifications"][0])
    elif change == "missing_fact":
        data["recall_classifications"][1]["statement"] = "Some deadline."
    elif change == "count":
        data["precision_verdicts"].pop()
    else:
        data["recall_classifications"] = []
    with pytest.raises(ValueError):
        validate_relevance(data, CASE, 3)


@pytest.mark.parametrize(
    "change", ["none", "checksum", "dataset", "question", "raw_calls", "unfinished"]
)
@title("Context evidence is bound to the sample, golden dataset and raw calls [{param_id}]")
def test_evidence_binding(change):
    sample = {
        "user_input": CASE.question,
        "reference": CASE.reference,
        "retrieved_contexts": ["A", "B", "C"],
    }
    data = result()
    evidence = {
        "schema_version": 1,
        "metric": "context_relevance",
        "status": "completed",
        "sample_sha256": "sample",
        "golden_dataset_sha256": DATASET.sha256,
        "golden_case_id": CASE.id,
        "question": CASE.question,
        "reference": CASE.reference,
        "result": data,
        "precision_calls": [{"output": deepcopy(v)} for v in data["precision_verdicts"]],
        "recall_calls": [{"output": {"classifications": deepcopy(data["recall_classifications"])}}],
    }
    if change == "none":
        assert check_relevance_evidence(evidence, "sample", sample, DATASET) == data
        return
    if change == "raw_calls":
        evidence["precision_calls"][0]["output"]["verdict"] = 0
    else:
        field = {
            "checksum": "sample_sha256",
            "dataset": "golden_dataset_sha256",
            "question": "question",
            "unfinished": "status",
        }[change]
        evidence[field] = "changed"
    with pytest.raises(ValueError):
        check_relevance_evidence(evidence, "sample", sample, DATASET)


@title("Context evaluation rejects oversized retrieval without silently truncating or judging")
def test_context_budget():
    pytest.importorskip("ragas")
    judge = Mock()
    with pytest.raises(ValueError, match="never truncated"):
        asyncio.run(score_relevance({"retrieved_contexts": ["Context"] * 5}, CASE, judge, judge))
    assert not judge.mock_calls
