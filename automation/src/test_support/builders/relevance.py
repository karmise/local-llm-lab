"""Scenario data builders and deterministic test doubles."""

from copy import deepcopy

from test_support.data.relevance import CASE as CASE
from test_support.data.relevance import DATASET as DATASET
from test_support.data.relevance import ROOT as ROOT


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


def make_Judge_schema():
    from pydantic import BaseModel
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

    return Judge


def prepare_invalid_relevance_case(change, data):
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


def prepare_evidence_binding_case(change, evidence):
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


def make_failed_score_stub():
    async def failed_score(*args):
        raise ValueError("Truncated response")

    return failed_score


def make_relevance_evidence():
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
    return sample, evidence, data


def make_oversized_retrieval():
    """Build input for test_context_budget."""
    return {"retrieved_contexts": ["Context"] * 5}
