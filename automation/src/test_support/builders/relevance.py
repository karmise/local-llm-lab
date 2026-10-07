"""Scenario data builders and deterministic test doubles."""

from llm_testkit.datasets.golden import load_golden_dataset
from test_support.paths import AUTOMATION_ROOT

ROOT = AUTOMATION_ROOT / "test_data"


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
