"""Scenario data builders and deterministic test doubles."""

import json

from requests import Response

from test_support.data.adversarial import CASES as CASES
from test_support.data.adversarial import DATA as DATA
from test_support.data.adversarial import DATASET as DATASET


def answer(case, text):
    response = Response()
    response.status_code = 200
    response._content = json.dumps(
        {
            "type": "textResponse",
            "textResponse": text,
            "close": True,
            "error": None,
            "sources": [{"title": "policy.txt", "text": (DATA / "company-policy.txt").read_text()}],
        }
    ).encode()
    return response


def prepare_catalog_case(change, data, row):
    if change == "hash":
        data["golden_dataset_sha256"] = "changed"
    elif change == "id":
        data["cases"][1]["id"] = row["id"]
    elif change == "category":
        row["category"] = "unknown"
    elif change == "base":
        row["golden_case_id"] = "unknown"
    elif change == "question":
        row["question"] = "Unrelated question"
    elif change == "appendix":
        row["document_appendix"] = "Unexpected modification"
    else:
        row["forbidden_patterns"] = {"empty": ".*"}
