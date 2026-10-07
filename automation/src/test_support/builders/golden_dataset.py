"""Scenario data builders and deterministic test doubles."""

import json
from copy import deepcopy

from requests import Response

from test_support.data.golden_dataset import DATA_ROOT as DATA_ROOT
from test_support.data.golden_dataset import DATASET as DATASET


def _response(answer: str, *, source: str | None = None, document: str = "policy.txt") -> Response:
    response = Response()
    response.status_code = 200
    response._content = json.dumps(
        {
            "type": "textResponse",
            "error": None,
            "close": True,
            "textResponse": answer,
            "sources": [
                {
                    "title": document,
                    "text": source or (DATA_ROOT / "company-policy.txt").read_text(),
                }
            ],
        }
    ).encode()
    return response


def prepare_invalid_catalog_is_rejected_case(case, data, mutation):
    if mutation == "duplicate":
        data["cases"].append(deepcopy(case))
    elif mutation == "checksum":
        data["policy_sha256"] = "0" * 64
    elif mutation == "category":
        case["category"] = "unknown"
    elif mutation == "pattern":
        case["required_patterns"] = {"bad": "["}
    elif mutation == "vacuous":
        case["required_patterns"] = {"bad": ".*"}
    elif mutation == "fragment":
        case["source_fragments"] = ["Invented source text"]
    elif mutation == "reference":
        case["reference"] = "No information."
    elif mutation == "conflict":
        case["forbidden_patterns"] = {"conflict": "23"}
    elif mutation == "schema":
        data["schema_version"] = True
    elif mutation == "empty":
        data["cases"] = []


def invent_missing_benefit(case_id):
    invented = (
        " An allowance of KGS 500 is available."
        if case_id != "parental_leave_missing"
        else " Employees receive 30 days."
    )

    return invented
