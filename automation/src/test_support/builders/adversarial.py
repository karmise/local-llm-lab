"""Scenario data builders and deterministic test doubles."""

import hashlib
import json

import pytest
from requests import Response

from llm_testkit import assertions
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


def check_poisoned_copy_outcome(before, case, original, poisoned):
    if case.document_appendix:
        assert poisoned != original
        assert poisoned.read_text() == before.decode() + case.document_appendix
        assert hashlib.sha256(poisoned.read_bytes()).hexdigest() != DATASET.policy_sha256
        assertions.assert_attack_exposure(
            [poisoned.read_text()], attack_text=case.document_appendix
        )
        with pytest.raises(AssertionError, match="not exposed"):
            assertions.assert_attack_exposure(
                [original.read_text()], attack_text=case.document_appendix
            )
    else:
        assert poisoned == original


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
