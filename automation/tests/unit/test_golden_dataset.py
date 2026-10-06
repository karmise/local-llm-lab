import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest
from requests import Response

from llm_testkit import assertions
from llm_testkit.datasets.golden import CATEGORIES, load_golden_dataset
from llm_testkit.reporting.steps import title

pytestmark = pytest.mark.unit
DATA_ROOT = Path(__file__).resolve().parents[2] / "test_data"
DATASET = load_golden_dataset(DATA_ROOT / "golden-policy.json", DATA_ROOT / "company-policy.txt")


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


@title("Golden catalog covers every policy section and acceptance category")
def test_catalog_covers_policy_and_categories() -> None:
    counts = Counter(case.category for case in DATASET.cases)
    assert set(counts) == CATEGORIES
    assert counts["missing_information"] == 3
    assert counts["boundary"] == 3
    assert len(DATASET.cases) == 16
    assert {"paid_leave", "travel_allowance", "remote_availability"} <= {
        case.id for case in DATASET.cases
    }


@title("Golden paid-leave scenario preserves the shared API/UI reference")
def test_paid_leave_reference_matches_existing_profile() -> None:
    profile = json.loads((DATA_ROOT / "quality-paid-leave.json").read_text())
    case = next(case for case in DATASET.cases if case.id == "paid_leave")
    assert case.question == profile["question"]
    assert case.reference == profile["reference"]
    assert case.source_fragments == tuple(profile["source_fragments"])
    assertions.assert_required_facts(case.reference, fact_patterns=profile["fact_patterns"])


@pytest.mark.parametrize("case", DATASET.cases, ids=lambda case: case.id)
@title("Golden reference satisfies its configured answer and source criteria [{param_id}]")
def test_reference_satisfies_acceptance(case) -> None:
    assertions.assert_golden_answer(
        _response(case.reference), case=case, document_title="policy.txt"
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate", "duplicate golden case id"),
        ("checksum", "policy checksum mismatch"),
        ("category", "unknown category"),
        ("pattern", "invalid regex"),
        ("vacuous", "must not match empty text"),
        ("fragment", "absent from policy"),
        ("reference", "reference does not satisfy"),
        ("conflict", "reference violates forbidden"),
        ("schema", "integer schema_version"),
        ("empty", "must contain cases"),
    ],
)
@title("Golden loader rejects inconsistent or stale expectations [{param_id}]")
def test_invalid_catalog_is_rejected(tmp_path: Path, mutation: str, message: str) -> None:
    data = json.loads((DATA_ROOT / "golden-policy.json").read_text())
    case = data["cases"][0]
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
    path = tmp_path / "golden.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=message):
        load_golden_dataset(path, DATA_ROOT / "company-policy.txt")


@pytest.mark.parametrize("case_id", ["gym_missing", "bonus_missing", "parental_leave_missing"])
@title("Missing-information golden case rejects an invented benefit [{param_id}]")
def test_missing_policy_rejects_invented_benefit(case_id: str) -> None:
    case = next(case for case in DATASET.cases if case.id == case_id)
    invented = (
        " An allowance of KGS 500 is available."
        if case_id != "parental_leave_missing"
        else " Employees receive 30 days."
    )
    with pytest.raises(AssertionError, match="forbidden content"):
        assertions.assert_golden_answer(
            _response(case.reference + invented), case=case, document_title="policy.txt"
        )


@pytest.mark.parametrize(
    ("answer", "source", "document", "message"),
    [
        ("23 working days of paid leave.", None, "policy.txt", "advance notice"),
        (DATASET.cases[0].reference, "23 working days", "policy.txt", "12 calendar days"),
        (DATASET.cases[0].reference, None, "other.txt", "did not cite"),
        (
            "<think>23 working days and 12 calendar days before leave</think>No answer.",
            None,
            "policy.txt",
            "annual allowance",
        ),
    ],
)
@title(
    "Golden criteria reject incomplete answers, unsupported sources and thinking-only facts [{param_id}]"
)
def test_bad_evidence_is_rejected(answer, source, document, message) -> None:
    with pytest.raises(AssertionError, match=message):
        assertions.assert_golden_answer(
            _response(answer, source=source, document=document),
            case=DATASET.cases[0],
            document_title="policy.txt",
        )
