import json
from collections import Counter
from pathlib import Path

import pytest

from llm_testkit import assertions
from llm_testkit.datasets.golden import CATEGORIES, load_golden_dataset
from llm_testkit.reporting.steps import title
from test_support.builders.golden_dataset import (
    _response,
    prepare_invalid_catalog_is_rejected_case,
    prepare_missing_benefit_step_2,
)
from test_support.data.golden_dataset import (
    BAD_EVIDENCE_IS_REJECTED_ANSWER_SOURCE_DOCUMENT_MESSAGE_CASES,
    DATA_ROOT,
    DATASET,
    INVALID_CATALOG_IS_REJECTED_MUTATION_MESSAGE_CASES,
    MISSING_BENEFIT_CASE_ID_CASES,
    REFERENCE_SATISFIES_ACCEPTANCE_CASE_CASES,
    REFERENCE_SATISFIES_ACCEPTANCE_CASE_IDS,
)

pytestmark = pytest.mark.unit


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


@pytest.mark.parametrize(
    "case", REFERENCE_SATISFIES_ACCEPTANCE_CASE_CASES, ids=REFERENCE_SATISFIES_ACCEPTANCE_CASE_IDS
)
@title("Golden reference satisfies its configured answer and source criteria [{param_id}]")
def test_reference_satisfies_acceptance(case) -> None:
    assertions.assert_golden_answer(
        _response(case.reference), case=case, document_title="policy.txt"
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    INVALID_CATALOG_IS_REJECTED_MUTATION_MESSAGE_CASES,
)
@title("Golden loader rejects inconsistent or stale expectations [{param_id}]")
def test_invalid_catalog_is_rejected(tmp_path: Path, mutation: str, message: str) -> None:
    data = json.loads((DATA_ROOT / "golden-policy.json").read_text())
    case = data["cases"][0]
    prepare_invalid_catalog_is_rejected_case(case, data, mutation)
    path = tmp_path / "golden.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=message):
        load_golden_dataset(path, DATA_ROOT / "company-policy.txt")


@pytest.mark.parametrize("case_id", MISSING_BENEFIT_CASE_ID_CASES)
@title("Missing-information golden case rejects an invented benefit [{param_id}]")
def test_missing_policy_rejects_invented_benefit(case_id: str) -> None:
    case = next(case for case in DATASET.cases if case.id == case_id)
    invented = prepare_missing_benefit_step_2(case_id)
    with pytest.raises(AssertionError, match="forbidden content"):
        assertions.assert_golden_answer(
            _response(case.reference + invented), case=case, document_title="policy.txt"
        )


@pytest.mark.parametrize(
    ("answer", "source", "document", "message"),
    BAD_EVIDENCE_IS_REJECTED_ANSWER_SOURCE_DOCUMENT_MESSAGE_CASES,
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
