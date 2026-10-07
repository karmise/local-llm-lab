import json
from pathlib import Path

import pytest

from llm_testkit import assertions
from llm_testkit.datasets.golden import CATEGORIES, load_golden_dataset
from llm_testkit.reporting.steps import title
from test_support.assertions import errors as errors
from test_support.assertions import values as value_checks
from test_support.builders.golden_dataset import (
    _response,
    invent_missing_benefit,
    prepare_invalid_catalog_is_rejected_case,
)
from test_support.data import common as case_data
from test_support.data.golden_dataset import (
    DATA_ROOT,
    DATASET,
    INVALID_ANSWER_EVIDENCE_CASES,
    INVALID_GOLDEN_CATALOG_CASES,
    MISSING_BENEFIT_CASE_IDS,
    REFERENCE_ACCEPTANCE_CASES,
    golden_case_id,
)
from test_support.fixtures.unit_golden_dataset import (
    golden_category_counts as golden_category_counts,
)

pytestmark = pytest.mark.unit


@title("Golden catalog covers every policy section and acceptance category")
def test_catalog_covers_policy_and_categories(golden_category_counts) -> None:
    counts = golden_category_counts
    value_checks.equal(set(counts), CATEGORIES)
    value_checks.equal(counts["missing_information"], 3)
    value_checks.equal(counts["boundary"], 3)
    value_checks.length(DATASET.cases, 16)
    value_checks.at_most(
        {"paid_leave", "travel_allowance", "remote_availability"},
        {case.id for case in DATASET.cases},
    )


@title("Golden paid-leave scenario preserves the shared API/UI reference")
def test_paid_leave_reference_matches_existing_profile() -> None:
    profile = json.loads((DATA_ROOT / "quality-paid-leave.json").read_text())
    case = next(case for case in DATASET.cases if case.id == "paid_leave")
    value_checks.equal(case.question, profile["question"])
    value_checks.equal(case.reference, profile["reference"])
    value_checks.equal(case.source_fragments, tuple(profile["source_fragments"]))
    assertions.assert_required_facts(case.reference, fact_patterns=profile["fact_patterns"])


@pytest.mark.parametrize("case", REFERENCE_ACCEPTANCE_CASES, ids=golden_case_id)
@title("Golden reference satisfies its configured answer and source criteria [{param_id}]")
def test_reference_satisfies_acceptance(case) -> None:
    assertions.assert_golden_answer(
        _response(case.reference), case=case, document_title=case_data.POLICY_DOCUMENT_TITLE
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    INVALID_GOLDEN_CATALOG_CASES,
)
@title("Golden loader rejects inconsistent or stale expectations [{param_id}]")
def test_invalid_catalog_is_rejected(tmp_path: Path, mutation: str, message: str) -> None:
    data = json.loads((DATA_ROOT / "golden-policy.json").read_text())
    case = data["cases"][0]
    prepare_invalid_catalog_is_rejected_case(case, data, mutation)
    path = tmp_path / "golden.json"
    path.write_text(json.dumps(data))
    errors.rejects(
        lambda: load_golden_dataset(path, DATA_ROOT / "company-policy.txt"),
        expected=ValueError,
        match=message,
    )


@pytest.mark.parametrize("case_id", MISSING_BENEFIT_CASE_IDS)
@title("Missing-information golden case rejects an invented benefit [{param_id}]")
def test_missing_policy_rejects_invented_benefit(case_id: str) -> None:
    case = next(case for case in DATASET.cases if case.id == case_id)
    invented = invent_missing_benefit(case_id)
    errors.rejects(
        lambda: assertions.assert_golden_answer(
            _response(case.reference + invented),
            case=case,
            document_title=case_data.POLICY_DOCUMENT_TITLE,
        ),
        expected=AssertionError,
        match="forbidden content",
    )


@pytest.mark.parametrize(
    ("answer", "source", "document", "message"),
    INVALID_ANSWER_EVIDENCE_CASES,
)
@title(
    "Golden criteria reject incomplete answers, unsupported sources and thinking-only facts [{param_id}]"
)
def test_bad_evidence_is_rejected(answer, source, document, message) -> None:
    errors.rejects(
        lambda: assertions.assert_golden_answer(
            _response(answer, source=source, document=document),
            case=DATASET.cases[0],
            document_title=case_data.POLICY_DOCUMENT_TITLE,
        ),
        expected=AssertionError,
        match=message,
    )
