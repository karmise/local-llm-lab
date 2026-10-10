"""Golden policy dataset: reviewed, source-bound expectations and the answer criteria built on them."""

import hashlib
import json
from collections import Counter

import pytest

from llm_testkit import assertions
from llm_testkit.datasets.golden import CATEGORIES, load_golden_dataset
from llm_testkit.reporting.steps import title
from test_support.builders.anythingllm import chat_reply
from test_support.builders.golden import CASES, GOLDEN_DATASET, GOLDEN_DATASET_FILE, PAID_LEAVE, POLICY_FILE, TEST_DATA
from test_support.data.common import POLICY_DOCUMENT_TITLE

pytestmark = pytest.mark.unit


def catalog() -> dict:
    return json.loads(GOLDEN_DATASET_FILE.read_text())


def load(data: dict, tmp_path):
    path = tmp_path / "golden.json"
    path.write_text(json.dumps(data))
    return load_golden_dataset(path, POLICY_FILE)


def first_case(**fields):
    return lambda data: data["cases"][0].update(fields)


def check_answer(answer: str, case=PAID_LEAVE, **reply):
    assertions.assert_golden_answer(chat_reply(answer, **reply), case=case, document_title=POLICY_DOCUMENT_TITLE)


@title("The golden catalog has 16 cases covering every category and the main policy sections")
def test_catalog_covers_policy_and_categories():
    counts = Counter(case.category for case in GOLDEN_DATASET.cases)

    assert len(GOLDEN_DATASET.cases) == 16
    assert set(counts) == CATEGORIES
    assert (counts["missing_information"], counts["boundary"]) == (3, 3)
    assert {"paid_leave", "travel_allowance", "remote_availability"} <= set(CASES)


@title("The loaded dataset records its version and the checksums of the catalog and the policy")
def test_catalog_provenance():
    # Load here: the shared GOLDEN_DATASET is built at import time, before a test could observe the loader.
    dataset = load_golden_dataset(GOLDEN_DATASET_FILE, POLICY_FILE)

    assert dataset.version == catalog()["version"]
    assert dataset.sha256 == hashlib.sha256(GOLDEN_DATASET_FILE.read_bytes()).hexdigest()
    assert dataset.policy_sha256 == hashlib.sha256(POLICY_FILE.read_bytes()).hexdigest()


@title("A loaded case keeps every reviewed field, with patterns and fragments as ordered tuples")
def test_case_fields():
    row = next(row for row in catalog()["cases"] if row["id"] == "paid_leave")
    loaded = next(
            case for case in load_golden_dataset(GOLDEN_DATASET_FILE, POLICY_FILE).cases if case.id == "paid_leave")

    assert (loaded.category, loaded.question, loaded.reference,
            loaded.rationale) == (row["category"], row["question"], row["reference"], row["rationale"])
    assert loaded.required_patterns == tuple(row["required_patterns"].items())
    assert loaded.forbidden_patterns == ()
    assert loaded.source_fragments == tuple(row["source_fragments"])


@title("The golden paid-leave case uses the same question, reference and sources as the API/UI profile")
def test_paid_leave_matches_shared_profile():
    profile = json.loads((TEST_DATA / "quality-paid-leave.json").read_text())

    assert (PAID_LEAVE.question, PAID_LEAVE.reference) == (profile["question"], profile["reference"])
    assert PAID_LEAVE.source_fragments == tuple(profile["source_fragments"])
    assertions.assert_required_facts(PAID_LEAVE.reference, fact_patterns=profile["fact_patterns"])


@title("Required rules match references regardless of letter case and line breaks")
def test_reference_rules_ignore_case_and_whitespace(tmp_path):
    data = catalog()
    data["cases"] = [data["cases"][0]]
    data["cases"][0]["required_patterns"] = {"allowance": "23 working days"}
    data["cases"][0]["reference"] = "Each employee receives 23\n  WORKING days."

    assert load(data, tmp_path).cases[0].reference == "Each employee receives 23\n  WORKING days."


@title("A case may have no forbidden rules")
def test_forbidden_rules_are_optional(tmp_path):
    data = catalog()
    data["cases"][0]["forbidden_patterns"] = {}

    assert load(data, tmp_path).cases[0].forbidden_patterns == ()


@pytest.mark.parametrize(("corrupt", "message"), [
        pytest.param(lambda d: d.update(schema_version=True), "integer schema_version 1", id="boolean-schema"),
        pytest.param(lambda d: d.update(schema_version=2), "integer schema_version 1", id="schema-two"),
        pytest.param(lambda d: d.pop("version"), "Dataset version must be a nonempty string", id="no-version"),
        pytest.param(lambda d: d.update(policy_sha256="0" * 64), "policy checksum mismatch", id="policy-checksum"),
        pytest.param(lambda d: d.update(cases=[]), "must contain cases", id="no-cases"),
        pytest.param(lambda d: d.update(cases={}), "must contain cases", id="cases-not-a-list"),
        pytest.param(
        lambda d: d["cases"].__setitem__(0, "paid_leave"), "Golden case must be an object", id="case-not-object"),
        pytest.param(first_case(id=""), "Case id must be a nonempty string", id="blank-id"),
        pytest.param(first_case(id="Paid Leave"), "Invalid or duplicate golden case id: Paid Leave", id="invalid-id"),
        pytest.param(
        lambda d: d["cases"].append(dict(d["cases"][0])), "Invalid or duplicate golden case id", id="duplicate-id"),
        pytest.param(first_case(category="unknown"), "unknown category", id="unknown-category"),
        pytest.param(first_case(category=["boundary"]), "unknown category", id="category-not-text"),
        pytest.param(first_case(question=" "), "paid_leave: question must be", id="blank-question"),
        pytest.param(first_case(reference=""), "paid_leave: reference must be", id="blank-reference"),
        pytest.param(first_case(rationale=None), "paid_leave: rationale must be", id="no-rationale"),
        pytest.param(first_case(required_patterns={}), "must be a nonempty object", id="no-required-rules"),
        pytest.param(first_case(required_patterns=["23"]), "paid_leave must be an object", id="rules-not-object"),
        pytest.param(first_case(required_patterns={"": "23"}), "label must be a nonempty", id="blank-rule-label"),
        pytest.param(first_case(required_patterns={"bad": ""}), "paid_leave.bad must be", id="blank-pattern"),
        pytest.param(first_case(required_patterns={"bad": "["}), "paid_leave.bad: invalid regex", id="invalid-regex"),
        pytest.param(first_case(required_patterns={"bad": ".*"}), "must not match empty text", id="vacuous-rule"),
        pytest.param(
        first_case(forbidden_patterns={"bad": "("}), "paid_leave.bad: invalid regex", id="invalid-forbidden-regex"),
        pytest.param(
        first_case(forbidden_patterns={"conflict": "WORKING DAYS"}), "violates forbidden rule: conflict",
        id="forbidden-rule-ignores-case"),
        pytest.param(first_case(source_fragments=[]), "source_fragments must be a nonempty list", id="no-fragments"),
        pytest.param(
        first_case(source_fragments="23 working days"), "source_fragments must be a nonempty list",
        id="fragments-not-a-list"),
        pytest.param(first_case(source_fragments=[" "]), "source fragment must be", id="blank-fragment"),
        pytest.param(
        first_case(source_fragments=["Invented source text"]),
        "source fragment is absent from policy: Invented source text", id="invented-fragment"),
        pytest.param(
        first_case(reference="No information."), "reference does not satisfy required rule: annual allowance",
        id="reference-misses-rule"),
        pytest.param(
        first_case(forbidden_patterns={"conflict": "23"}), "reference violates forbidden rule: conflict",
        id="reference-hits-forbidden")])
@title("The loader rejects malformed, inconsistent or stale expectations rule by rule [{param_id}]")
def test_loader_rejects_invalid_catalog(tmp_path, corrupt, message):
    data = catalog()
    corrupt(data)

    with pytest.raises(ValueError, match=message):
        load(data, tmp_path)


@pytest.mark.parametrize(("raw", "message"), [
        pytest.param(
        '{"schema_version": 1, "schema_version": 1}', "duplicate JSON field: schema_version", id="duplicate-field"),
        pytest.param("[]", "golden dataset must be an object", id="not-an-object")])
@title("A catalog with duplicate JSON fields or a non-object root is rejected [{param_id}]")
def test_loader_rejects_malformed_json(tmp_path, raw, message):
    path = tmp_path / "golden.json"
    path.write_text(raw)

    with pytest.raises(ValueError, match=message):
        load_golden_dataset(path, POLICY_FILE)


@pytest.mark.parametrize("case_id", ["paid_leave", "leave_approver", "hotel_receipt_condition", "gym_missing"])
@title("A reference answer cited from the policy satisfies its case's answer and source criteria [{param_id}]")
def test_reference_satisfies_acceptance(case_id):
    check_answer(CASES[case_id].reference, case=CASES[case_id])


@pytest.mark.parametrize(("case_id", "invented"), [("gym_missing", " An allowance of KGS 500 is available."),
        ("bonus_missing", " An allowance of KGS 500 is available."),
        ("parental_leave_missing", " Employees receive 30 days.")])
@title("A missing-information answer that invents a benefit violates a forbidden rule [{param_id}]")
def test_missing_policy_rejects_invented_benefit(case_id, invented):
    with pytest.raises(AssertionError, match="forbidden content"):
        check_answer(CASES[case_id].reference + invented, case=CASES[case_id])


@pytest.mark.parametrize(("answer", "reply", "message"), [
        pytest.param("23 working days of paid leave.", {}, "advance notice", id="incomplete-answer"),
        pytest.param(
        PAID_LEAVE.reference, {"source": "23 working days"}, "12 calendar days", id="source-lacks-fragment"),
        pytest.param(PAID_LEAVE.reference, {"document": "other.txt"}, "did not cite", id="other-document"),
        pytest.param(
        "<think>23 working days and 12 calendar days before leave</think>No answer.", {}, "annual allowance",
        id="facts-only-in-thinking")])
@title("Incomplete answers, unsupported sources and facts that appear only in thinking are rejected [{param_id}]")
def test_golden_answer_rejects_bad_evidence(answer, reply, message):
    with pytest.raises(AssertionError, match=message):
        check_answer(answer, **reply)
