"""JUnit reader: one immutable input, every phase outcome kept and conflicting metadata exposed."""

import hashlib
from pathlib import Path

import pytest

from llm_testkit.reporting.junit import TestCaseResult as CaseResult
from llm_testkit.reporting.junit import read_junit
from llm_testkit.reporting.steps import title

pytestmark = pytest.mark.unit


def report(tmp_path, body: str, **options):
    path = tmp_path / "report.xml"
    path.write_text(f"<testsuites><testsuite>{body}</testsuite></testsuites>")
    return read_junit(path, **options)


def only_case(tmp_path, body: str, **options) -> CaseResult:
    case, = report(tmp_path, body, **options).cases.values()
    return case


def case_xml(name="test_answer", classname="example", *, properties=None, children="") -> str:
    rows = "".join(f'<property name="{key}" value="{value}"/>' for key, value in (properties or {}).items())
    return (
            f'<testcase classname="{classname}" name="{name}">'
            f'<properties>{rows}</properties>{children}</testcase>')


@title("A passing test case is read with its identity and properties")
def test_reads_passing_case(tmp_path):
    result = report(tmp_path, case_xml(properties={"model": "qwen", "run": "1"}))

    assert result.cases == {("example", "test_answer"):
            CaseResult("example", "test_answer", properties={
            "model": "qwen",
            "run": "1"})}
    assert result.cases["example", "test_answer"].status == "passed"


@title("The report checksum covers exactly the bytes that were parsed, read once")
def test_reader_hashes_exactly_the_bytes_it_parses(tmp_path, monkeypatch):
    path = tmp_path / "report.xml"
    raw = b'<testsuite><testcase name="original"/></testsuite>'
    reads = []

    def read_once(self):
        reads.append(self)
        return raw if len(reads) == 1 else b'<testsuite><testcase name="changed"/></testsuite>'

    monkeypatch.setattr(Path, "read_bytes", read_once)

    result = read_junit(path)

    assert result.sha256 == hashlib.sha256(raw).hexdigest()
    assert [case.name for case in result.cases.values()] == ["original"]
    assert reads == [path]


@title("A test case without a classname is read with an empty classname")
def test_missing_classname_is_empty(tmp_path):
    case = only_case(tmp_path, '<testcase name="test_answer"/>')

    assert (case.classname, case.name) == ("", "test_answer")


@pytest.mark.parametrize(("body", "message"), [
        pytest.param("<testcase/>", "JUnit testcase requires a name", id="testcase-without-name"),
        pytest.param(
        '<testcase name="t"><properties><property value="v"/></properties></testcase>',
        "JUnit property requires a name", id="property-without-name"),
        pytest.param(
        '<testcase name="t"><properties><property name="" value="v"/></properties></testcase>',
        "JUnit property requires a name", id="property-with-empty-name")])
@title("Malformed JUnit is rejected with an actionable error [{param_id}]")
def test_malformed_junit_is_rejected(tmp_path, body, message):
    with pytest.raises(ValueError, match=f"^{message}$"):
        report(tmp_path, body)


@title("A property without a value is read as empty text")
def test_property_without_value_is_empty(tmp_path):
    case = only_case(tmp_path, '<testcase name="t"><properties><property name="model"/></properties></testcase>')

    assert case.properties == {"model": ""}


@pytest.mark.parametrize(("tag", "outcome"), [
        pytest.param("failure", "failed", id="failure"),
        pytest.param("skipped", "skipped", id="skipped"),
        pytest.param("error", "error", id="error")])
@title("Each JUnit outcome element is read with its message and text [{param_id}]")
def test_outcome_details(tmp_path, tag, outcome):
    case = only_case(tmp_path, case_xml(children=f'<{tag} message="why">details</{tag}>'))

    assert (case.outcomes, case.status) == ({outcome}, outcome)
    assert case.details == [{"kind": outcome, "message": "why", "text": "details"}]


@title("An outcome element without message or text is read as empty text")
def test_outcome_without_message_or_text(tmp_path):
    case = only_case(tmp_path, case_xml(children="<failure/>"))

    assert case.details == [{"kind": "failed", "message": "", "text": ""}]


@title("Call and teardown entries of one test keep every outcome and its original details")
def test_phase_entries_retain_all_outcomes(tmp_path):
    case = only_case(
            tmp_path,
            case_xml(properties={"model": "qwen"}, children='<failure message="answer">Missing fact</failure>') +
            case_xml(properties={"model": "qwen"}, children='<error message="cleanup">Workspace remains</error>'))

    assert (case.outcomes, case.status, case.conflicts) == ({"failed", "error"}, "error", set())
    assert [(d["message"], d["text"]) for d in case.details] == [("answer", "Missing fact"),
            ("cleanup", "Workspace remains")]


@pytest.mark.parametrize(("outcomes", "status"), [
        pytest.param("<failure/><skipped/>", "skipped", id="skipped-over-failed"),
        pytest.param("<skipped/><error/>", "error", id="error-over-skipped"),
        pytest.param("<failure/><failure/>", "failed", id="repeated-failure")])
@title("The status is the most severe outcome: error, then skipped, then failed [{param_id}]")
def test_status_priority(tmp_path, outcomes, status):
    case = only_case(tmp_path, case_xml(children=outcomes))

    assert case.status == status


@title("Conflicting values for one property are exposed and make the case an error; the first value is kept")
def test_conflicting_properties(tmp_path):
    case = only_case(
            tmp_path,
            case_xml(properties={
            "model": "first",
            "run": "1"}) + case_xml(properties={
            "model": "second",
            "run": "1"}))

    assert (case.conflicts, case.properties, case.status) == ({"model"}, {"model": "first", "run": "1"}, "error")


@title("Repeated properties within one element conflict too")
def test_conflict_within_one_element(tmp_path):
    body = (
            '<testcase name="t"><properties><property name="model" value="a"/>'
            '<property name="model" value="b"/></properties></testcase>')

    assert only_case(tmp_path, body).conflicts == {"model"}


@title("The same test name in different modules is two cases")
def test_identity_includes_classname(tmp_path):
    result = report(tmp_path, case_xml(classname="tests.a") + case_xml(classname="tests.b"))

    assert list(result.cases) == [("tests.a", "test_answer"), ("tests.b", "test_answer")]


@title("An identity property merges entries of one node reported under different names")
def test_identity_property_merges_entries(tmp_path):
    result = report(
            tmp_path,
            case_xml("call", properties={
            "model": "qwen",
            "node": "n1"}, children="<failure/>") +
            case_xml("teardown", properties={
            "model": "qwen",
            "node": "n1"}, children="<error/>") + case_xml("other", properties={
            "model": "qwen",
            "node": "n2"}), identity_property="node")

    assert list(result.cases) == [("", "n1"), ("", "n2")]
    assert (result.cases["", "n1"].name, result.cases["", "n1"].outcomes) == ("call", {"failed", "error"})


@pytest.mark.parametrize("properties", [pytest.param({}, id="missing"), pytest.param({"node": ""}, id="empty")])
@title("Without an identity property value the case is identified by classname and name [{param_id}]")
def test_identity_property_falls_back(tmp_path, properties):
    result = report(tmp_path, case_xml(properties=properties), identity_property="node")

    assert list(result.cases) == [("example", "test_answer")]
