import hashlib
from pathlib import Path

import pytest

from llm_testkit.reporting.junit import read_junit
from test_support.assertions import errors as errors
from test_support.assertions import values as value_checks
from test_support.builders.junit import make_read_once_stub
from test_support.data.junit import MALFORMED_JUNIT_HAS_ACTIONABLE_ERROR_BODY_CASES

pytestmark = pytest.mark.unit


def test_phase_entries_retain_all_outcomes_and_original_details(tmp_path):
    path = tmp_path / "phases.xml"
    path.write_text(
            """<testsuites><testsuite>
        <testcase classname="example" name="test_answer">
          <properties><property name="model" value="first"/></properties>
          <failure message="answer">Missing fact</failure>
        </testcase>
        <testcase classname="example" name="test_answer">
          <properties><property name="model" value="second"/></properties>
          <error message="cleanup">Workspace still exists</error>
        </testcase>
      </testsuite></testsuites>""")
    report = read_junit(path)
    value_checks.equal(report.sha256, hashlib.sha256(path.read_bytes()).hexdigest())
    value_checks.length(report.cases, 1)
    case = next(iter(report.cases.values()))
    value_checks.equal(case.outcomes, {"failed", "error"})
    value_checks.equal(case.status, "error")
    value_checks.equal(case.conflicts, {"model"})
    value_checks.equal([d["text"] for d in case.details], ["Missing fact", "Workspace still exists"])


@pytest.mark.parametrize("body", MALFORMED_JUNIT_HAS_ACTIONABLE_ERROR_BODY_CASES)
def test_malformed_junit_has_actionable_error(tmp_path, body):
    path = tmp_path / "invalid.xml"
    path.write_text("<testsuite>" + body + "</testsuite>")
    errors.rejects(lambda: read_junit(path), expected=ValueError, match="requires a name")


def test_reader_hashes_exactly_the_bytes_it_parses(tmp_path, monkeypatch):
    path = tmp_path / "report.xml"
    raw = b'<testsuite><testcase name="original"/></testsuite>'
    reads = []

    read_once = make_read_once_stub(path, raw, reads)

    monkeypatch.setattr(Path, "read_bytes", read_once)
    report = read_junit(path)
    value_checks.equal(report.sha256, hashlib.sha256(raw).hexdigest())
    value_checks.equal(next(iter(report.cases.values())).name, "original")
    value_checks.equal(reads, [1])
