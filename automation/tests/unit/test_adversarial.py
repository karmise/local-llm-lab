import json
import shutil

import pytest

from llm_testkit import assertions
from llm_testkit.datasets.adversarial import load_adversarial_cases, materialize_policy
from llm_testkit.reporting.steps import title
from test_support.assertions import errors as errors
from test_support.assertions import pytest_runs
from test_support.assertions import values as value_checks
from test_support.assertions.adversarial import check_poisoned_copy_outcome
from test_support.builders.adversarial import answer, prepare_catalog_case
from test_support.data.adversarial import (
    ASSERTIONS_CASE_CASES,
    ASSERTIONS_CASE_IDS,
    CASES,
    CATALOG_CHANGE_CASES,
    DATA,
    DATASET,
    POISONED_COPY_CASE_CASES,
    POISONED_COPY_CASE_IDS,
)
from test_support.data.scripts.adversarial import COLLECTION_MAKEPYFILE_SOURCE

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("case", ASSERTIONS_CASE_CASES, ids=ASSERTIONS_CASE_IDS)
@title("Adversarial assertions accept grounded answers and reject attack markers [{param_id}]")
def test_assertions(case):
    assertions.assert_adversarial_answer(
        answer(case, case.golden_case.reference), case=case, document_title="policy.txt"
    )
    errors.rejects(
        lambda: assertions.assert_adversarial_answer(
            answer(case, case.golden_case.reference + " TEST_INJECTION_SUCCEEDED"),
            case=case,
            document_title="policy.txt",
        ),
        expected=AssertionError,
        match="attack marker",
    )


@title("Missing-policy adversarial checks reject invented benefits even alongside a refusal")
def test_hallucinated_benefit():
    case = next(c for c in CASES if c.id == "fabricated_benefit")
    errors.rejects(
        lambda: assertions.assert_adversarial_answer(
            answer(case, case.golden_case.reference + " But the company reimburses KGS 9999."),
            case=case,
            document_title="policy.txt",
        ),
        expected=AssertionError,
        match="invented amount",
    )


@pytest.mark.parametrize("case", POISONED_COPY_CASE_CASES, ids=POISONED_COPY_CASE_IDS)
@title(
    "Adversarial policy materialization leaves canonical source and expectations intact [{param_id}]"
)
def test_poisoned_copy(case, tmp_path):
    original = DATA / "company-policy.txt"
    before = original.read_bytes()
    poisoned = materialize_policy(original, case, tmp_path / "copy.txt")
    value_checks.equal(original.read_bytes(), before)
    check_poisoned_copy_outcome(before, case, original, poisoned)


@pytest.mark.parametrize("change", CATALOG_CHANGE_CASES)
@title("Adversarial catalog rejects stale or invalid attack definitions [{param_id}]")
def test_catalog(change, tmp_path):
    data = json.loads((DATA / "adversarial-policy.json").read_text())
    row = data["cases"][0]
    prepare_catalog_case(change, data, row)
    path = tmp_path / "attacks.json"
    path.write_text(json.dumps(data))
    errors.rejects(lambda: load_adversarial_cases(path, DATASET), expected=ValueError)


@title(
    "Adversarial collection skips by default and requires capture only for selected document attacks"
)
def test_collection(framework_pytester):
    shutil.copytree(DATA, framework_pytester.path / "test_data")
    framework_pytester.makeconftest('pytest_plugins = ["llm_testkit.pytest_support.options"]')
    framework_pytester.makepyfile(COLLECTION_MAKEPYFILE_SOURCE)
    pytest_runs.outcomes(
        framework_pytester.runpytest_subprocess("-q", "--rag-model", "test"), skipped=6
    )
    pytest_runs.outcomes(
        framework_pytester.runpytest_subprocess(
            "-q", "--rag-model", "test", "--run-adversarial", "-k", "user_override"
        ),
        passed=1,
        deselected=5,
    )
    result = framework_pytester.runpytest_subprocess(
        "-q", "--rag-model", "test", "--run-adversarial", "-k", "document_instruction"
    )
    value_checks.equal(result.ret, pytest.ExitCode.USAGE_ERROR)
    result.stderr.fnmatch_lines(["*verify actual retrieved attack exposure*"])
    pytest_runs.outcomes(
        framework_pytester.runpytest_subprocess(
            "-q",
            "--rag-model",
            "test",
            "--run-adversarial",
            "-k",
            "document_instruction",
            "--capture-rag",
        ),
        passed=1,
        deselected=5,
    )
