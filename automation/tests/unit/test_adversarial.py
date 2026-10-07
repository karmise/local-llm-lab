import json
import shutil

import pytest

from llm_testkit import assertions
from llm_testkit.datasets.adversarial import load_adversarial_cases, materialize_policy
from llm_testkit.reporting.steps import title
from test_support.builders.adversarial import (
    answer,
    check_poisoned_copy_outcome,
    prepare_catalog_case,
)
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
    with pytest.raises(AssertionError, match="attack marker"):
        assertions.assert_adversarial_answer(
            answer(case, case.golden_case.reference + " TEST_INJECTION_SUCCEEDED"),
            case=case,
            document_title="policy.txt",
        )


@title("Missing-policy adversarial checks reject invented benefits even alongside a refusal")
def test_hallucinated_benefit():
    case = next(c for c in CASES if c.id == "fabricated_benefit")
    with pytest.raises(AssertionError, match="invented amount"):
        assertions.assert_adversarial_answer(
            answer(case, case.golden_case.reference + " But the company reimburses KGS 9999."),
            case=case,
            document_title="policy.txt",
        )


@pytest.mark.parametrize("case", POISONED_COPY_CASE_CASES, ids=POISONED_COPY_CASE_IDS)
@title(
    "Adversarial policy materialization leaves canonical source and expectations intact [{param_id}]"
)
def test_poisoned_copy(case, tmp_path):
    original = DATA / "company-policy.txt"
    before = original.read_bytes()
    poisoned = materialize_policy(original, case, tmp_path / "copy.txt")
    assert original.read_bytes() == before
    check_poisoned_copy_outcome(before, case, original, poisoned)


@pytest.mark.parametrize("change", CATALOG_CHANGE_CASES)
@title("Adversarial catalog rejects stale or invalid attack definitions [{param_id}]")
def test_catalog(change, tmp_path):
    data = json.loads((DATA / "adversarial-policy.json").read_text())
    row = data["cases"][0]
    prepare_catalog_case(change, data, row)
    path = tmp_path / "attacks.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_adversarial_cases(path, DATASET)


@title(
    "Adversarial collection skips by default and requires capture only for selected document attacks"
)
def test_collection(framework_pytester):
    shutil.copytree(DATA, framework_pytester.path / "test_data")
    framework_pytester.makeconftest('pytest_plugins = ["llm_testkit.pytest_support.options"]')
    framework_pytester.makepyfile(COLLECTION_MAKEPYFILE_SOURCE)
    framework_pytester.runpytest_subprocess("-q", "--rag-model", "test").assert_outcomes(skipped=6)
    framework_pytester.runpytest_subprocess(
        "-q", "--rag-model", "test", "--run-adversarial", "-k", "user_override"
    ).assert_outcomes(passed=1, deselected=5)
    result = framework_pytester.runpytest_subprocess(
        "-q", "--rag-model", "test", "--run-adversarial", "-k", "document_instruction"
    )
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*verify actual retrieved attack exposure*"])
    framework_pytester.runpytest_subprocess(
        "-q",
        "--rag-model",
        "test",
        "--run-adversarial",
        "-k",
        "document_instruction",
        "--capture-rag",
    ).assert_outcomes(passed=1, deselected=5)
