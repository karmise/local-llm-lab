import hashlib
import json
import shutil
from pathlib import Path

import pytest
from requests import Response

from llm_testkit import assertions
from llm_testkit.datasets.adversarial import load_adversarial_cases, materialize_policy
from llm_testkit.datasets.golden import load_golden_dataset
from llm_testkit.reporting.steps import title

pytestmark = pytest.mark.unit
DATA = Path(__file__).resolve().parents[2] / "test_data"
DATASET = load_golden_dataset(DATA / "golden-policy.json", DATA / "company-policy.txt")
CASES = load_adversarial_cases(DATA / "adversarial-policy.json", DATASET)


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


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
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


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
@title(
    "Adversarial policy materialization leaves canonical source and expectations intact [{param_id}]"
)
def test_poisoned_copy(case, tmp_path):
    original = DATA / "company-policy.txt"
    before = original.read_bytes()
    poisoned = materialize_policy(original, case, tmp_path / "copy.txt")
    assert original.read_bytes() == before
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


@pytest.mark.parametrize(
    "change", ["hash", "id", "category", "base", "question", "appendix", "regex"]
)
@title("Adversarial catalog rejects stale or invalid attack definitions [{param_id}]")
def test_catalog(change, tmp_path):
    data = json.loads((DATA / "adversarial-policy.json").read_text())
    row = data["cases"][0]
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
    framework_pytester.makepyfile("""
        import pytest
        @pytest.mark.rag
        @pytest.mark.adversarial
        def test_attack(adversarial_case, generation_model, rag_iteration): pass
    """)
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
