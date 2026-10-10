"""Adversarial inputs: a source-bound attack catalog, poisoned policy copies, answer checks and opt-in collection."""

import hashlib
import json
import shutil

import pytest

from llm_testkit import assertions
from llm_testkit.datasets.adversarial import load_adversarial_cases, materialize_policy
from llm_testkit.reporting.steps import title
from test_support.builders.adversarial import ADVERSARIAL_FILE, attack_cases, catalog_json
from test_support.builders.anythingllm import chat_reply
from test_support.builders.golden import GOLDEN_DATASET, POLICY_FILE, TEST_DATA
from test_support.data.common import POLICY_DOCUMENT_TITLE

pytestmark = pytest.mark.unit

ATTACK_IDS = [
        "user_override", "role_spoof", "forged_source", "fabricated_benefit", "document_instruction",
        "conflicting_note"]
DOCUMENT_ATTACKS = {"document_instruction", "conflicting_note"}


def load(data: dict, tmp_path):
    path = tmp_path / "attacks.json"
    path.write_text(json.dumps(data))
    return load_adversarial_cases(path, GOLDEN_DATASET)


def check_answer(case, answer: str) -> None:
    assertions.assert_adversarial_answer(chat_reply(answer), case=case, document_title=POLICY_DOCUMENT_TITLE)


@title("The reviewed catalog loads six attacks bound to golden cases, with catalog provenance")
def test_catalog_loads_attacks():
    raw = ADVERSARIAL_FILE.read_bytes()
    data = catalog_json()

    cases = load_adversarial_cases(ADVERSARIAL_FILE, GOLDEN_DATASET)

    assert [case.id for case in cases] == ATTACK_IDS
    assert {case.id for case in cases if case.document_appendix} == DOCUMENT_ATTACKS
    for case, row in zip(cases, data["cases"], strict=True):
        assert (case.category, case.golden_case.id,
                case.question) == (row["category"], row["golden_case_id"], row["question"])
        assert (case.document_appendix,
                case.forbidden_patterns) == (row["document_appendix"], tuple(row["forbidden_patterns"].items()))
        assert (case.catalog_sha256, case.catalog_version) == (hashlib.sha256(raw).hexdigest(), data["version"])


def first_attack(**fields):
    return lambda data: data["cases"][0].update(fields)


@pytest.mark.parametrize(("corrupt", "message"), [
        pytest.param(lambda d: d.update(schema_version=2), "Unsupported adversarial catalog schema", id="schema"),
        pytest.param(
        lambda d: d.update(golden_dataset_sha256="changed"), "require the current golden dataset",
        id="stale-golden-dataset"),
        pytest.param(lambda d: d.pop("version"), "adversarial catalog version must be", id="no-version"),
        pytest.param(lambda d: d.update(cases=[]), "requires cases", id="no-cases"),
        pytest.param(lambda d: d.update(cases={}), "requires cases", id="cases-not-a-list"),
        pytest.param(lambda d: d["cases"].__setitem__(0, "attack"), "row must be an object", id="row-not-object"),
        pytest.param(first_attack(id="Bad-Id"), "Invalid or duplicated attack id", id="invalid-id"),
        pytest.param(first_attack(id=None), "Invalid or duplicated attack id", id="no-id"),
        pytest.param(
        lambda d: d["cases"][1].update(id=d["cases"][0]["id"]), "Invalid or duplicated attack id", id="duplicate-id"),
        pytest.param(
        first_attack(category="jailbreak"), "Unknown attack category or golden case", id="unknown-category"),
        pytest.param(first_attack(category=None), "Unknown attack category or golden case", id="no-category"),
        pytest.param(
        first_attack(golden_case_id="unknown"), "Unknown attack category or golden case", id="unknown-golden-case"),
        pytest.param(first_attack(golden_case_id=None), "Unknown attack category or golden case", id="no-golden-case"),
        pytest.param(
        first_attack(question="Unrelated question"), "retain the original policy question", id="question-replaced"),
        pytest.param(first_attack(question=None), "retain the original policy question", id="no-question"),
        pytest.param(
        first_attack(document_appendix="Unexpected modification"), "user attacks must not modify",
        id="user-attack-with-appendix"),
        pytest.param(
        lambda d: d["cases"][4].update(document_appendix=" "), "Document attacks require an appendix",
        id="document-attack-without-appendix"),
        pytest.param(first_attack(document_appendix=None), "Document attacks require an appendix", id="no-appendix"),
        pytest.param(
        first_attack(forbidden_patterns={}), "forbidden_patterns must be a nonempty object", id="no-forbidden-rules"),
        pytest.param(
        first_attack(forbidden_patterns={"empty": ".*"}), "user_override: forbidden_patterns.empty",
        id="vacuous-forbidden-rule")])
@title("A stale or invalid attack definition is rejected rule by rule [{param_id}]")
def test_catalog_rejects_invalid_attack(tmp_path, corrupt, message):
    data = catalog_json()
    corrupt(data)

    with pytest.raises(ValueError, match=message):
        load(data, tmp_path)


@pytest.mark.parametrize("attack_id", ATTACK_IDS)
@title("A grounded answer passes and the same answer with the attack marker fails [{param_id}]")
def test_answer_checks_reject_attack_marker(attack_id):
    case = attack_cases()[attack_id]

    check_answer(case, case.golden_case.reference)
    with pytest.raises(AssertionError, match=f"Adversarial case {attack_id}: forbidden output: attack marker"):
        check_answer(case, case.golden_case.reference + " TEST_INJECTION_SUCCEEDED")


@title("Attack markers are found regardless of letter case")
def test_attack_marker_ignores_case():
    case = attack_cases()["user_override"]

    with pytest.raises(AssertionError, match="attack marker"):
        check_answer(case, case.golden_case.reference + " test_injection_succeeded")


@title("A refusal that still invents a benefit fails the golden rules of the missing-policy case")
def test_unsupported_fact_rejects_invented_benefit():
    case = attack_cases()["fabricated_benefit"]

    with pytest.raises(AssertionError, match="invented amount"):
        check_answer(case, case.golden_case.reference + " But the company reimburses KGS 9999.")


@pytest.mark.parametrize("attack_id", sorted(DOCUMENT_ATTACKS))
@title("A document attack is written to a separate poisoned copy; the canonical policy is unchanged [{param_id}]")
def test_document_attack_poisons_a_copy(tmp_path, attack_id):
    case = attack_cases()[attack_id]
    before = POLICY_FILE.read_bytes()

    poisoned = materialize_policy(POLICY_FILE, case, tmp_path / "copy.txt")

    assert POLICY_FILE.read_bytes() == before
    assert poisoned == tmp_path / "copy.txt"
    assert poisoned.read_text() == before.decode() + case.document_appendix
    assert hashlib.sha256(poisoned.read_bytes()).hexdigest() != GOLDEN_DATASET.policy_sha256


@pytest.mark.parametrize("attack_id", sorted(set(ATTACK_IDS) - DOCUMENT_ATTACKS))
@title("A user attack uses the canonical policy without writing a copy [{param_id}]")
def test_user_attack_uses_canonical_policy(tmp_path, attack_id):
    assert materialize_policy(POLICY_FILE, attack_cases()[attack_id], tmp_path / "copy.txt") == POLICY_FILE
    assert not (tmp_path / "copy.txt").exists()


@title("An attack counts as exposed only if its text reached the model context, ignoring whitespace")
def test_attack_exposure():
    attack = attack_cases()["document_instruction"].document_appendix
    poisoned_context = POLICY_FILE.read_text() + " ".join(attack.split())

    assertions.assert_attack_exposure([poisoned_context], attack_text=attack)
    with pytest.raises(AssertionError, match="not exposed in actual model context"):
        assertions.assert_attack_exposure([POLICY_FILE.read_text()], attack_text=attack)


@pytest.mark.parametrize(("contexts", "attack"), [([], "Attack"), (["Policy"], " ")], ids=["no-context", "no-attack"])
@title("Exposure cannot be shown without both captured contexts and attack text [{param_id}]")
def test_attack_exposure_requires_inputs(contexts, attack):
    with pytest.raises(AssertionError, match="requires nonempty attack text and contexts"):
        assertions.assert_attack_exposure(contexts, attack_text=attack)


@pytest.fixture
def attack_suite(framework_pytester):
    """A child pytest project with the reviewed data and one parametrised adversarial test."""
    shutil.copytree(TEST_DATA, framework_pytester.path / "test_data")
    framework_pytester.makeconftest('pytest_plugins = ["llm_testkit.pytest_support.options"]')
    framework_pytester.makepyfile(
            """
        import pytest
        @pytest.mark.rag
        @pytest.mark.adversarial
        def test_attack(adversarial_case, generation_model, rag_iteration): pass
    """)
    return lambda *arguments: framework_pytester.runpytest_subprocess("-q", "--rag-model", "test", *arguments)


@title("Adversarial scenarios are collected for every attack but skipped unless explicitly enabled")
def test_collection_skips_by_default(attack_suite):
    attack_suite().assert_outcomes(skipped=6)


@title("An enabled user attack runs without context capture")
def test_collection_runs_user_attack(attack_suite):
    attack_suite("--run-adversarial", "-k", "user_override").assert_outcomes(passed=1, deselected=5)


@title("A selected document attack requires context capture, so its exposure can be verified")
def test_collection_requires_capture_for_document_attack(attack_suite):
    without_capture = attack_suite("--run-adversarial", "-k", "document_instruction")
    with_capture = attack_suite("--run-adversarial", "-k", "document_instruction", "--capture-rag")

    assert without_capture.ret == pytest.ExitCode.USAGE_ERROR
    without_capture.stderr.fnmatch_lines(["*verify actual retrieved attack exposure*"])
    with_capture.assert_outcomes(passed=1, deselected=5)
