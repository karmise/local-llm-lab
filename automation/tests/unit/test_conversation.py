"""Conversational assistant: a reviewed acceptance catalog, answer criteria, Chat-mode fixtures and opt-in selection."""

import hashlib
import json
import shutil
from dataclasses import replace
from unittest.mock import Mock

import pytest

from llm_testkit import assertions
from llm_testkit.config import Settings
from llm_testkit.datasets.conversation import load_conversation_catalog
from llm_testkit.reporting.steps import title
from test_support.builders.anythingllm import chat_reply
from test_support.builders.conversation import CONVERSATION_FILE, catalog_json, conversation_cases
from test_support.builders.golden import POLICY_FILE, TEST_DATA
from test_support.builders.identities import POLICY_DOCUMENT_TITLE
from test_support.fixtures.conversation import conversation_chat, conversation_metadata
from test_support.paths import AUTOMATION_ROOT

pytestmark = pytest.mark.unit

CASE_IDS = ["greeting", "small_talk", "own_policy", "other_company", "mixed_request"]
WALK = "A walk can be a nice way to relax."


def load(data: dict, tmp_path):
    path = tmp_path / "conversation.json"
    path.write_text(json.dumps(data))
    return load_conversation_catalog(path, POLICY_FILE)


def check_answer(case_id: str, answer: str, **reply) -> None:
    case = conversation_cases()[case_id]
    assertions.assert_conversation_answer(chat_reply(answer, **reply), case=case, document_title=POLICY_DOCUMENT_TITLE)


@title("The reviewed catalog loads five conversation cases bound to the policy, with every field")
def test_catalog_loads_cases():
    raw, data = CONVERSATION_FILE.read_bytes(), catalog_json()

    catalog = load_conversation_catalog(CONVERSATION_FILE, POLICY_FILE)

    assert [case.id for case in catalog.cases] == CASE_IDS
    assert (catalog.version, catalog.sha256) == (data["version"], hashlib.sha256(raw).hexdigest())
    assert catalog.policy_sha256 == hashlib.sha256(POLICY_FILE.read_bytes()).hexdigest()
    for case, row in zip(catalog.cases, data["cases"], strict=True):
        assert (case.title, case.question, case.reference,
                case.rationale) == (row["title"], row["question"], row["reference"], row["rationale"])
        assert (case.required_patterns, case.forbidden_patterns) == (
                tuple(row["required_patterns"].items()), tuple(row["forbidden_patterns"].items()))
        assert (case.source_fragments, case.max_words) == (tuple(row["source_fragments"]), row["max_words"])


def greeting(**fields):
    return lambda data: data["cases"][0].update(fields)


@pytest.mark.parametrize(("corrupt", "message"), [
        pytest.param(lambda d: d.update(schema_version=2), "Unsupported conversation catalog schema", id="schema"),
        pytest.param(lambda d: d.pop("version"), "Conversation catalog version must be", id="no-version"),
        pytest.param(lambda d: d.update(policy_sha256="0" * 64), "policy checksum mismatch", id="policy-checksum"),
        pytest.param(lambda d: d.update(cases=[]), "must contain cases", id="no-cases"),
        pytest.param(lambda d: d.update(cases={}), "must contain cases", id="cases-not-a-list"),
        pytest.param(
        lambda d: d["cases"].__setitem__(0, "greeting"), "Conversation case must be an object", id="case-not-object"),
        pytest.param(greeting(id=""), "Conversation case id must be", id="blank-id"),
        pytest.param(greeting(id="Greeting"), "Invalid or duplicate conversation case id: Greeting", id="invalid-id"),
        pytest.param(
        lambda d: d["cases"][1].update(id="greeting"), "duplicate conversation case id: greeting", id="duplicate-id"),
        *[
        pytest.param(greeting(**{name: " "}), f"greeting: {name} must be", id=f"blank-{name}")
        for name in ("title", "question", "reference", "rationale")],
        pytest.param(greeting(required_patterns={}), "must be a nonempty object", id="no-required-rules"),
        pytest.param(greeting(forbidden_patterns={"bad": "["}), "greeting.bad: invalid regex", id="invalid-forbidden"),
        pytest.param(greeting(required_patterns={"bad": "["}), "greeting.bad: invalid regex", id="invalid-required"),
        pytest.param(greeting(max_words=True), "max_words must be an integer from 1 to 500", id="boolean-word-limit"),
        pytest.param(greeting(max_words=0), "max_words must be an integer from 1 to 500", id="zero-word-limit"),
        pytest.param(greeting(max_words=501), "max_words must be an integer from 1 to 500", id="word-limit-too-high"),
        pytest.param(greeting(source_fragments="23"), "source_fragments must be a list", id="fragments-not-a-list"),
        pytest.param(greeting(source_fragments=[" "]), "greeting: source fragment must be", id="blank-fragment"),
        pytest.param(
        greeting(source_fragments=["Not in the policy"]), "source fragment is absent from policy",
        id="invented-fragment"),
        pytest.param(greeting(max_words=1), "reference exceeds max_words", id="reference-too-long"),
        pytest.param(greeting(reference="Ready to help."), "reference is missing required rule", id="missing-rule"),
        pytest.param(
        greeting(reference="Hi! You have 23 working days of paid leave."), "reference violates forbidden rule",
        id="forbidden-reference")])
@title("An invalid conversation catalog fails rule by rule before any generation [{param_id}]")
def test_catalog_rejects_invalid_case(tmp_path, corrupt, message):
    data = catalog_json()
    corrupt(data)

    with pytest.raises(ValueError, match=message):
        load(data, tmp_path)


@pytest.mark.parametrize(("max_words", "reference"), [(1, "Hi!\n\n  "), (500, "Hi " * 500)],
        ids=["one", "five-hundred"])
@title("Word limits from 1 to 500 are accepted, counting words after collapsing whitespace [{param_id}]")
def test_catalog_accepts_word_limit_bounds(tmp_path, max_words, reference):
    data = catalog_json()
    data["cases"][0].update(max_words=max_words, reference=reference)

    assert load(data, tmp_path).cases[0].max_words == max_words


@title("A case may have no forbidden rules")
def test_catalog_accepts_case_without_forbidden_rules(tmp_path):
    data = catalog_json()
    data["cases"][0]["forbidden_patterns"] = {}

    assert load(data, tmp_path).cases[0].forbidden_patterns == ()


@title("A case without source fragments is accepted, since small talk needs no citation")
def test_catalog_accepts_case_without_sources(tmp_path):
    data = catalog_json()
    data["cases"][0]["source_fragments"] = []

    assert load(data, tmp_path).cases[0].source_fragments == ()


@title("Required rules match references regardless of letter case")
def test_catalog_required_rules_ignore_case(tmp_path):
    data = catalog_json()
    data["cases"][0]["reference"] = data["cases"][0]["reference"].upper()

    assert load(data, tmp_path).cases[0].reference.isupper()


@title("Forbidden rules match references regardless of letter case")
def test_catalog_forbidden_rules_ignore_case(tmp_path):
    data = catalog_json()
    data["cases"][0]["forbidden_patterns"] = {"shouting": "HELLO|HI"}

    with pytest.raises(ValueError, match="violates forbidden rule: shouting"):
        load(data, tmp_path)


@pytest.mark.parametrize("case_id", CASE_IDS)
@title("Each reviewed reference satisfies its own acceptance rules [{param_id}]")
def test_reference_satisfies_acceptance(case_id):
    check_answer(case_id, conversation_cases()[case_id].reference)


@pytest.mark.parametrize(("case_id", "answer", "message"), [
        pytest.param(
        "greeting", "Hi! Employees get 23 working days of paid leave.", "forbidden content",
        id="greeting-with-policy-dump"),
        pytest.param("greeting", "Hello! Source: company-policy.txt.", "forbidden content", id="greeting-with-source"),
        pytest.param("greeting", "Hi! " + "word " * 41, "answer exceeds 40 words", id="greeting-too-long"),
        pytest.param(
        "small_talk", "A walk sounds nice; it is sunny.", "invented current conditions", id="invented-weather"),
        pytest.param(
        "small_talk", "The uploaded documents have no information about that.", "missing expected fact",
        id="refusal-instead-of-chat"),
        pytest.param(
        "small_talk", "<think>A walk sounds nice.</think>Hello!", "missing expected fact", id="intent-only-in-thinking")
])
@title("Conversation checks reject policy dumps, invented conditions and missing intents [{param_id}]")
def test_conversation_rejects_answer(case_id, answer, message):
    with pytest.raises(AssertionError, match=message):
        check_answer(case_id, answer)


@pytest.mark.parametrize(("case_id", "change", "message"), [
        pytest.param(
        "own_policy", lambda r: r + " Submit it 12 working days before leave.", "wrong day units", id="wrong-units"),
        pytest.param(
        "other_company", lambda r: r + " They get 23 working days.", "invented leave entitlement",
        id="invented-entitlement"),
        pytest.param(
        "mixed_request", lambda r: r.replace(WALK, ""), "missing expected fact: walking", id="walking-intent-dropped"),
        pytest.param(
        "mixed_request", lambda r: r.split("I do not have")[0], "other company limitation",
        id="unknown-company-dropped"),
        pytest.param(
        "mixed_request", lambda r: r + " HarborWorks receives 23 working days of leave.", "transferred entitlement",
        id="entitlement-transferred")])
@title("Changing a reviewed reference in a disallowed way makes it fail [{param_id}]")
def test_conversation_rejects_changed_reference(case_id, change, message):
    with pytest.raises(AssertionError, match=message):
        check_answer(case_id, change(conversation_cases()[case_id].reference))


@pytest.mark.parametrize("case_id", ["own_policy", "mixed_request"])
@pytest.mark.parametrize(("reply", "message"), [
        pytest.param({"document": "other.txt"}, "did not cite the uploaded document", id="wrong-document"),
        pytest.param({"source": "Unrelated passage"}, "source_text", id="unsupported-passage")])
@title("Policy facts in a conversation require the expected supporting source [{param_id}]")
def test_policy_facts_require_source(case_id, reply, message):
    with pytest.raises(AssertionError, match=message):
        check_answer(case_id, conversation_cases()[case_id].reference, **reply)


@title("Small talk needs no citation, so any cited document is accepted")
def test_small_talk_ignores_sources():
    check_answer("greeting", conversation_cases()["greeting"].reference, document="other.txt", source="Unrelated")


@pytest.mark.parametrize("case_id", ["small_talk", "mixed_request"])
@title("A walking suggestion may use natural paraphrases such as fresh air [{param_id}]")
def test_walking_intent_accepts_paraphrase(case_id):
    suggestion = "That sounds like a lovely idea; enjoy the fresh air and nature!"
    reference = conversation_cases()[case_id].reference

    check_answer(case_id, suggestion if case_id == "small_talk" else reference.replace(WALK, suggestion))


@title("Company scope remains valid when an inline source filename precedes its facts")
def test_inline_filename_keeps_company_scope():
    check_answer(
            "mixed_request", "Hi! Enjoy the fresh air. At Northern Lighthouse, according to company-policy.txt, "
            "employees get 23 working days of paid leave per year. I have no policy information about HarborWorks.")


@title("Another company's entitlement cannot satisfy the Northern Lighthouse allowance")
def test_other_company_fact_cannot_satisfy_own_scope():
    with pytest.raises(AssertionError, match="own company allowance"):
        check_answer(
                "mixed_request", "Hi! Enjoy your walk. Northern Lighthouse: see company-policy.txt; "
                "HarborWorks employees get 23 working days per year. I have no information about HarborWorks.")


@title("The conversation fixture sends an explicit Chat-mode request with the LLM timeout")
def test_chat_fixture_uses_chat_mode():
    api = Mock()
    chat = conversation_chat.__wrapped__(None, {"chatMode": "chat"}, api, {"slug": "temporary"}, Settings())

    assert chat("Hello!") is api.chat.return_value
    api.chat.assert_called_once_with("temporary", "Hello!", mode="chat", timeout=Settings().llm_timeout)


@title("The conversation fixture rejects a document-only Query profile")
def test_chat_fixture_rejects_query_profile():
    with pytest.raises(AssertionError, match="chatMode"):
        conversation_chat.__wrapped__(None, {"chatMode": "query"}, Mock(), {"slug": "temporary"}, Settings())


@title("Conversation metadata rejects expectations changed after collection")
def test_metadata_rejects_changed_case():
    changed = replace(conversation_cases()["greeting"], question="Different question")

    with pytest.raises(pytest.fail.Exception, match="changed after collection"):
        conversation_metadata.__wrapped__(AUTOMATION_ROOT, POLICY_FILE, changed, lambda *_: None)


@pytest.fixture
def conversation_suite(framework_pytester):
    shutil.copytree(TEST_DATA, framework_pytester.path / "test_data")
    framework_pytester.makeconftest('pytest_plugins = ["llm_testkit.pytest_support.options"]')
    framework_pytester.makepyfile(
            """
        import pytest
        @pytest.mark.rag
        @pytest.mark.conversation
        def test_conversation(conversation_case, generation_model, rag_iteration):
            assert conversation_case.id == 'greeting'
            assert generation_model in {'qwen3.5:4b', 'model-a', 'model-b'}
            assert rag_iteration in {1, 2}
    """)
    return lambda *arguments: framework_pytester.runpytest_subprocess("--run-conversation", *arguments)


@title("Conversation scenarios run on one default model unless a model matrix and repetitions are requested")
def test_selection_defaults_to_one_model(conversation_suite):
    conversation_suite("-k", "greeting", "-q").assert_outcomes(passed=1, deselected=4)
    conversation_suite("--rag-model", "model-a", "--rag-model", "model-b", "--rag-repeat", "2", "-k", "greeting",
            "-q").assert_outcomes(passed=4, deselected=16)


@title("Conversation scenarios refuse context capture, because small talk is not a RAG sample")
def test_selection_rejects_capture(conversation_suite):
    result = conversation_suite("--capture-rag", "-k", "greeting")

    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*small talk is not a RAG sample*"])


@title("Conversation scenarios skip unless explicitly enabled")
def test_selection_is_opt_in(framework_pytester, conversation_suite):
    framework_pytester.runpytest_subprocess("-k", "greeting", "-q").assert_outcomes(skipped=1, deselected=4)
