import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from llm_testkit import assertions
from llm_testkit.datasets.conversation import ConversationCatalog, load_conversation_catalog
from llm_testkit.reporting.steps import title
from test_support.builders.conversation import DATA, response
from test_support.fixtures.unit_conversation import catalog as catalog

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "case_id", ["greeting", "small_talk", "own_policy", "other_company", "mixed_request"]
)
@title("Reviewed conversation reference satisfies its acceptance rules [{param_id}]")
def test_reviewed_references_satisfy_their_independent_acceptance_rules(
    catalog: ConversationCatalog, case_id: str
) -> None:
    case = next(case for case in catalog.cases if case.id == case_id)
    assertions.assert_conversation_answer(
        response(case.reference), case=case, document_title="policy.txt"
    )


@pytest.mark.parametrize(
    ("case_id", "change", "message"),
    [
        ("greeting", "Hi! Employees get 23 working days of paid leave.", "forbidden content"),
        ("greeting", "Hello! Source: company-policy.txt.", "forbidden content"),
        ("greeting", "Hi! " + "word " * 41, "exceeds 40 words"),
        ("small_talk", "A walk sounds nice; it is sunny.", "invented current conditions"),
        (
            "small_talk",
            "The uploaded documents have no information about that.",
            "missing expected fact",
        ),
        ("small_talk", "<think>A walk sounds nice.</think>Hello!", "missing expected fact"),
        ("own_policy", "append wrong units", "wrong day units"),
        ("other_company", "append invented amount", "invented leave entitlement"),
        ("mixed_request", "remove walk", "missing expected fact"),
        ("mixed_request", "remove unknown", "missing expected fact"),
        ("mixed_request", "append transfer", "transferred entitlement"),
    ],
)
@title("Conversation checks reject irrelevant, invented or incomplete answers [{param_id}]")
def test_conversation_checks_reject_policy_dumps_hallucinations_and_missing_intents(
    catalog: ConversationCatalog, case_id: str, change: str, message: str
) -> None:
    case = next(case for case in catalog.cases if case.id == case_id)
    changes = {
        "append wrong units": case.reference + " Submit it 12 working days before leave.",
        "append invented amount": case.reference + " They get 23 working days.",
        "remove walk": case.reference.replace("A walk can be a nice way to relax.", ""),
        "remove unknown": case.reference.split("I do not have")[0],
        "append transfer": case.reference + " HarborWorks receives 23 working days of leave.",
    }
    with pytest.raises(AssertionError, match=message):
        assertions.assert_conversation_answer(
            response(changes.get(change, change)), case=case, document_title="policy.txt"
        )


@pytest.mark.parametrize("case_id", ["own_policy", "mixed_request"])
@pytest.mark.parametrize("defect", ["wrong document", "unsupported passage"])
@title("Conversational policy facts require the expected supporting source [{param_id}]")
def test_policy_parts_require_the_expected_document_and_supporting_passage(
    catalog: ConversationCatalog, case_id: str, defect: str
) -> None:
    case = next(case for case in catalog.cases if case.id == case_id)
    reply = response(
        case.reference,
        title="other.txt" if defect == "wrong document" else "policy.txt",
        source="Unrelated passage" if defect == "unsupported passage" else None,
    )
    with pytest.raises(AssertionError, match="did not cite|source_text"):
        assertions.assert_conversation_answer(reply, case=case, document_title="policy.txt")


@title("A concise greeting does not imply that retrieval was skipped")
def test_small_talk_sources_are_not_misrepresented_as_evidence_of_skipped_retrieval(
    catalog: ConversationCatalog,
) -> None:
    case = catalog.cases[0]
    assertions.assert_conversation_answer(
        response(case.reference), case=case, document_title="policy.txt"
    )


@pytest.mark.parametrize("case_id", ["small_talk", "mixed_request"])
@title("A walking suggestion may use natural paraphrases [{param_id}]")
def test_walking_intent_accepts_fresh_air_without_requiring_literal_walk(
    catalog: ConversationCatalog, case_id: str
) -> None:
    case = next(case for case in catalog.cases if case.id == case_id)
    suggestion = "That sounds like a lovely idea; enjoy the fresh air and nature!"
    answer = (
        suggestion
        if case_id == "small_talk"
        else case.reference.replace("A walk can be a nice way to relax.", suggestion)
    )
    assertions.assert_conversation_answer(response(answer), case=case, document_title="policy.txt")


@title("Company scope remains valid when an inline source filename precedes its facts")
def test_inline_filename_does_not_break_own_company_fact_scope(
    catalog: ConversationCatalog,
) -> None:
    case = next(case for case in catalog.cases if case.id == "mixed_request")
    answer = (
        "Hi! Enjoy the fresh air. At Northern Lighthouse, according to company-policy.txt, "
        "employees get 23 working days of paid leave per year. "
        "I have no policy information about HarborWorks."
    )
    assertions.assert_conversation_answer(response(answer), case=case, document_title="policy.txt")


@title("Another company's entitlement cannot satisfy the Northern Lighthouse allowance")
def test_own_company_scope_cannot_consume_another_company_fact(
    catalog: ConversationCatalog,
) -> None:
    case = next(case for case in catalog.cases if case.id == "mixed_request")
    answer = (
        "Hi! Enjoy your walk. Northern Lighthouse: see company-policy.txt; "
        "HarborWorks employees get 23 working days per year. I have no information about HarborWorks."
    )
    with pytest.raises(AssertionError, match="own company allowance"):
        assertions.assert_conversation_answer(
            response(answer), case=case, document_title="policy.txt"
        )


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        ("checksum", "checksum mismatch"),
        ("duplicate id", "duplicate conversation case id"),
        ("word limit type", "max_words"),
        ("absent source", "absent from policy"),
        ("missing reference rule", "reference is missing"),
        ("forbidden reference", "reference violates"),
    ],
)
@title("Invalid conversation acceptance catalog fails before generation [{param_id}]")
def test_invalid_catalogs_fail_before_any_generation(
    tmp_path: Path, defect: str, message: str
) -> None:
    data = json.loads((DATA / "conversation-policy.json").read_text())
    row = data["cases"][0]
    if defect == "checksum":
        data["policy_sha256"] = "0" * 64
    elif defect == "duplicate id":
        data["cases"][1]["id"] = row["id"]
    elif defect == "word limit type":
        row["max_words"] = True
    elif defect == "absent source":
        row["source_fragments"] = ["Not in the policy"]
    elif defect == "missing reference rule":
        row["reference"] = "Ready to help."
    else:
        row["reference"] = "Hi! You have 23 working days of paid leave."
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=message):
        load_conversation_catalog(path, DATA / "company-policy.txt")


@title("Conversation fixture sends an explicit Chat-mode request")
def test_chat_request_explicitly_uses_chat_mode_without_changing_query_client_default() -> None:
    from unittest.mock import Mock

    from llm_testkit.config import Settings
    from test_support.fixtures.conversation import conversation_chat

    api = Mock()
    chat = conversation_chat.__wrapped__(
        None, {"chatMode": "chat"}, api, {"slug": "temporary"}, Settings()
    )
    result = chat("Hello!")
    assert result is api.chat.return_value
    api.chat.assert_called_once_with("temporary", "Hello!", mode="chat", timeout=300)


@title("Conversation fixture rejects a document-only Query profile")
def test_conversation_fixture_rejects_a_query_profile() -> None:
    from unittest.mock import Mock

    from llm_testkit.config import Settings
    from test_support.fixtures.conversation import conversation_chat

    with pytest.raises(AssertionError, match="chatMode"):
        conversation_chat.__wrapped__(
            None, {"chatMode": "query"}, Mock(), {"slug": "temporary"}, Settings()
        )


@title("Conversation selection honors opt-in, model budget and independent repetitions")
def test_conversation_collection_defaults_to_one_model_and_accepts_explicit_matrix(
    framework_pytester: pytest.Pytester,
) -> None:
    runner = framework_pytester
    shutil.copytree(DATA, runner.path / "test_data")
    runner.makeconftest('pytest_plugins = ["llm_testkit.pytest_support.options"]')
    runner.makepyfile("""
        import pytest
        @pytest.mark.rag
        @pytest.mark.conversation
        def test_conversation(conversation_case, generation_model, rag_iteration):
            assert conversation_case.id == 'greeting'
            assert generation_model in {'qwen3.5:4b', 'model-a', 'model-b'}
            assert rag_iteration in {1, 2}
    """)
    runner.runpytest_subprocess("--run-conversation", "-k", "greeting", "-q").assert_outcomes(
        passed=1, deselected=4
    )
    runner.runpytest_subprocess(
        "--run-conversation",
        "--rag-model",
        "model-a",
        "--rag-model",
        "model-b",
        "--rag-repeat",
        "2",
        "-k",
        "greeting",
        "-q",
    ).assert_outcomes(passed=4, deselected=16)
    result = runner.runpytest_subprocess("--run-conversation", "--capture-rag", "-k", "greeting")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*small talk is not a RAG sample*"])


@title("Conversation fixture rejects expectations changed after collection")
def test_catalog_change_after_collection_is_not_accepted_as_the_original_case(
    catalog: ConversationCatalog,
) -> None:
    from test_support.fixtures.conversation import conversation_metadata

    changed_case = replace(catalog.cases[0], question="Different question")
    with pytest.raises(pytest.fail.Exception, match="changed after collection"):
        conversation_metadata.__wrapped__(
            DATA.parent, DATA / "company-policy.txt", changed_case, lambda *_: None
        )
