import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from llm_testkit import assertions
from llm_testkit.datasets.conversation import ConversationCatalog, load_conversation_catalog
from llm_testkit.reporting.steps import title
from test_support.assertions import errors as errors
from test_support.assertions import mocks as mock_checks
from test_support.assertions import pytest_runs
from test_support.assertions import values as value_checks
from test_support.builders.conversation import (
        make_defective_policy_source, make_invalid_conversation_answers, make_walking_response,
        prepare_invalid_catalogs_fail_before_any_generation_case, response)
from test_support.data import common as case_data
from test_support.data.conversation import (
        CHAT_MODE_CONFIGURATION,
        CONVERSATION_CHECKS_REJECT_POLICY_DUMPS_HALLUCINATIONS_AND_MISSING_INTENTS_CASE_ID_CHANGE_MESSAGE_CASES, DATA,
        INVALID_CATALOGS_FAIL_BEFORE_ANY_GENERATION_DEFECT_MESSAGE_CASES,
        POLICY_PARTS_REQUIRE_THE_EXPECTED_DOCUMENT_AND_SUPPORTING_PASSAGE_CASE_ID_CASES,
        POLICY_PARTS_REQUIRE_THE_EXPECTED_DOCUMENT_AND_SUPPORTING_PASSAGE_DEFECT_CASES, QUERY_MODE_CONFIGURATION,
        REVIEWED_REFERENCES_SATISFY_THEIR_INDEPENDENT_ACCEPTANCE_RULES_CASE_ID_CASES, TEMPORARY_WORKSPACE,
        WALKING_INTENT_ACCEPTS_FRESH_AIR_WITHOUT_REQUIRING_LITERAL_WALK_CASE_ID_CASES)
from test_support.data.scripts.conversation import (
        CONVERSATION_COLLECTION_DEFAULTS_TO_ONE_MODEL_AND_ACCEPTS_EXPLICIT_MATRIX_MAKEPYFILE_SOURCE)
from test_support.fixtures.conversation import conversation_chat, conversation_metadata
from test_support.fixtures.unit_conversation import catalog as catalog

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("case_id", REVIEWED_REFERENCES_SATISFY_THEIR_INDEPENDENT_ACCEPTANCE_RULES_CASE_ID_CASES)
@title("Reviewed conversation reference satisfies its acceptance rules [{param_id}]")
def test_reviewed_references_satisfy_their_independent_acceptance_rules(
        catalog: ConversationCatalog, case_id: str) -> None:
    case = next(case for case in catalog.cases if case.id == case_id)
    assertions.assert_conversation_answer(
            response(case.reference), case=case, document_title=case_data.POLICY_DOCUMENT_TITLE)


@pytest.mark.parametrize(("case_id", "change", "message"),
        CONVERSATION_CHECKS_REJECT_POLICY_DUMPS_HALLUCINATIONS_AND_MISSING_INTENTS_CASE_ID_CHANGE_MESSAGE_CASES)
@title("Conversation checks reject irrelevant, invented or incomplete answers [{param_id}]")
def test_conversation_checks_reject_policy_dumps_hallucinations_and_missing_intents(
        catalog: ConversationCatalog, case_id: str, change: str, message: str) -> None:
    case = next(case for case in catalog.cases if case.id == case_id)
    changes = make_invalid_conversation_answers(case)
    errors.rejects(
            lambda: assertions.assert_conversation_answer(
            response(changes.get(change, change)), case=case, document_title=case_data.POLICY_DOCUMENT_TITLE),
            expected=AssertionError, match=message)


@pytest.mark.parametrize("case_id", POLICY_PARTS_REQUIRE_THE_EXPECTED_DOCUMENT_AND_SUPPORTING_PASSAGE_CASE_ID_CASES)
@pytest.mark.parametrize("defect", POLICY_PARTS_REQUIRE_THE_EXPECTED_DOCUMENT_AND_SUPPORTING_PASSAGE_DEFECT_CASES)
@title("Conversational policy facts require the expected supporting source [{param_id}]")
def test_policy_parts_require_the_expected_document_and_supporting_passage(
        catalog: ConversationCatalog, case_id: str, defect: str) -> None:
    case = next(case for case in catalog.cases if case.id == case_id)
    reply = make_defective_policy_source(case, defect)
    errors.rejects(
            lambda: assertions.assert_conversation_answer(
            reply, case=case, document_title=case_data.POLICY_DOCUMENT_TITLE), expected=AssertionError,
            match="did not cite|source_text")


@title("A concise greeting does not imply that retrieval was skipped")
def test_small_talk_sources_are_not_misrepresented_as_evidence_of_skipped_retrieval(
        catalog: ConversationCatalog) -> None:
    case = catalog.cases[0]
    assertions.assert_conversation_answer(
            response(case.reference), case=case, document_title=case_data.POLICY_DOCUMENT_TITLE)


@pytest.mark.parametrize("case_id", WALKING_INTENT_ACCEPTS_FRESH_AIR_WITHOUT_REQUIRING_LITERAL_WALK_CASE_ID_CASES)
@title("A walking suggestion may use natural paraphrases [{param_id}]")
def test_walking_intent_accepts_fresh_air_without_requiring_literal_walk(
        catalog: ConversationCatalog, case_id: str) -> None:
    case = next(case for case in catalog.cases if case.id == case_id)
    suggestion = "That sounds like a lovely idea; enjoy the fresh air and nature!"
    answer = make_walking_response(case, case_id, suggestion)
    assertions.assert_conversation_answer(response(answer), case=case, document_title=case_data.POLICY_DOCUMENT_TITLE)


@title("Company scope remains valid when an inline source filename precedes its facts")
def test_inline_filename_does_not_break_own_company_fact_scope(catalog: ConversationCatalog) -> None:
    case = next(case for case in catalog.cases if case.id == "mixed_request")
    answer = (
            "Hi! Enjoy the fresh air. At Northern Lighthouse, according to company-policy.txt, "
            "employees get 23 working days of paid leave per year. "
            "I have no policy information about HarborWorks.")
    assertions.assert_conversation_answer(response(answer), case=case, document_title=case_data.POLICY_DOCUMENT_TITLE)


@title("Another company's entitlement cannot satisfy the Northern Lighthouse allowance")
def test_own_company_scope_cannot_consume_another_company_fact(catalog: ConversationCatalog) -> None:
    case = next(case for case in catalog.cases if case.id == "mixed_request")
    answer = (
            "Hi! Enjoy your walk. Northern Lighthouse: see company-policy.txt; "
            "HarborWorks employees get 23 working days per year. I have no information about HarborWorks.")
    errors.rejects(
            lambda: assertions.assert_conversation_answer(
            response(answer), case=case, document_title=case_data.POLICY_DOCUMENT_TITLE), expected=AssertionError,
            match="own company allowance")


@pytest.mark.parametrize(("defect", "message"), INVALID_CATALOGS_FAIL_BEFORE_ANY_GENERATION_DEFECT_MESSAGE_CASES)
@title("Invalid conversation acceptance catalog fails before generation [{param_id}]")
def test_invalid_catalogs_fail_before_any_generation(tmp_path: Path, defect: str, message: str) -> None:
    data = json.loads((DATA / "conversation-policy.json").read_text())
    row = data["cases"][0]
    prepare_invalid_catalogs_fail_before_any_generation_case(data, defect, row)
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(data))
    errors.rejects(
            lambda: load_conversation_catalog(path, DATA / "company-policy.txt"), expected=ValueError, match=message)


@title("Conversation fixture sends an explicit Chat-mode request")
def test_chat_request_explicitly_uses_chat_mode_without_changing_query_client_default(
        mock_factory, unit_settings) -> None:
    api = mock_factory()
    chat = conversation_chat.__wrapped__(
            None, case_data.fresh(CHAT_MODE_CONFIGURATION), api, case_data.fresh(TEMPORARY_WORKSPACE), unit_settings)
    result = chat("Hello!")
    value_checks.identical(result, api.chat.return_value)
    mock_checks.called_once_with(api.chat, "temporary", "Hello!", mode="chat", timeout=300)


@title("Conversation fixture rejects a document-only Query profile")
def test_conversation_fixture_rejects_a_query_profile(mock_factory, unit_settings) -> None:

    errors.rejects(
            lambda: conversation_chat.__wrapped__(
            None, case_data.fresh(QUERY_MODE_CONFIGURATION), mock_factory(), case_data.fresh(TEMPORARY_WORKSPACE),
            unit_settings), expected=AssertionError, match="chatMode")


@title("Conversation selection honors opt-in, model budget and independent repetitions")
def test_conversation_collection_defaults_to_one_model_and_accepts_explicit_matrix(
        framework_pytester: pytest.Pytester) -> None:
    runner = framework_pytester
    shutil.copytree(DATA, runner.path / "test_data")
    runner.makeconftest('pytest_plugins = ["llm_testkit.pytest_support.options"]')
    runner.makepyfile(CONVERSATION_COLLECTION_DEFAULTS_TO_ONE_MODEL_AND_ACCEPTS_EXPLICIT_MATRIX_MAKEPYFILE_SOURCE)
    pytest_runs.outcomes(
            runner.runpytest_subprocess("--run-conversation", "-k", "greeting", "-q"), passed=1, deselected=4)
    pytest_runs.outcomes(
            runner.runpytest_subprocess(
            "--run-conversation", "--rag-model", "model-a", "--rag-model", "model-b", "--rag-repeat", "2", "-k",
            "greeting", "-q"), passed=4, deselected=16)
    result = runner.runpytest_subprocess("--run-conversation", "--capture-rag", "-k", "greeting")
    value_checks.equal(result.ret, pytest.ExitCode.USAGE_ERROR)
    result.stderr.fnmatch_lines(["*small talk is not a RAG sample*"])


@title("Conversation fixture rejects expectations changed after collection")
def test_catalog_change_after_collection_is_not_accepted_as_the_original_case(catalog: ConversationCatalog) -> None:
    changed_case = replace(catalog.cases[0], question="Different question")
    errors.rejects(
            lambda: conversation_metadata.__wrapped__(
            DATA.parent, DATA / "company-policy.txt", changed_case, lambda *_: None), expected=pytest.fail.Exception,
            match="changed after collection")
