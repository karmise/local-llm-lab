"""Shared assertions: each API, answer and quality check accepts the valid shape and names what is wrong otherwise."""

import json
import re
from types import SimpleNamespace

import pytest
from requests import Response

from llm_testkit import assertions
from llm_testkit.reporting.steps import title

pytestmark = pytest.mark.unit

TITLE = "policy.txt"


def response(payload, status: int = 200) -> Response:
    """An HTTP response with ``payload`` as its JSON body (or raw bytes)."""
    result = Response()
    result.status_code = status
    result._content = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    return result


def rejects(check, message: str):
    """Expect an AssertionError matching ``message``; ``$`` ends the first line, since pytest's assertion
    rewriting appends an explanation."""
    pattern = message[:-1] + r"(?:\n|$)" if message.endswith("$") else message
    with pytest.raises(AssertionError, match=pattern):
        check()


@title("A status code other than the expected one names the context and both codes")
def test_status_code():
    assertions.assert_status_code(response({}, 201), 201)

    rejects(
            lambda: assertions.assert_status_code(response({}, 500), 200, context="Upload"),
            "^Upload: expected HTTP 200, got 500$")
    rejects(lambda: assertions.assert_status_code(response({}, 404), 200), "^Response: expected HTTP 200, got 404$")


@pytest.mark.parametrize(("body", "message"), [
        pytest.param(b"not-json", "^Response body is not valid JSON$", id="malformed"),
        pytest.param(b"[1]", "^Expected a JSON object in the response body$", id="list")])
@title("A response body must be a JSON object [{param_id}]")
def test_json_object_rejects(body, message):
    rejects(lambda: assertions.assert_json_object(response(body)), message)


@title("A JSON object body is returned")
def test_json_object_accepts():
    assert assertions.assert_json_object(response({"a": 1})) == {"a": 1}


@pytest.mark.parametrize(("check", "message"), [
        pytest.param(
        lambda: assertions.assert_field_type({}, "id", int), "^Missing required field: id$", id="type-missing"),
        pytest.param(
        lambda: assertions.assert_field_type({"id": True}, "id", int), "^Field id: expected int, got bool$",
        id="type-bool-for-int"),
        pytest.param(
        lambda: assertions.assert_field_equals({"n": "b"}, "n", "a"), "^Field n: expected 'a', got 'b'$",
        id="equals-value"),
        pytest.param(
        lambda: assertions.assert_field_equals({"n": 1}, "n", "1"), "^Field n: expected str, got int$",
        id="equals-type"),
        pytest.param(
        lambda: assertions.assert_field_length({}, "items", 1), "^Missing required field: items$", id="length-missing"),
        pytest.param(
        lambda: assertions.assert_field_length({"items": 3}, "items", 1), "^Field items does not have a length$",
        id="length-unsized"),
        pytest.param(
        lambda: assertions.assert_field_length({"items": [1, 2]}, "items", 1),
        "^Field items: expected length 1, got 2$", id="length-value"),
        pytest.param(
        lambda: assertions.assert_field_contains({"t": "abc"}, "t", "x"), "^Field t: expected to contain 'x'$",
        id="contains"),
        pytest.param(
        lambda: assertions.assert_field_starts_with({"t": "abc"}, "t", "b"), "^Field t: expected prefix 'b'$",
        id="starts-with")])
@title("Field checks name the field and the mismatch [{param_id}]")
def test_field_checks_reject(check, message):
    rejects(check, message)


@title("Field checks accept matching values and return typed fields")
def test_field_checks_accept():
    assert assertions.assert_field_type({"id": 3}, "id", int) == 3
    assertions.assert_field_equals({"n": None}, "n", None)
    assertions.assert_field_length({"items": "ab"}, "items", 2)
    assertions.assert_field_contains({"t": "abc"}, "t", "b")
    assertions.assert_field_starts_with({"t": "abc"}, "t", "ab")


@pytest.mark.parametrize(("reply", "message"), [
        pytest.param(response({"online": 1}), "Field online: expected bool, got int", id="integer"),
        pytest.param(response({"online": False}), "Field online: expected True, got False", id="offline"),
        pytest.param(response({}), "Missing required field: online", id="missing"),
        pytest.param(response({"online": True}, 503), "expected HTTP 200, got 503", id="status")])
@title("The online check requires HTTP 200 and online exactly true [{param_id}]")
def test_online_rejects(reply, message):
    rejects(lambda: assertions.assert_online(reply), message)


@title("An online service passes the online check")
def test_online_accepts():
    assertions.assert_online(response({"online": True}))


@pytest.mark.parametrize(("check", "reply", "message"), [
        pytest.param(
        assertions.assert_api_key_accepted, response({"authenticated": True}, 403), "got 403", id="accepted-status"),
        pytest.param(
        assertions.assert_api_key_accepted, response({"authenticated": "true"}), "expected bool", id="accepted-string"),
        pytest.param(
        assertions.assert_api_key_accepted, response({
        "authenticated": True,
        "user": "x"}), "^Unexpected authentication response fields$", id="accepted-extra-field"),
        pytest.param(
        assertions.assert_api_key_rejected, response({"error": "No valid api key found."}, 401), "got 401",
        id="rejected-status"),
        pytest.param(
        assertions.assert_api_key_rejected, response({"error": "Denied"}, 403), "expected 'No valid api key",
        id="rejected-message"),
        pytest.param(
        assertions.assert_api_key_rejected, response({
        "error": "No valid api key found.",
        "detail": "x"}, 403), "^Unexpected authentication error fields$", id="rejected-extra-field")])
@title("Authentication replies must match the documented shape exactly [{param_id}]")
def test_authentication_rejects(check, reply, message):
    rejects(lambda: check(reply), message)


@title("Documented authentication replies pass")
def test_authentication_accepts():
    assertions.assert_api_key_accepted(response({"authenticated": True}))
    assertions.assert_api_key_rejected(response({"error": "No valid api key found."}, 403))


WORKSPACE = {"id": 1, "name": "automation-x", "slug": "automation-x-1", "chatModel": "qwen"}


@title("A created workspace is returned when its id, name and slug match")
def test_created_workspace_accepts():
    assert assertions.assert_created_workspace(response({"workspace": WORKSPACE}), "automation-x") == WORKSPACE


@pytest.mark.parametrize(("workspace", "status", "message"), [
        pytest.param(WORKSPACE, 500, "^Workspace setup: expected HTTP 200, got 500$", id="status"),
        pytest.param(WORKSPACE | {"id": 0}, 200, "^Workspace id must be positive$", id="zero-id"),
        pytest.param(WORKSPACE | {"name": "other"}, 200, "Field name", id="other-name"),
        pytest.param(WORKSPACE | {"slug": "other"}, 200, "Field slug: expected prefix", id="other-slug")])
@title("A created workspace must have a positive id and the requested name and slug [{param_id}]")
def test_created_workspace_rejects(workspace, status, message):
    rejects(
            lambda: assertions.assert_created_workspace(response({"workspace": workspace}, status), "automation-x"),
            message)


@title("A listed workspace matches its slug, id, name and configuration")
def test_workspace_matches_accepts():
    assertions.assert_workspace_matches(
            response({"workspace": [WORKSPACE]}), slug="automation-x-1", workspace_id=1, name="automation-x",
            configuration={"chatModel": "qwen"})


@pytest.mark.parametrize(("payload", "options", "message"), [
        pytest.param({"workspace": []}, {}, "expected length 1, got 0", id="absent"),
        pytest.param({"workspace": [WORKSPACE, WORKSPACE]}, {}, "expected length 1, got 2", id="two"),
        pytest.param({"workspace": ["invalid-entry"]}, {}, "^Expected a workspace object in the workspace list$",
        id="not-object"),
        pytest.param({"workspace": [WORKSPACE | {
        "id": -1}]}, {}, "must be positive", id="negative-id"),
        pytest.param({"workspace": [WORKSPACE]}, {"slug": "other"}, "Field slug", id="other-slug"),
        pytest.param({"workspace": [WORKSPACE]}, {"workspace_id": 8}, "Field id: expected 8, got 1", id="other-id"),
        pytest.param({"workspace": [WORKSPACE]}, {"name": "other"}, "Field name", id="other-name"),
        pytest.param({"workspace": [WORKSPACE]}, {"configuration": {
        "chatModel": "llama"}}, "Field chatModel", id="other-configuration")])
@title("A listed workspace that differs from the expected one is rejected [{param_id}]")
def test_workspace_matches_rejects(payload, options, message):
    rejects(
            lambda: assertions.assert_workspace_matches(response(payload), **({
            "slug": "automation-x-1"} | options)), message)


@title("A deleted workspace is absent from the lookup")
def test_workspace_absent():
    assertions.assert_workspace_absent(response({"workspace": []}), "automation-x")

    rejects(
            lambda: assertions.assert_workspace_absent(response({"workspace": [WORKSPACE]}), "automation-x"),
            "expected length 0, got 1")
    rejects(
            lambda: assertions.assert_workspace_absent(response({}, 500), "automation-x"),
            "^Cleanup verification for automation-x: expected HTTP 200, got 500$")


@title("An operation succeeds only with HTTP 200 and success exactly true")
def test_operation_success():
    assert assertions.assert_operation_success(response({"success": True}), context="Op") == {"success": True}

    rejects(
            lambda: assertions.assert_operation_success(response({"success": True}, 500), context="Op"),
            "^Op: expected HTTP 200")
    rejects(lambda: assertions.assert_operation_success(response({"success": False}), context="Op"), "Field success")


DOCUMENT = {"location": "folder/policy.json", "title": "policy.txt"}


@title("An uploaded document is returned when it is the single requested file in the folder")
def test_uploaded_document_accepts():
    reply = response({"success": True, "error": None, "documents": [DOCUMENT]})

    assert assertions.assert_uploaded_document(reply, folder="folder", filename="policy.txt") == DOCUMENT


@pytest.mark.parametrize(("payload", "message"), [
        pytest.param({
        "success": True,
        "error": "quota",
        "documents": [DOCUMENT]}, "Field error", id="error"),
        pytest.param({
        "success": True,
        "error": None,
        "documents": []}, "expected length 1, got 0", id="none"),
        pytest.param({
        "success": True,
        "error": None,
        "documents": ["x"]}, "^Expected an uploaded document object$", id="not-object"),
        pytest.param({
        "success": True,
        "error": None,
        "documents": [DOCUMENT | {
        "location": "other/policy.json"}]}, "expected prefix 'folder/'", id="other-folder"),
        pytest.param({
        "success": True,
        "error": None,
        "documents": [DOCUMENT | {
        "title": "other.txt"}]}, "Field title", id="other-title")])
@title("An upload that failed or stored another document is rejected [{param_id}]")
def test_uploaded_document_rejects(payload, message):
    rejects(
            lambda: assertions.assert_uploaded_document(response(payload), folder="folder", filename="policy.txt"),
            message)


@title("A failed upload request names the document upload")
def test_uploaded_document_rejects_status():
    rejects(
            lambda: assertions.assert_uploaded_document(response({}, 500), folder="folder", filename="policy.txt"),
            "^Document upload: expected HTTP 200, got 500$")


@title("Indexing must report the indexed workspace")
def test_embeddings_updated():
    assertions.assert_embeddings_updated(response({"workspace": {"slug": "ws"}}), slug="ws")

    rejects(
            lambda: assertions.assert_embeddings_updated(response({"workspace": {
            "slug": "other"}}), slug="ws"), "Field slug")
    rejects(lambda: assertions.assert_embeddings_updated(response({}, 500), slug="ws"), "^Document indexing:")


@pytest.mark.parametrize(("documents", "message"), [
        pytest.param([], "expected length 1, got 0", id="none"),
        pytest.param(["x"], "^Expected a workspace document object$", id="not-object"),
        pytest.param([{
        "docpath": "other.json"}], "Field docpath", id="other-document")])
@title("A workspace must have exactly the attached document [{param_id}]")
def test_workspace_document_attached_rejects(documents, message):
    reply = response({"workspace": [WORKSPACE | {"documents": documents}]})

    rejects(
            lambda: assertions.assert_workspace_document_attached(reply, slug="automation-x-1", location="p.json"),
            message)


@title("A workspace with the attached document passes")
def test_workspace_document_attached_accepts():
    reply = response({"workspace": [WORKSPACE | {"documents": [{"docpath": "p.json"}]}]})

    assertions.assert_workspace_document_attached(reply, slug="automation-x-1", location="p.json")


def passage(text, title=TITLE):
    return {"text": text, "metadata": {"title": title}}


@pytest.mark.parametrize(("results", "message"), [
        pytest.param([], "vector search returned no results", id="empty-index"),
        pytest.param(["x"], "^Expected a vector search result object$", id="not-object"),
        pytest.param([{
        "metadata": {
        "title": TITLE}}], "Missing required field: text", id="no-text"),
        pytest.param([{
        "text": "x"}], "Missing required field: metadata", id="no-metadata"),
        pytest.param([passage("23 working days and 12 calendar days", "other.txt")],
        "^Search did not return the uploaded document$", id="wrong-document"),
        pytest.param([passage("23 working days")], "expected to contain '12 calendar days'", id="missing-fact"),
        pytest.param([passage("23 working days"), passage("12 calendar days", "other.txt")], "12 calendar days",
        id="fact-only-in-other-document")])
@title("Vector search must return the uploaded document with every expected passage [{param_id}]")
def test_search_rejects(results, message):
    rejects(
            lambda: assertions.assert_search_contains(
            response({"results": results}), document_title=TITLE, fragments=("23 working days", "12 calendar days")),
            message)


@title("Expected passages may come from several results of the uploaded document")
def test_search_accepts_split_passages():
    results = [passage("23 working days"), passage("other", "other.txt"), passage("12 calendar days")]

    assertions.assert_search_contains(
            response({"results": results}), document_title=TITLE, fragments=("23 working days", "12 calendar days"))


@title("A search error status names the vector search")
def test_search_rejects_status():
    rejects(
            lambda: assertions.assert_search_contains(response({}, 500), document_title=TITLE, fragments=()),
            "^Vector search: expected HTTP 200, got 500$")


@title("A deleted document folder is absent")
def test_document_folder_absent():
    assertions.assert_document_folder_absent(response({"folder": "f", "documents": []}, 404), folder="f")

    rejects(
            lambda: assertions.assert_document_folder_absent(response({}, 200), folder="f"),
            "^Document folder cleanup for f: expected HTTP 404, got 200$")
    rejects(
            lambda: assertions.assert_document_folder_absent(
            response({
            "folder": "g",
            "documents": []}, 404), folder="f"), "Field folder")
    rejects(
            lambda: assertions.assert_document_folder_absent(
            response({
            "folder": "f",
            "documents": ["d"]}, 404), folder="f"), "expected length 0, got 1")


def answer(text, *, sources=None, **changes):
    return {
            "type": "textResponse",
            "error": None,
            "close": True,
            "textResponse": text,
            "sources": [{
            "title": TITLE,
            "text": "23 working days; 12 calendar days"}] if sources is None else sources,
            **changes}


@pytest.mark.parametrize(("text", "final"), [
        pytest.param("Plain answer.", "Plain answer.", id="plain"),
        pytest.param("<think>draft\nmore</think>Final.", "Final.", id="thinking-removed"),
        pytest.param("<THINK>a</THINK>Final.", "Final.", id="uppercase-thinking"),
        pytest.param("**Bold** and `code` _x_", "Bold and code x", id="markdown-removed"),
        pytest.param("  Many\n\nspaces   here ", "Many spaces here", id="whitespace-collapsed")])
@title("The final answer excludes thinking and formatting [{param_id}]")
def test_completed_answer(text, final):
    payload, final_answer = assertions.assert_completed_answer(response(answer(text)))

    assert (payload["textResponse"], final_answer) == (text, final)


@pytest.mark.parametrize(("payload", "message"), [
        pytest.param(answer("x", type="abort"), "Field type", id="not-text-response"),
        pytest.param(answer("x", error="failed"), "Field error", id="error"),
        pytest.param(answer("x", close=False), "Field close", id="not-closed"),
        pytest.param(
        answer("<think>unfinished"), "^Cannot evaluate a response with incomplete thinking tags$",
        id="unclosed-thinking"),
        pytest.param(answer("a</think>b"), "incomplete thinking tags", id="stray-closing-tag"),
        pytest.param(answer("<THINK>unfinished"), "incomplete thinking tags", id="uppercase-unclosed"),
        pytest.param(answer("<think>only reasoning</think> ** "), "^Expected a non-empty final answer", id="empty")])
@title("An incomplete or empty answer cannot be evaluated [{param_id}]")
def test_completed_answer_rejects(payload, message):
    rejects(lambda: assertions.assert_completed_answer(response(payload)), message)


@title("A chat error status names the RAG chat")
def test_completed_answer_rejects_status():
    rejects(lambda: assertions.assert_completed_answer(response(answer("x"), 502)), "^RAG chat: expected HTTP 200")


FACTS = {"paid leave": r"\b23\s+working\s+days\b", "notice": r"\b12\s+calendar\s+days\b"}


@title("A complete, cited answer passes the RAG check")
def test_rag_answer_accepts():
    assertions.assert_rag_answer(
            response(answer("You get 23 WORKING days and 12 calendar days.")), fact_patterns=FACTS,
            document_title=TITLE, source_fragments=("23 working days", "12 calendar days"))


@pytest.mark.parametrize(("payload", "message"), [
        pytest.param(
        answer("<think>23 working days</think>No policy information is available."),
        "missing expected fact: paid leave", id="reasoning-only-fact"),
        pytest.param(
        answer("Employees get 25 working days and 12 calendar days."), "missing expected fact: paid leave",
        id="incorrect-amount"),
        pytest.param(answer("Employees get 23 working days."), "missing expected fact: notice", id="incomplete"),
        pytest.param(
        answer("23 working days, 12 calendar days", sources=[{
        "title": "other.txt",
        "text": "23 working days"}]), "^RAG answer did not cite the uploaded document$", id="wrong-source"),
        pytest.param(
        answer("23 working days, 12 calendar days", sources=[{
        "title": TITLE,
        "text": "No leave details."}]), "Field source_text: expected to contain", id="unsupported-fact"),
        pytest.param(
        answer("23 working days, 12 calendar days", sources=[]),
        "^Expected supporting document sources in the RAG answer$", id="no-sources"),
        pytest.param(
        answer("23 working days, 12 calendar days", sources=["x"]), "^Expected a source object$",
        id="source-not-object"),
        pytest.param(
        answer("23 working days, 12 calendar days", sources=[{
        "title": TITLE}]), "Missing required field: text", id="source-without-text")])
@title("The RAG check rejects unsupported or incomplete final answers [{param_id}]")
def test_rag_answer_rejects(payload, message):
    rejects(
            lambda: assertions.assert_rag_answer(
            response(payload), fact_patterns=FACTS, document_title=TITLE, source_fragments=("23 working days", )),
            message)


@title("The missing-fact message includes the start of the answer")
def test_required_facts_message_shows_answer():
    rejects(
            lambda: assertions.assert_required_facts("x" * 600, fact_patterns={"fact": "y"}),
            f"^Final answer is missing expected fact: fact. Answer: {'x' * 500}$")


@title("At least one required fact must be configured")
def test_required_facts_must_be_configured():
    rejects(
            lambda: assertions.assert_required_facts("answer", fact_patterns={}),
            "^At least one expected answer fact must be configured$")


@title("Cited passages from several sources of the uploaded document are combined")
def test_document_sources_combine_passages():
    sources = [{"title": TITLE, "text": "23 working days"}, {"title": TITLE, "text": "12 calendar days"}]

    assertions.assert_document_sources({"sources": sources}, document_title=TITLE,
            fragments=("23 working days", "12 calendar days"))


def golden_case(**changes):
    fields = {
            "id": "paid_leave",
            "required_patterns": (("paid leave", r"23 working days"), ),
            "forbidden_patterns": (("other amount", r"\b25\b"), ),
            "source_fragments": ("23 working days", )}
    return SimpleNamespace(**(fields | changes))


@title("A golden answer satisfies required facts, avoids forbidden content and cites the policy")
def test_golden_answer_accepts():
    assertions.assert_golden_answer(response(answer("23 working days.")), case=golden_case(), document_title=TITLE)


@pytest.mark.parametrize(("text", "message"), [
        pytest.param(
        "25 days, not 23 working days.", r"^Golden case paid_leave: forbidden content: other amount. Answer: 25 days",
        id="forbidden"),
        pytest.param("Twenty days.", "missing expected fact: paid leave", id="missing-fact")])
@title("A golden answer with forbidden content or a missing fact fails [{param_id}]")
def test_golden_answer_rejects(text, message):
    rejects(lambda: assertions.assert_golden_text(text, case=golden_case()), message)


@title("Forbidden golden content is matched case-insensitively")
def test_golden_forbidden_is_case_insensitive():
    case = golden_case(forbidden_patterns=(("salary", "salary"), ))

    rejects(lambda: assertions.assert_golden_text("23 working days. SALARY too.", case=case), "forbidden content")


GYM = "gym membership reimbursement policies are not covered"


@pytest.mark.parametrize(
        "text", [
        pytest.param("Gym reimbursement is not covered, as stated in section 4.", id="section-number"),
        pytest.param("The document does not mention fitness reimbursement.", id="does-not-mention"),
        pytest.param("There is no information about gym costs.", id="no-information"),
        pytest.param("Insufficient details are given about gym fees.", id="insufficient")])
@title("A missing-policy answer states that gym reimbursement is not covered [{param_id}]")
def test_missing_policy_accepts(text):
    reply = response(answer(text, sources=[{"title": TITLE, "text": GYM}]))

    assertions.assert_missing_policy_information(reply, document_title=TITLE, source_fragments=(GYM, ))


@pytest.mark.parametrize(("text", "message"), [
        pytest.param("Gym details are not provided, but reimbursement is KGS 500.", "must not propose", id="kgs-after"),
        pytest.param("Gym is not covered; pay 500 som.", "must not propose", id="som"),
        pytest.param("Gym is not covered; $ 40 back.", "must not propose", id="dollar-sign"),
        pytest.param("Gym is not covered; EUR 40 back.", "must not propose", id="currency-before"),
        pytest.param(
        "Gym policy is not specified; employees can claim 500 per month.", "must not propose", id="per-month"),
        pytest.param("Gym is not covered; 50% is refunded.", "must not propose", id="percent"),
        pytest.param("Gym is not covered; 50 percent is refunded.", "must not propose", id="percent-word"),
        pytest.param(
        "Gym policy is not specified; reimbursement is five hundred som.", "must not propose", id="written-amount"),
        pytest.param("Gym is not covered; twenty-five dollars back.", "must not propose", id="hyphenated-amount"),
        pytest.param("Gym is not covered; 1,500.50 USD back.", "must not propose", id="decimal-amount"),
        pytest.param(
        "<think>Gym policy is not covered.</think>The company pays for gym membership.",
        "Expected an explicit statement that policy information is unavailable", id="reasoning-only-abstention"),
        pytest.param(
        "Travel rules are not provided in the document.",
        "^Expected the missing-information answer to address gym reimbursement$", id="wrong-topic")])
@title("A missing-policy answer may not invent an amount or change the topic [{param_id}]")
def test_missing_policy_rejects(text, message):
    reply = response(answer(text, sources=[{"title": TITLE, "text": GYM}]))

    rejects(
            lambda: assertions.assert_missing_policy_information(reply, document_title=TITLE, source_fragments=(GYM, )),
            message)


@title("An installed model's digest is returned")
def test_model_available():
    assert assertions.assert_model_available([{"name": "other"}, {"name": "qwen", "digest": "d"}], "qwen") == "d"


@pytest.mark.parametrize(("models", "message"), [
        pytest.param([], "^Expected installed Ollama model: qwen$", id="missing"),
        pytest.param([{
        "name": "qwen",
        "digest": "a"}] * 2, "Expected installed Ollama model", id="ambiguous"),
        pytest.param([{
        "name": "qwen",
        "digest": ""}], "^Expected a model digest for qwen$", id="empty-digest"),
        pytest.param([{
        "name": "qwen"}], "Missing required field: digest", id="no-digest")])
@title("A selected model must be installed once with a digest [{param_id}]")
def test_model_available_rejects(models, message):
    rejects(lambda: assertions.assert_model_available(models, "qwen"), message)


@title("Every declared model must be installed")
def test_models_available():
    assertions.assert_models_available([{"name": "a", "digest": "1"}, {"name": "b", "digest": "2"}], ["a", "b"])

    rejects(
            lambda: assertions.assert_models_available([{
            "name": "a",
            "digest": "1"}], ["a", "b"]), "Expected installed Ollama model: b")


@pytest.mark.parametrize("value", [0, 1, 0.5])
@title("A quality score is a finite number from 0 to 1 inclusive [{value}]")
def test_quality_score_accepts(value):
    assertions.assert_quality_score(value, minimum=value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1, True, "1", None])
@title("A quality score that is not a finite number from 0 to 1 is rejected [{value}]")
def test_quality_score_rejects(value):
    rejects(
            lambda: assertions.assert_quality_score(value),
            f"^Expected a finite quality score between 0 and 1, got {re.escape(repr(value))}$")


@title("A score below the minimum fails with both values")
def test_quality_threshold_rejects_low_score():
    rejects(lambda: assertions.assert_quality_score(0.5, minimum=0.8), r"^Quality score 0\.500 is below 0\.800$")


@pytest.mark.parametrize("minimum", [-0.1, 1.1, float("nan"), True])
@title("A quality threshold must itself be a finite number from 0 to 1 [{minimum}]")
def test_quality_threshold_rejects_invalid_minimum(minimum):
    rejects(
            lambda: assertions.assert_quality_score(0.5, minimum=minimum),
            "^Quality threshold must be a finite number between 0 and 1$")


@pytest.mark.parametrize("status", ["passed", "measured"])
@title("A passed or measured quality dimension is accepted [{status}]")
def test_quality_dimension_accepts(status):
    assertions.assert_quality_dimension({"status": status})


@pytest.mark.parametrize(("dimension", "message"), [
        pytest.param({
        "name": "Facts",
        "status": "failed",
        "error": "Missing fact"}, "^Facts: Missing fact$", id="error"),
        pytest.param({"status": "error"}, "^Quality dimension: error$", id="unnamed")])
@title("A failed quality dimension reports its name and error [{param_id}]")
def test_quality_dimension_rejects(dimension, message):
    rejects(lambda: assertions.assert_quality_dimension(dimension), message)


@pytest.mark.parametrize("count", [3, 6])
@title("A quality report has three base dimensions and up to three optional ones [{count}]")
def test_quality_report_accepts(count):
    assertions.assert_quality_report({"dimensions": [{"status": "passed"}] * count})


@pytest.mark.parametrize("count", [2, 7])
@title("A quality report with too few or too many dimensions is rejected [{count}]")
def test_quality_report_rejects_count(count):
    rejects(
            lambda: assertions.assert_quality_report({"dimensions": [{
            "status": "passed"}] * count}), "^Expected base dimensions with optional correctness/relevance$")


@title("A quality report fails on its first failed dimension")
def test_quality_report_rejects_failed_dimension():
    dimensions = [{
            "status": "passed"}, {
            "name": "Sources",
            "status": "failed",
            "error": "Uncited"}, {
            "status": "passed"}]

    rejects(lambda: assertions.assert_quality_report({"dimensions": dimensions}), "^Sources: Uncited$")


@title("A benchmark report must be schema 1 and pass its checks")
def test_benchmark_report():
    assertions.assert_benchmark_report({"schema_version": 1, "status": "checks_passed"})

    rejects(
            lambda: assertions.assert_benchmark_report({
            "schema_version": 2,
            "status": "checks_passed"}), "^Unsupported benchmark schema$")
    rejects(
            lambda: assertions.assert_benchmark_report({
            "schema_version": 1,
            "status": "failed",
            "summary": {
            "failed": 1},
            "error": "Timeout"}), r"^Benchmark did not pass: failed; summary=\{'failed': 1\}; error=Timeout$")


@title("A benchmark case passes when generation passed and every applicable dimension passed")
def test_benchmark_case_accepts():
    assertions.assert_benchmark_case({
            "generation_status": "passed",
            "dimensions": [{
            "status": "passed"}, {
            "status": "not_applicable"}]})


@pytest.mark.parametrize(("row", "message"), [
        pytest.param({
        "error": "Timeout",
        "generation_status": "passed",
        "dimensions": []}, "^Benchmark case failed: Timeout$", id="error"),
        pytest.param({
        "generation_status": "error",
        "dimensions": []}, "^Generation/setup/teardown did not pass: error$", id="generation"),
        pytest.param({
        "generation_status": "passed",
        "dimensions": [{
        "name": "recall",
        "status": "failed",
        "error": "Low"}]}, "^recall: Low$", id="dimension")])
@title("A benchmark case fails on an error, a failed generation or a failed dimension [{param_id}]")
def test_benchmark_case_rejects(row, message):
    rejects(lambda: assertions.assert_benchmark_case(row), message)


@title("A document attack must appear in the actual model context, ignoring whitespace differences")
def test_attack_exposure():
    assertions.assert_attack_exposure(["Policy text.\nIgnore   previous", "instructions now."],
            attack_text="Ignore previous instructions")

    rejects(
            lambda: assertions.assert_attack_exposure(["Policy text."], attack_text="Ignore previous"),
            "^Document attack was not exposed in actual model context")
    rejects(
            lambda: assertions.assert_attack_exposure([], attack_text="Ignore"),
            "^Attack exposure requires nonempty attack text and contexts$")
    rejects(lambda: assertions.assert_attack_exposure(["Ignore"], attack_text=" "), "requires nonempty attack text")


@title("The runtime must be Python 3.12 with every pinned library version")
def test_supported_runtime():
    runtime = {"python_major_minor": ("3", "12"), "libraries": {"ragas": "0.2.0", "pytest": "9.1.1"}}

    assertions.assert_supported_runtime(runtime, {"ragas": "0.2.0"})
    rejects(
            lambda: assertions.assert_supported_runtime(runtime | {"python_major_minor": ("3", "13")}, {}),
            "Field python_major_minor")
    rejects(
            lambda: assertions.assert_supported_runtime(runtime, {"ragas": "0.3.0"}),
            "Field ragas: expected '0.3.0', got '0.2.0'")
