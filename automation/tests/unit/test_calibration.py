"""Hand-labelled faithfulness controls: catalog loading, selection, matching judge output, the runner and the CLI."""

import asyncio
import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timedelta
from importlib.metadata import version
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call

import pytest

from llm_testkit import assertions
from llm_testkit.config import Settings
from llm_testkit.core.number_words import normalize_number_words
from llm_testkit.evaluation.calibration import evaluate_controls, load_controls, main, select_controls
from llm_testkit.reporting.steps import title
from test_support.builders.calibration import CONTROLS_FILE, make_catalog, make_control, make_matching_result
from test_support.builders.golden import make_paid_leave_sample
from test_support.builders.identities import MODEL_DIGEST, TEST_MODEL
from test_support.builders.ollama import model_catalog

pytestmark = pytest.mark.unit

SCORE = "llm_testkit.evaluation.calibration.score_sample"


def check_result(result: dict, control: dict) -> None:
    assertions.assert_calibration_result(result, expected_score=control["expected_score"], claims=control["claims"])


@pytest.fixture
def curated_controls() -> dict[str, dict]:
    return {case["id"]: case for case in json.loads(CONTROLS_FILE.read_text())["cases"]}


@pytest.mark.parametrize(("text", "expected"), [
        pytest.param("THIRTY working days; two calendar days", "30 working days; 2 calendar days", id="mixed-case"),
        pytest.param("twenty-three working days", "23 working days", id="hyphenated"),
        pytest.param("twenty three", "23", id="spaced"),
        pytest.param("thirty one", "31", id="compound-with-one"),
        pytest.param("ninety-nine", "99", id="largest-compound"),
        pytest.param("zero", "0", id="zero"),
        pytest.param("nineteen", "19", id="teen"),
        pytest.param("twenty", "20", id="tens"),
        pytest.param("twenty zero", "twenty zero", id="tens-and-zero"),
        pytest.param("twenty ten", "twenty ten", id="tens-and-ten"),
        pytest.param("thirty thousand", "30000", id="thousands"),
        pytest.param("five thousand KGS", "5000 KGS", id="judge-rewritten-amount"),
        pytest.param("eight thousand four hundred", "8400", id="thousands-and-hundreds"),
        pytest.param("one hundred and thirty", "130", id="hundreds-with-and"),
        pytest.param("two million three hundred thousand", "2300000", id="millions"),
        pytest.param("thirty point two", "thirty point two", id="decimal"),
        pytest.param("twelve hundred", "twelve hundred", id="teen-hundreds"),
        pytest.param("one two", "one two", id="two-units"),
        pytest.param("thousand", "thousand", id="bare-scale"),
        pytest.param("one thousand one million", "one thousand one million", id="rising-scales"),
        pytest.param("zero thousand", "zero thousand", id="zero-scaled")])
@title("Well-formed integer spellings become digits; decimal or malformed phrases are preserved [{param_id}]")
def test_number_spellings(text, expected):
    assert normalize_number_words(text) == expected


@title("A judge result with one statement per labelled claim and the expected verdicts matches its control")
def test_matching_result_satisfies_control():
    check_result(make_matching_result(), make_control())


@pytest.mark.parametrize(("corrupt", "message"), [
        pytest.param(lambda result: result.update(value=1.0), "Control score: expected 0.5, got 1.0", id="other-score"),
        pytest.param(
        lambda result: [verdict.update(verdict=1 - verdict["verdict"])
        for verdict in result["verdicts"]], "Field verdict: expected 1, got 0", id="reversed-verdicts-same-score"),
        pytest.param(lambda result: result["verdicts"].pop(), "number of claims", id="missing-claim"),
        pytest.param(
        lambda result: result["verdicts"][1].update(statement="Unrelated claim."), "one extracted claim matching 5000",
        id="unmatched-claim"),
        pytest.param(
        lambda result: result["verdicts"][0].update(statement="Leave 23; gym 5000."), "one extracted claim matching",
        id="claim-matches-twice")])
@title("A judge result that differs from the hand labels is a mismatch, even when the score agrees [{param_id}]")
def test_disagreeing_result_fails_control(corrupt, message):
    result = make_matching_result()
    corrupt(result)

    with pytest.raises(AssertionError, match=message):
        check_result(result, make_control())


@title("Two labelled claims cannot both be satisfied by one combined judge statement")
def test_two_claims_cannot_share_one_statement():
    result = make_matching_result()
    result["verdicts"] = [{"statement": "Leave 23; gym 5000.", "verdict": 1}, {"statement": "Other.", "verdict": 0}]

    with pytest.raises(AssertionError, match="distinct extracted statements"):
        check_result(result, make_control())


@title("A recorded control accepts the judge spelling its numbers as words")
def test_number_words_match_digit_patterns(curated_controls):
    result = {
            "value":
            0.0,
            "verdicts": [{
            "statement": "Each employee receives thirty working days of paid leave per year.",
            "verdict": 0}, {
            "statement": "A request must be submitted two calendar days before the leave starts.",
            "verdict": 0}]}

    check_result(result, curated_controls["wrong_numbers"])


@pytest.mark.parametrize(
        "statement", [
        pytest.param("Each employee receives twenty working days of paid leave per year.", id="other-number"),
        pytest.param("Each employee receives thirty calendar days of paid leave per year.", id="other-unit"),
        pytest.param("Each employee receives thirty thousand working days of paid leave per year.", id="large-number")])
@title("Control matching still rejects a changed number, a changed unit or an unsupported large value [{param_id}]")
def test_changed_numbers_do_not_match(curated_controls, statement):
    result = {
            "value":
            0.0,
            "verdicts": [{
            "statement": statement,
            "verdict": 0}, {
            "statement": "A request must be submitted two calendar days before the leave starts.",
            "verdict": 0}]}

    with pytest.raises(AssertionError, match="one extracted claim matching"):
        check_result(result, curated_controls["wrong_numbers"])


@title("A valid catalog loads its controls and is identified by the checksum of its bytes")
def test_load_controls_returns_cases_and_checksum(tmp_path):
    path = tmp_path / "controls.json"
    path.write_text(json.dumps(make_catalog(make_control("first"), make_control("second"))))

    cases, checksum = load_controls(path, ["Unrelated", "Policy text"])

    assert [case["id"] for case in cases] == ["first", "second"]
    assert checksum == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("score", [0, 1])
@title("Integer expected scores on the bounds of zero and one are accepted [{param_id}]")
def test_load_controls_accepts_score_bounds(tmp_path, score):
    control = {**make_control(), "expected_score": score, "claims": [{"pattern": "23", "verdict": score}]}
    path = tmp_path / "controls.json"
    path.write_text(json.dumps(make_catalog(control)))

    cases, _ = load_controls(path, ["Policy"])

    assert cases[0]["expected_score"] == score


def corrupt_control(**changes):
    return lambda catalog: catalog["cases"][0].update(changes)


def corrupt_claim(**changes):
    return lambda catalog: catalog["cases"][0]["claims"][0].update(changes)


@pytest.mark.parametrize(("corrupt", "message"), [
        pytest.param(lambda catalog: catalog.update(schema_version=2), "Unsupported calibration schema", id="schema"),
        pytest.param(lambda catalog: catalog.pop("required_context_fragments"), "context anchors", id="no-anchors"),
        pytest.param(
        lambda catalog: catalog.update(required_context_fragments=[]), "context anchors", id="empty-anchors"),
        pytest.param(
        lambda catalog: catalog.update(required_context_fragments=[""]), "context anchors", id="blank-anchor"),
        pytest.param(
        lambda catalog: catalog.update(required_context_fragments=[1]), "context anchors", id="anchor-type"),
        pytest.param(
        lambda catalog: catalog.update(required_context_fragments=["Policy", "Other document"]),
        "does not contain the policy", id="anchor-missing-from-context"),
        pytest.param(lambda catalog: catalog.update(cases=[]), "at least one case", id="no-cases"),
        pytest.param(lambda catalog: catalog.update(cases={}), "at least one case", id="cases-not-a-list"),
        pytest.param(lambda catalog: catalog.update(cases=["mixed"]), "must be an object", id="case-not-an-object"),
        pytest.param(corrupt_control(id=""), "nonempty and unique", id="blank-id"),
        pytest.param(corrupt_control(id=7), "nonempty and unique", id="id-type"),
        pytest.param(lambda catalog: catalog["cases"].append(make_control()), "nonempty and unique", id="duplicate-id"),
        pytest.param(corrupt_control(response=" "), "response must be nonempty", id="blank-response"),
        pytest.param(corrupt_control(expected_score=1.5), "between zero and one", id="score-above-one"),
        pytest.param(corrupt_control(expected_score=-0.5), "between zero and one", id="negative-score"),
        pytest.param(corrupt_control(expected_score=True), "between zero and one", id="boolean-score"),
        pytest.param(corrupt_control(claims=[]), "hand-labelled claims", id="no-claims"),
        pytest.param(corrupt_control(claims=["23"]), "claim must be an object", id="claim-not-an-object"),
        pytest.param(corrupt_claim(verdict=True), "zero or one", id="boolean-verdict"),
        pytest.param(corrupt_claim(verdict=2), "zero or one", id="verdict-out-of-range"),
        pytest.param(corrupt_claim(pattern=""), "patterns must be nonempty", id="blank-pattern"),
        pytest.param(corrupt_claim(pattern="("), "missing \\)", id="invalid-pattern"),
        pytest.param(corrupt_control(expected_score=1.0), "does not match hand-labelled verdicts", id="score-differs")])
@title("Catalogs with unrelated context, malformed controls or inconsistent labels are rejected [{param_id}]")
def test_load_controls_rejects_invalid_catalog(tmp_path, corrupt, message):
    catalog = make_catalog()
    corrupt(catalog)
    path = tmp_path / "controls.json"
    path.write_text(json.dumps(catalog))

    with pytest.raises(Exception, match=message):
        load_controls(path, ["Policy"])


@title("The curated faithfulness control catalog is valid for the policy document")
def test_curated_catalog_is_valid():
    cases, _ = load_controls(CONTROLS_FILE, make_paid_leave_sample()["retrieved_contexts"])

    assert len({case["id"] for case in cases}) == len(cases) >= 4


@title("Without a selection, a catalog of up to three controls is selected in full")
def test_select_all_small_catalog():
    cases = [make_control("first"), make_control("second"), make_control("third")]

    assert select_controls(cases, None) == cases


@title("Explicit selection keeps catalog order and removes duplicates")
def test_select_keeps_catalog_order_without_duplicates():
    cases = [make_control(f"control-{i}") for i in range(6)]

    chosen = select_controls(cases, ["control-5", "control-4", "control-5"])

    assert [case["id"] for case in chosen] == ["control-4", "control-5"]


@pytest.mark.parametrize(("identifiers", "message"), [
        pytest.param(None, "maximum six judge calls", id="whole-large-catalog"),
        pytest.param([f"control-{i}" for i in range(4)], "maximum six judge calls", id="four-explicit"),
        pytest.param([], "Select at least one control", id="empty-selection"),
        pytest.param(["typo", "control-0", "other"], "Unknown controls: other, typo", id="unknown")])
@title("Selections that are empty, unknown or exceed three controls are rejected before model calls [{param_id}]")
def test_select_rejects_invalid_selection(identifiers, message):
    cases = [make_control(f"control-{i}") for i in range(6)]

    with pytest.raises(ValueError, match=message):
        select_controls(cases, identifiers)


@pytest.mark.parametrize("count", [0, 4])
@title("The runner rejects an empty or oversized batch before creating any judge [{param_id}]")
def test_runner_rejects_batch_size(count):
    factory = Mock()

    with pytest.raises(ValueError, match="between one and three"):
        asyncio.run(evaluate_controls({}, [make_control()] * count, factory))

    factory.assert_not_called()


@title("The runner scores each control's own answer with a fresh judge and records match, mismatch and error")
def test_runner_records_each_outcome_and_continues(monkeypatch):
    sample = {"response": "Original application answer", "retrieved_contexts": ["Policy"]}
    controls = [make_control("matched"), make_control("mismatch"), make_control("error")]
    score = AsyncMock(
            side_effect=[
            make_matching_result(), {
            **make_matching_result(), "value": 1.0},
            ValueError("Truncated judge response")])
    monkeypatch.setattr(SCORE, score)
    judges = [Mock(calls=[f"calls {i}"]) for i in range(3)]

    results = asyncio.run(evaluate_controls(sample, controls, Mock(side_effect=judges)))

    assert [row["status"] for row in results] == ["matched", "mismatch", "error"]
    assert [row["control"] for row in results] == controls
    assert [row["judge_calls"] for row in results] == [judge.calls for judge in judges]
    assert results[0]["result"] == make_matching_result()
    assert "Control score" in results[1]["mismatch"]
    assert results[2]["error"] == {"type": "ValueError", "message": "Truncated judge response"}
    assert [awaited.args for awaited in score.await_args_list] == [({
            **sample, "response": control["response"]}, judge) for control, judge in zip(controls, judges, strict=True)]
    assert sample["response"] == "Original application answer"


OPTIMIZED_RUN = """
import asyncio
from llm_testkit.evaluation import calibration

async def contradicting_score(sample, judge):
    return {"value": 0.0, "statements": ["23 working days"],
            "verdicts": [{"statement": "23 working days", "verdict": 0}]}

calibration.score_sample = contradicting_score
control = {"id": "c", "response": "23 working days", "expected_score": 1.0,
           "claims": [{"pattern": "23 working days", "verdict": 1}]}
judge = type("Judge", (), {"calls": []})
rows = asyncio.run(calibration.evaluate_controls({}, [control], judge))
print(rows[0]["status"])
"""


@title("A judge that contradicts the labels is a mismatch even when Python strips assert statements (-O)")
def test_mismatch_is_detected_under_optimized_python():
    result = subprocess.run([sys.executable, "-O", "-c", OPTIMIZED_RUN], capture_output=True, text=True, check=True)

    assert result.stdout == "mismatch\n"


@pytest.fixture
def offline_cli(tmp_path, monkeypatch):
    """The calibration CLI with a real sample and catalog, and the Ollama transport and judges replaced."""
    pytest.importorskip("ragas")
    monkeypatch.delenv("ANYTHINGLLM_API_KEY_FILE", raising=False)
    cli = SimpleNamespace(sample=tmp_path / "sample.json", output=tmp_path / "report.json", transport=Mock())
    cli.sample.write_text(json.dumps(make_paid_leave_sample()))
    cli.ollama = Mock()
    cli.ollama.list_models.return_value = model_catalog((TEST_MODEL, MODEL_DIGEST))
    cli.judges = [Mock(calls=[], options={"temperature": 0}) for _ in range(4)]
    cli.http_class = Mock(return_value=cli.transport)
    cli.judge_class = Mock(side_effect=cli.judges)
    monkeypatch.setattr("llm_testkit.evaluation.calibration.HttpClient", cli.http_class)
    cli.ollama_class = Mock(return_value=cli.ollama)
    monkeypatch.setattr("llm_testkit.evaluation.calibration.OllamaClient", cli.ollama_class)
    monkeypatch.setattr("llm_testkit.evaluation.ollama_judge.OllamaJudge", cli.judge_class)

    def run(*arguments: str) -> int:
        command = ["calibration", str(cli.sample), "--controls", str(CONTROLS_FILE), "--output", str(cli.output)]
        monkeypatch.setattr("sys.argv", [*command, "--judge-model", TEST_MODEL, *arguments])
        return main()

    cli.run = run
    cli.report = lambda: json.loads(cli.output.read_text())
    return cli


def supported_answer_result(verdicts: tuple[int, int] = (1, 1)) -> dict:
    """A judge result for the curated supported_answer control."""
    statements = (
            "Each employee receives 23 working days of paid leave per year.",
            "A leave request must be submitted at least 12 calendar days before leave starts.")
    return {
            "value": sum(verdicts) / 2,
            "verdicts": [{
            "statement": s,
            "verdict": v} for s, v in zip(statements, verdicts, strict=True)]}


@title("A matched control run records its provenance, results and judge, prints each score and exits with 0")
def test_cli_records_matched_run(offline_cli, monkeypatch, capsys):
    monkeypatch.setattr(SCORE, AsyncMock(return_value=supported_answer_result()))
    settings = Settings()

    exit_code = offline_cli.run("--control", "supported_answer")

    report = offline_cli.report()
    assert exit_code == 0
    assert (report["schema_version"], report["metric"], report["status"]) == (1, "faithfulness", "matched")
    assert report["response_origin"] == "synthetic_hand_labelled_controls"
    assert "not statistical judge calibration" in report["interpretation"]
    assert report["ragas_version"] == version("ragas")
    assert datetime.fromisoformat(report["created_at"]).utcoffset() == timedelta(0)
    assert report["sample_path"] == str(offline_cli.sample.resolve())
    assert report["sample_sha256"] == hashlib.sha256(offline_cli.sample.read_bytes()).hexdigest()
    assert report["controls_path"] == str(CONTROLS_FILE.resolve())
    assert report["controls_sha256"] == hashlib.sha256(CONTROLS_FILE.read_bytes()).hexdigest()
    assert report["selected_controls"] == ["supported_answer"]
    assert (report["judge_model"], report["judge_model_digest"]) == (TEST_MODEL, MODEL_DIGEST)
    assert report["judge_configuration"] == {"options": {"temperature": 0}, "think": False, "retries": 0}
    assert [row["status"] for row in report["results"]] == ["matched"]
    output = capsys.readouterr().out
    assert "supported_answer: matched; score=1.0" in output
    assert "Calibration status: matched" in output
    offline_cli.http_class.assert_called_once_with(settings.ollama_base_url, settings.http_timeout)
    offline_cli.ollama_class.assert_called_once_with(offline_cli.transport)
    assert offline_cli.judge_class.call_args_list == [call(offline_cli.ollama, TEST_MODEL, settings.llm_timeout)] * 2
    offline_cli.transport.close.assert_called_once()


@pytest.mark.parametrize(("selection", "outcomes", "status"), [
        pytest.param(["--control", "supported_answer"], [supported_answer_result((1, 0))], "mismatch", id="mismatch"),
        pytest.param(["--control", "supported_answer", "--control", "wrong_numbers"],
        [supported_answer_result(), ValueError("Truncated")], "error", id="error-wins")])
@title("Any mismatch makes the run a mismatch and any error makes it an error; both exit with 1 [{param_id}]")
def test_cli_aggregates_control_outcomes(offline_cli, monkeypatch, selection, outcomes, status):
    monkeypatch.setattr(SCORE, AsyncMock(side_effect=outcomes))

    assert offline_cli.run(*selection) == 1
    assert offline_cli.report()["status"] == status


@title("A failed control prints an unavailable score")
def test_cli_prints_unavailable_score_for_failed_control(offline_cli, monkeypatch, capsys):
    monkeypatch.setattr(SCORE, AsyncMock(side_effect=ValueError("Truncated")))

    offline_cli.run("--control", "supported_answer")

    assert "supported_answer: error; score=unavailable" in capsys.readouterr().out


@title("A judge model missing from the Ollama catalog is saved as an error before any control runs")
def test_cli_requires_installed_judge_model(offline_cli, monkeypatch):
    offline_cli.ollama.list_models.return_value = model_catalog(("other-model", MODEL_DIGEST))
    score = AsyncMock()
    monkeypatch.setattr(SCORE, score)

    assert offline_cli.run("--control", "supported_answer") == 1

    report = offline_cli.report()
    assert report["status"] == "error"
    assert "Expected installed Ollama model" in report["error"]["message"]
    score.assert_not_awaited()
    offline_cli.transport.close.assert_called_once()


@title("A failed Ollama catalog request is saved as an error before any control runs")
def test_cli_requires_reachable_model_catalog(offline_cli, monkeypatch):
    offline_cli.ollama.list_models.return_value = model_catalog(status_code=503)
    monkeypatch.setattr(SCORE, AsyncMock())

    offline_cli.run("--control", "supported_answer")

    assert "Calibration judge model catalog: expected HTTP 200, got 503" in offline_cli.report()["error"]["message"]


@pytest.mark.parametrize(("arguments", "message"), [
        pytest.param(["--control", "typo"], "Unknown controls: typo", id="unknown-control"),
        pytest.param([], "maximum six judge calls", id="whole-catalog")])
@title("An invalid control selection is saved as an error before connecting to Ollama [{param_id}]")
def test_cli_rejects_selection_before_connecting(offline_cli, arguments, message):
    assert offline_cli.run(*arguments) == 1

    report = offline_cli.report()
    assert (report["status"], report["error"]["type"]) == ("error", "ValueError")
    assert message in report["error"]["message"]
    offline_cli.http_class.assert_not_called()


@title("Without --judge-model the CLI records the default judge model")
def test_cli_defaults_judge_model(offline_cli, monkeypatch):
    command = ["calibration", str(offline_cli.sample), "--controls", str(CONTROLS_FILE)]
    monkeypatch.setattr("sys.argv", [*command, "--output", str(offline_cli.output), "--control", "typo"])

    main()

    assert offline_cli.report()["judge_model"] == "qwen3.5:4b"


@title("A sample whose context lacks the catalog's policy is saved as an error before connecting to Ollama")
def test_cli_rejects_unrelated_sample_before_connecting(offline_cli):
    offline_cli.sample.write_text(json.dumps(make_paid_leave_sample(["Unrelated document"])))

    offline_cli.run("--control", "supported_answer")

    assert "does not contain the policy" in offline_cli.report()["error"]["message"]
    offline_cli.http_class.assert_not_called()


@title("The CLI refuses to overwrite an existing report")
def test_cli_refuses_existing_output(offline_cli, capsys):
    offline_cli.output.write_text("existing")

    with pytest.raises(SystemExit) as stopped:
        offline_cli.run("--control", "supported_answer")

    assert stopped.value.code == 2
    assert "Output already exists" in capsys.readouterr().err
    assert offline_cli.output.read_text() == "existing"


@title("The CLI refuses to run without the evaluation dependencies")
def test_cli_requires_evaluation_dependencies(offline_cli, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "llm_testkit.evaluation.ollama_judge", None)

    with pytest.raises(SystemExit) as stopped:
        offline_cli.run("--control", "supported_answer")

    assert stopped.value.code == 2
    assert "Install evaluation dependencies" in capsys.readouterr().err
    assert not offline_cli.output.exists()


@pytest.mark.parametrize("missing", ["--controls", "--output"])
@title("The CLI requires a control catalog and an output file [{param_id}]")
def test_cli_requires_catalog_and_output(monkeypatch, tmp_path, missing):
    monkeypatch.chdir(tmp_path)
    arguments = {"--controls": str(CONTROLS_FILE), "--output": "out.json"}
    del arguments[missing]
    monkeypatch.setattr(
            "sys.argv", ["calibration", "sample.json", *[item for pair in arguments.items() for item in pair]])

    with pytest.raises(SystemExit) as stopped:
        main()

    assert stopped.value.code == 2
    assert not (tmp_path / "out.json").exists()


@title("Runner results keep the control definitions unchanged")
def test_runner_does_not_modify_controls(monkeypatch):
    controls = [make_control()]
    original = deepcopy(controls)
    monkeypatch.setattr(SCORE, AsyncMock(return_value=make_matching_result()))

    asyncio.run(evaluate_controls({"response": "Answer"}, controls, Mock(return_value=Mock(calls=[]))))

    assert controls == original
