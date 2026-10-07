import asyncio
import hashlib
import json
from unittest.mock import Mock

import pytest
from requests import Response

from llm_testkit.evaluation.correctness import (
    bind_case,
    check_control,
    check_correctness_evidence,
    score_correctness,
    validate_control,
    validate_result,
)
from llm_testkit.reporting.quality import build_quality_report
from llm_testkit.reporting.steps import title
from test_support.builders.correctness import (
    CASE,
    DATASET,
    ROOT,
    _calls,
    _evidence,
    _result,
    _sample,
    _verdicts,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("rv", "gv", "expected"),
    [((1, 1), (1, 1), 1.0), ((1,), (1, 0), 0.67), ((0, 0), (0, 0), 0.0), ((1, 1, 0), (1, 1), 0.8)],
    ids=["correct", "incomplete", "contradicted", "extra-claim"],
)
@title(
    "Real RAGAS factual F1 distinguishes correct, incomplete and incorrect evidence [{param_id}]"
)
def test_real_ragas_correctness_with_mocked_judge(rv, gv, expected):
    pytest.importorskip("ragas")
    from llm_testkit.evaluation.ollama_judge import OllamaJudge

    result = _result(rv, gv, expected)
    outputs = [
        {"claims": result["response_claims"]},
        {"statements": result["response_verdicts"]},
        {"claims": result["reference_claims"]},
        {"statements": result["reference_verdicts"]},
    ]
    responses = []
    for output in outputs:
        response = Response()
        response.status_code = 200
        response._content = json.dumps(
            {
                "model": "test-model",
                "done": True,
                "done_reason": "stop",
                "message": {"content": json.dumps(output)},
            }
        ).encode()
        responses.append(response)
    client = Mock()
    client.structured_chat.side_effect = responses
    judge = OllamaJudge(client, "test-model", max_calls=4)
    scored = asyncio.run(score_correctness(_sample(), judge))
    assert scored["value"] == expected
    assert client.structured_chat.call_count == 4
    with pytest.raises(ValueError, match="budget"):
        judge.generate("extra", Mock())


@pytest.mark.parametrize(
    "change", ["missing", "duplicate", "invalid-verdict", "score", "counts", "empty", "reason"]
)
@title("Correctness rejects incomplete, duplicated or inconsistent judge evidence [{param_id}]")
def test_invalid_claim_evidence_is_rejected(change):
    result = _result()
    if change == "missing":
        result["reference_verdicts"].pop()
    elif change == "duplicate":
        result["response_claims"][1] = result["response_claims"][0]
    elif change == "invalid-verdict":
        result["response_verdicts"][0]["verdict"] = True
    elif change == "score":
        result["value"] = 0.5
    elif change == "counts":
        result["counts"] = {"tp": 9, "fp": 0, "fn": 0}
    elif change == "empty":
        result["response_claims"] = []
    else:
        result["reference_verdicts"][0]["reason"] = ""
    with pytest.raises(ValueError):
        validate_result(result)


@pytest.mark.parametrize(
    "change", ["sample", "dataset", "reference", "response", "config", "control", "unfinished"]
)
@title(
    "Correctness evidence remains bound to application sample and golden expectations [{param_id}]"
)
def test_unrelated_or_synthetic_evidence_is_rejected(change):
    sample = _sample()
    evidence = _evidence(sample)
    field = {
        "sample": "sample_sha256",
        "dataset": "golden_dataset_sha256",
        "reference": "reference",
        "response": "response",
        "config": "metric_configuration",
        "control": "response_origin",
        "unfinished": "status",
    }[change]
    evidence[field] = "changed"
    with pytest.raises(ValueError):
        check_correctness_evidence(evidence, "sample", sample, DATASET)


@title("Correctness rejects substituted reference and stale sample provenance")
def test_sample_must_use_current_golden_reference():
    sample = _sample()
    sample["reference"] = "Edited reference"
    with pytest.raises(ValueError, match="question/reference"):
        bind_case(sample, DATASET, CASE.id)
    sample = _sample()
    sample["metadata"] = {"golden_dataset_sha256": "stale"}
    with pytest.raises(ValueError, match="provenance"):
        bind_case(sample, DATASET, CASE.id)


@title("Optional correctness records a valid low score and independently rejects broken evidence")
def test_quality_report_adds_independent_correctness_measurement(tmp_path):
    sample = _sample()
    source = tmp_path / "sample.json"
    source.write_text(json.dumps(sample))
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    faithful = {
        "schema_version": 1,
        "metric": "faithfulness",
        "status": "completed",
        "sample_sha256": checksum,
        "result": {
            "value": 1.0,
            "statements": ["Claim"],
            "verdicts": [{"statement": "Claim", "verdict": 1}],
        },
        "judge_model": "judge",
        "judge_model_digest": "digest",
        "judge_configuration": {},
        "ragas_version": "test",
        "created_at": "now",
    }
    faith_path = tmp_path / "faith.json"
    faith_path.write_text(json.dumps(faithful))
    correctness = _evidence(sample, checksum)
    correctness["result"] = _result((0, 0), (0, 0), 0.0)
    correctness["judge_calls"] = _calls(correctness["result"])
    correct_path = tmp_path / "correct.json"
    correct_path.write_text(json.dumps(correctness))
    report = build_quality_report(
        source, faith_path, ROOT / "quality-paid-leave.json", correctness_path=correct_path
    )
    assert len(report["dimensions"]) == 4
    assert report["status"] == "checks_passed"
    assert report["dimensions"][-1]["details"]["value"] == 0.0
    correctness["sample_sha256"] = "other"
    correct_path.write_text(json.dumps(correctness))
    report = build_quality_report(
        source, faith_path, ROOT / "quality-paid-leave.json", correctness_path=correct_path
    )
    assert report["dimensions"][-1]["status"] == "error"
    assert report["dimensions"][2]["status"] == "measured"


@title("Correctness CLI refuses overwriting evidence before model calls")
def test_correctness_cli_refuses_overwrite(tmp_path, monkeypatch):
    from llm_testkit.evaluation.correctness import main

    output = tmp_path / "existing.json"
    output.write_text("existing")
    monkeypatch.setattr(
        "sys.argv", ["correctness", "sample", "--case", "paid_leave", "--output", str(output)]
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert output.read_text() == "existing"


@title("Correctness controls tolerate decomposition variation while verifying omission labels")
def test_incomplete_control_checks_semantics_instead_of_fixed_claim_count():
    control = next(
        c
        for c in json.loads((ROOT / "correctness-controls.json").read_text())["cases"]
        if c["id"] == "incomplete"
    )
    result = _result((1, 1), (1, 0), 0.8)
    claims = [
        "Each employee receives 23 working days of paid leave.",
        "A request needs 12 calendar days before leave starts.",
    ]
    result["reference_claims"] = claims
    result["reference_verdicts"] = _verdicts(claims, (1, 0))
    check_control(result, control)
    result["reference_verdicts"] = _verdicts(claims, (1, 1))
    result["value"] = 1.0
    with pytest.raises(ValueError):
        check_control(result, control)


@title("All curated correctness controls have valid labelled expectations")
def test_control_catalog_has_valid_expectations():
    cases = json.loads((ROOT / "correctness-controls.json").read_text())["cases"]
    assert len({c["id"] for c in cases}) == len(cases) == 4
    for control in cases:
        validate_control(control)
        assert control["case_id"] == CASE.id


@title("Correctness report rejects summaries that differ from raw judge evidence")
def test_raw_calls_must_match_summary():
    sample = _sample()
    evidence = _evidence(sample)
    evidence["judge_calls"][0]["output"]["claims"] = ["Edited"]
    with pytest.raises(ValueError):
        check_correctness_evidence(evidence, "sample", sample, DATASET)


@title("Correctness service preserves failure evidence and closes its transport")
def test_correctness_service_closes_transport_on_judge_error(tmp_path, monkeypatch):
    pytest.importorskip("ragas")
    from llm_testkit.config import Settings
    from llm_testkit.evaluation.correctness import evaluate_correctness_report

    path = tmp_path / "sample.json"
    path.write_text(json.dumps(_sample()))
    transport = Mock()
    client = Mock()
    catalog = Response()
    catalog.status_code = 200
    catalog._content = json.dumps({"models": [{"name": "test-model", "digest": "digest"}]}).encode()
    client.list_models.return_value = catalog
    judge = Mock(calls=[{"error": "truncated"}], options={})
    monkeypatch.setattr(
        "llm_testkit.evaluation.correctness.HttpClient", Mock(return_value=transport)
    )
    monkeypatch.setattr(
        "llm_testkit.evaluation.correctness.OllamaClient", Mock(return_value=client)
    )
    monkeypatch.setattr("llm_testkit.evaluation.ollama_judge.OllamaJudge", Mock(return_value=judge))

    async def failed_score(sample, judge):
        raise ValueError("Truncated judge response")

    monkeypatch.setattr("llm_testkit.evaluation.correctness.score_correctness", failed_score)
    report = evaluate_correctness_report(
        path,
        dataset_path=ROOT / "golden-policy.json",
        policy_file=ROOT / "company-policy.txt",
        case_id=CASE.id,
        settings=Settings(),
        judge_model="test-model",
    )
    assert report["status"] == "error"
    assert report["judge_calls"] == judge.calls
    assert transport.close.call_count == 1
