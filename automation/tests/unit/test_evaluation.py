import asyncio
import json
from pathlib import Path

import pytest

from llm_testkit import assertions
from llm_testkit.evaluation.faithfulness import (
    load_sample,
    main,
    score_sample,
)
from llm_testkit.reporting.steps import title
from test_support.assertions import errors as errors
from test_support.assertions import judges as judge_checks
from test_support.assertions import mocks as mock_checks
from test_support.assertions import values as value_checks
from test_support.builders.evaluation import (
    _judge_class,
    _sample,
)
from test_support.data import common as case_data
from test_support.data.evaluation import (
    DETERMINISTIC_JUDGE_OPTIONS,
    EXTRACTED_CLAIM_RESPONSE,
    OBJECT_RESPONSE_SCHEMA,
    PIPELINE_REJECTS_MISSING_ALTERED_OR_INVALID_VERDICTS_OUTPUT_CASES,
    QUALITY_SCORE_REJECTS_INVALID_RESULTS_VALUE_CASES,
    UNSUPPORTED_CLAIM_VERDICT,
)
from test_support.fixtures.unit_evaluation import (
    failed_evaluation_service as failed_evaluation_service,
)
from test_support.fixtures.unit_evaluation import faithfulness_judge as faithfulness_judge
from test_support.fixtures.unit_evaluation import (
    invalid_faithfulness_judge as invalid_faithfulness_judge,
)
from test_support.fixtures.unit_evaluation import truncated_judge as truncated_judge

pytestmark = pytest.mark.unit


@title("Evaluation sample loader rejects substituted model context")
def test_sample_loader_rejects_context_substitution(tmp_path: Path) -> None:
    sample = _sample()
    sample["retrieved_contexts"] = ["An unrelated policy."]
    path = tmp_path / case_data.SAMPLE_FILE_NAME
    path.write_text(json.dumps(sample))
    errors.rejects(lambda: load_sample(path), expected=ValueError, match="contexts do not match")


@pytest.mark.parametrize("value", QUALITY_SCORE_REJECTS_INVALID_RESULTS_VALUE_CASES)
@title("Quality-score check rejects invalid values [{param_id}]")
def test_quality_score_rejects_invalid_results(value: object) -> None:
    errors.rejects(lambda: assertions.assert_quality_score(value), expected=AssertionError)


@title("Quality threshold rejects a score below the required minimum")
def test_quality_threshold_rejects_low_score() -> None:
    errors.rejects(
        lambda: assertions.assert_quality_score(0.5, minimum=0.8),
        expected=AssertionError,
        match="below",
    )


@title("RAGAS computes supported-claim ratio from mocked judge responses")
def test_real_ragas_pipeline_computes_supported_claim_ratio_without_network(faithfulness_judge):
    result = faithfulness_judge.evaluate()
    judge_checks.score_matches(faithfulness_judge, result)
    judge_checks.budget_is_exhausted(faithfulness_judge)


@pytest.mark.parametrize(
    "output",
    PIPELINE_REJECTS_MISSING_ALTERED_OR_INVALID_VERDICTS_OUTPUT_CASES,
)
@title("Faithfulness pipeline rejects missing, altered or invalid claim verdicts [{param_id}]")
def test_pipeline_rejects_missing_altered_or_invalid_verdicts(output, invalid_faithfulness_judge):
    errors.rejects(invalid_faithfulness_judge.evaluate, expected=(ValueError, AssertionError))


@title("Local judge rejects truncated generation without retrying")
def test_judge_rejects_truncated_generation_without_retry(truncated_judge):
    errors.rejects(truncated_judge.evaluate, expected=ValueError, match="truncated")
    mock_checks.called_once(truncated_judge.client.structured_chat)


@title("Native judge request disables thinking and includes the response schema")
def test_native_judge_request_disables_thinking_and_passes_schema(
    mock_factory, ollama_factory
) -> None:
    http = mock_factory()
    schema = case_data.fresh(OBJECT_RESPONSE_SCHEMA)
    ollama_factory(http).structured_chat(
        model=case_data.TEST_MODEL,
        prompt="Judge this",
        schema=schema,
        options=case_data.fresh(DETERMINISTIC_JUDGE_OPTIONS),
    )
    payload = http.request.call_args.kwargs["json"]
    value_checks.identical(payload["think"], False)
    value_checks.identical(payload["stream"], False)
    value_checks.equal(payload["format"], schema)


@title("Faithfulness CLI preserves a failure in its error report")
def test_cli_preserves_failure_as_error_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _judge_class()

    source = tmp_path / case_data.SAMPLE_FILE_NAME
    source.write_text("{}")
    output = tmp_path / "report.json"
    monkeypatch.setattr("sys.argv", ["faithfulness", str(source), "--output", str(output)])
    value_checks.equal(main(), 1)
    report = json.loads(output.read_text())
    value_checks.equal(report["status"], "error")
    value_checks.equal(report["error"]["type"], "KeyError")


@title("Faithfulness CLI refuses to overwrite an existing report before model calls")
def test_cli_refuses_to_overwrite_report_before_model_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    output = tmp_path / "report.json"
    output.write_text("existing")
    monkeypatch.setattr("sys.argv", ["faithfulness", "missing-sample", "--output", str(output)])
    error = errors.rejects(lambda: main(), expected=SystemExit)
    value_checks.equal(error.value.code, 2)
    value_checks.equal(output.read_text(), "existing")


@title("Evaluation service preserves judge failure and closes its HTTP transport")
def test_evaluation_service_preserves_judge_failure_and_closes_transport(failed_evaluation_service):
    report = failed_evaluation_service.evaluate()
    judge_checks.failure_evidence_is_retained(failed_evaluation_service, report)
    value_checks.equal(report["error"]["type"], "ValueError")


def test_live_faithfulness_rejects_score_inconsistent_with_verdicts(
    monkeypatch, async_mock_factory, mock_factory
):
    collections = pytest.importorskip("ragas.metrics.collections")

    metric = mock_factory()

    metric.ascore = async_mock_factory(return_value=mock_factory(value=1.0))
    monkeypatch.setattr(collections, "Faithfulness", mock_factory(return_value=metric))
    judge = mock_factory(
        calls=[
            case_data.fresh(EXTRACTED_CLAIM_RESPONSE),
            case_data.fresh(UNSUPPORTED_CLAIM_VERDICT),
        ]
    )
    errors.rejects(
        lambda: asyncio.run(score_sample(_sample(), judge)), expected=ValueError, match="score"
    )
