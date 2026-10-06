import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock

import pytest

from llm_testkit import assertions
from llm_testkit.observation.evaluation_sample import build_sample
from llm_testkit.reporting.allure_report import present_quality_report
from llm_testkit.reporting.quality import build_quality_report
from llm_testkit.reporting.steps import title

pytestmark = pytest.mark.unit


def _files(tmp_path: Path, *, incomplete: bool = False) -> tuple[Path, Path, Path]:
    identifier = "a" * 32
    title = f"automation-{identifier}-company-policy.txt"
    content = f"<document_metadata>\nsourceDocument: {title}\n</document_metadata>\n23 working days; 12 calendar days"
    capture = {
        "schema_version": 1,
        "boundary": "ollama-sdk-chat",
        "request": {
            "model": "test-model",
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": f"[LLM_TESTKIT_CAPTURE:{identifier}]\n[CONTEXT 0]:\n{content}\n[END CONTEXT 0]",
                },
                {"role": "user", "content": "Leave?"},
            ],
        },
    }
    answer = "23 working days" if incomplete else "23 working days; 12 calendar days"
    sample = build_sample(
        capture,
        question="Leave?",
        answer=answer,
        reference="Expected facts",
        expected_model="test-model",
        capture_id=identifier,
    )
    sample["response_sources"] = [{"title": title, "text": "23 working days; 12 calendar days"}]
    sample_path = tmp_path / "sample.json"
    sample_path.write_text(json.dumps(sample))
    evidence = {
        "schema_version": 1,
        "metric": "faithfulness",
        "status": "completed",
        "sample_sha256": hashlib.sha256(sample_path.read_bytes()).hexdigest(),
        "result": {
            "value": 1.0,
            "statements": [answer],
            "verdicts": [{"statement": answer, "verdict": 1}],
        },
        "judge_model": "test-judge",
        "judge_model_digest": "digest",
        "judge_configuration": {"think": False},
        "ragas_version": "test-version",
        "created_at": "test-time",
    }
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(json.dumps(evidence))
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "id": "leave",
                "question": "Leave?",
                "fact_patterns": {
                    "leave allowance": "23 working days",
                    "notice period": "12 calendar days",
                },
                "document_title_pattern": r"automation-[a-f0-9]{32}-company-policy\.txt",
                "source_fragments": ["23 working days", "12 calendar days"],
            }
        )
    )
    return sample_path, evidence_path, profile_path


@title("Quality report combines fact checks, source checks and faithfulness measurement")
def test_report_combines_checks_and_measurement_without_quality_threshold(tmp_path: Path) -> None:
    report = build_quality_report(*_files(tmp_path))
    assert report["status"] == "checks_passed"
    assert [d["status"] for d in report["dimensions"]] == ["passed", "passed", "measured"]
    assert report["dimensions"][2]["details"]["threshold"] is None


@title("Perfect faithfulness does not hide missing required answer facts")
def test_faithfulness_one_does_not_hide_incomplete_answer(tmp_path: Path) -> None:
    report = build_quality_report(*_files(tmp_path, incomplete=True))
    assert report["status"] == "failed"
    assert [d["status"] for d in report["dimensions"]] == ["failed", "passed", "measured"]
    with pytest.raises(AssertionError, match="notice period"):
        assertions.assert_quality_report(report)


@pytest.mark.parametrize("change", ["checksum", "score", "unfinished", "missing_claim"])
@title("Invalid judge evidence is reported as an independent quality error [{param_id}]")
def test_invalid_judge_evidence_remains_an_independent_error(tmp_path: Path, change: str) -> None:
    paths = _files(tmp_path)
    evidence = json.loads(paths[1].read_text())
    if change == "checksum":
        evidence["sample_sha256"] = "another-sample"
    elif change == "score":
        evidence["result"]["value"] = 0.5
    elif change == "unfinished":
        evidence["status"] = "error"
    else:
        evidence["result"]["verdicts"] = []
    paths[1].write_text(json.dumps(evidence))
    report = build_quality_report(*paths)
    assert report["status"] == "error"
    assert [d["status"] for d in report["dimensions"]] == ["passed", "passed", "error"]


@title("Expected source identity comes from captured context rather than response citations")
def test_expected_source_is_derived_from_context_not_citations(tmp_path: Path) -> None:
    paths = _files(tmp_path)
    sample = json.loads(paths[0].read_text())
    sample["response_sources"][0]["title"] = f"automation-{'b' * 32}-company-policy.txt"
    paths[0].write_text(json.dumps(sample))
    evidence = json.loads(paths[1].read_text())
    evidence["sample_sha256"] = hashlib.sha256(paths[0].read_bytes()).hexdigest()
    paths[1].write_text(json.dumps(evidence))
    report = build_quality_report(*paths)
    assert report["dimensions"][1]["status"] == "failed"
    assert report["dimensions"][2]["status"] == "measured"


@title("Allure renders all quality dimensions even when one check fails")
def test_allure_renders_remaining_steps_after_a_failed_dimension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = build_quality_report(*_files(tmp_path, incomplete=True))
    fake_allure = Mock()
    visited = []

    @contextmanager
    def step(name: str):
        visited.append(name)
        yield

    fake_allure.step.side_effect = step
    monkeypatch.setitem(__import__("sys").modules, "allure", fake_allure)
    with pytest.raises(AssertionError):
        present_quality_report(report)
    assert visited == [d["name"] for d in report["dimensions"]]


@title("Missing context evidence creates two visible errors without hiding other quality checks")
def test_optional_relevance_has_independent_dimensions(tmp_path: Path) -> None:
    report = build_quality_report(*_files(tmp_path), relevance_path=tmp_path / "absent.json")
    assert [d["status"] for d in report["dimensions"]] == [
        "passed",
        "passed",
        "measured",
        "error",
        "error",
    ]
    assert [d["name"] for d in report["dimensions"][-2:]] == ["context_precision", "context_recall"]


@title(
    "Gated quality reports reject omitted semantic evidence instead of silently passing measurements"
)
def test_saved_report_gates_require_all_evidence(tmp_path: Path) -> None:
    gates = Path(__file__).resolve().parents[2] / "test_data/quality-gates.json"
    report = build_quality_report(*_files(tmp_path), gates_path=gates)
    assert report["status"] == "error"
    assert report["dimensions"][2]["status"] == "passed"
    assert len(report["dimensions"]) == 6
    with pytest.raises(AssertionError, match="not supplied"):
        assertions.assert_quality_report(report)
