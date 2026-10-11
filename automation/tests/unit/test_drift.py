"""Quality history: sealed snapshots, immutable recording and explicit baseline comparison."""

import hashlib
import json
import sys
from copy import deepcopy
from datetime import datetime
from types import SimpleNamespace

import pytest

from llm_testkit.reporting import drift
from llm_testkit.reporting.drift import (
        compare_snapshots, digest, make_snapshot, record_snapshot, seal_snapshot, validate_snapshot)
from llm_testkit.reporting.gates import METRICS
from llm_testkit.reporting.steps import title
from test_support.builders.golden import (
        CAPTURE_ID, GOLDEN_DATASET, GOLDEN_DATASET_FILE, POLICY_FILE, TEST_DATA, make_paid_leave_sample)

pytestmark = pytest.mark.unit

JUDGED = ("faithfulness", "correctness", "relevance")
PROMPT = "Policy"


def snapshot(sample: str = "a", **changes) -> dict:
    """A sealed snapshot of one captured sample; checksums follow the configuration and judges unless given."""
    payload = {
            "schema_version": 1,
            "kind": "quality_snapshot",
            "run_id": sample,
            "recorded_at": "2026-10-06T00:00:00+00:00",
            "case_id": "paid_leave",
            "sample_sha256": sample * 64,
            "model": "model",
            "model_digest": "b" * 64,
            "policy_sha256": "c" * 64,
            "dataset_sha256": "d" * 64,
            "configuration": {
            "chatModel": "model",
            "openAiPrompt": PROMPT,
            "topN": 4},
            "judges": {
            name: {
            "judge_model": "judge",
            "judge_model_digest": "f" * 64,
            "judge_configuration": {
            "think": False},
            "ragas_version": "test"}
            for name in JUDGED},
            "evidence_sha256": dict.fromkeys(JUDGED, "0" * 64),
            "thinking_mode": "default",
            "context_parser": "test",
            "metrics": dict.fromkeys(METRICS, 1.0),
            "acceptance": {
            "facts": "passed",
            "sources": "passed"},
            **changes}
    payload.setdefault("prompt_sha256", hashlib.sha256(payload["configuration"]["openAiPrompt"].encode()).hexdigest())
    payload.setdefault("judge_sha256", digest(payload["judges"]))
    return seal_snapshot(payload)


def judges(**changes) -> dict:
    rows = deepcopy(snapshot()["judges"])
    rows["faithfulness"].update(changes)
    return rows


def configuration(**changes) -> dict:
    return {"chatModel": "model", "openAiPrompt": PROMPT, "topN": 4, **changes}


@title("Sealing adds a checksum of the payload and is idempotent")
def test_seal_snapshot():
    row = snapshot()
    payload = {k: v for k, v in row.items() if k != "snapshot_sha256"}

    assert row["snapshot_sha256"] == hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    assert seal_snapshot(row) == row


@title("Digests reject values that JSON cannot represent exactly")
def test_digest_rejects_nan():
    with pytest.raises(ValueError, match="Out of range float values are not JSON compliant"):
        digest({"value": float("nan")})


@title("A valid sealed snapshot is accepted and returned unchanged")
def test_validate_accepts_snapshot():
    row = snapshot()

    assert validate_snapshot(row) is row


@title("A snapshot edited after sealing fails its integrity check")
def test_validate_rejects_edited_snapshot():
    row = snapshot()
    row["metrics"]["faithfulness"] = 0.0

    with pytest.raises(ValueError, match="Snapshot integrity checksum mismatch"):
        validate_snapshot(row)


def without(field):
    return lambda row: row.pop(field)


def update(**fields):
    return lambda row: row.update(fields)


def nested(field, key, value):
    return lambda row: row[field].__setitem__(key, value)


@pytest.mark.parametrize(("corrupt", "message"), [
        pytest.param(update(schema_version=2), "Unsupported history snapshot", id="schema-two"),
        pytest.param(update(schema_version=True), "Unsupported history snapshot", id="schema-boolean"),
        pytest.param(without("schema_version"), "Unsupported history snapshot", id="no-schema"),
        pytest.param(update(kind="report"), "Unsupported history snapshot", id="other-kind"),
        pytest.param(lambda row: row["metrics"].pop("faithfulness"), "all four measured metrics", id="missing-metric"),
        pytest.param(nested("metrics", "accuracy", 1.0), "all four measured metrics", id="extra-metric"),
        pytest.param(nested("metrics", "faithfulness", True), "finite quality score", id="boolean-metric"),
        pytest.param(nested("metrics", "faithfulness", 1.5), "finite quality score", id="metric-above-one"), *(
        pytest.param(update(**{field: value}), f"Missing or invalid history provenance: {field}", id=f"{field}-{name}")
        for field in
        ("sample_sha256", "model_digest", "policy_sha256", "dataset_sha256", "prompt_sha256", "judge_sha256")
        for name, value in (("empty", ""), ("upper", "B" * 64), ("short", "b" * 63))),
        pytest.param(without("model_digest"), "provenance: model_digest", id="no-model-digest"), *(
        pytest.param(update(**{field: " "}), f"Missing history identity: {field}", id=f"blank-{field}")
        for field in ("run_id", "recorded_at", "case_id", "model", "thinking_mode", "context_parser")),
        pytest.param(update(context_parser=1), "Missing history identity: context_parser", id="numeric-parser"),
        pytest.param(without("recorded_at"), "Missing history identity: recorded_at", id="no-recorded-at"),
        pytest.param(update(run_id="a.b"), "run_id must be a single safe file identifier", id="unsafe-run-id"),
        pytest.param(update(configuration=[]), "Workspace configuration must be an object", id="config-not-object"),
        pytest.param(
        update(configuration={"chatModel": "model"}), "bind to the generation model and prompt", id="no-prompt"),
        pytest.param(
        update(configuration=configuration(chatModel="other")), "bind to the generation model and prompt",
        id="other-chat-model"),
        pytest.param(
        update(configuration=configuration(openAiPrompt="Changed")), "prompt checksum differs from configuration",
        id="stale-prompt-checksum"),
        pytest.param(update(judges=[]), "all three judge configurations", id="judges-not-object"),
        pytest.param(lambda row: row["judges"].pop("relevance"), "all three judge configurations", id="missing-judge"),
        pytest.param(update(judges=judges(judge_model="other")), "judge checksum differs", id="stale-judge-checksum"),
        *(
        pytest.param(
        update(judges=judges(**{field: " "}), judge_sha256=digest(judges(**{field: " "}))),
        "Incomplete history judge provenance", id=f"blank-{field}")
        for field in ("judge_model", "judge_model_digest", "ragas_version")),
        pytest.param(
        update(judges=judges(judge_configuration=None), judge_sha256=digest(judges(judge_configuration=None))),
        "Incomplete history judge provenance", id="no-judge-configuration"),
        pytest.param(update(evidence_sha256={}), "checksums for all measured evidence", id="no-evidence"),
        pytest.param(update(evidence_sha256=[]), "checksums for all measured evidence", id="evidence-not-object"),
        pytest.param(
        nested("evidence_sha256", "relevance", "0"), "checksums for all measured evidence", id="short-evidence"),
        pytest.param(
        nested("evidence_sha256", "relevance", 0), "checksums for all measured evidence", id="numeric-evidence"),
        pytest.param(nested("acceptance", "facts", "error"), "Incomplete deterministic acceptance", id="facts-error"),
        pytest.param(lambda row: row["acceptance"].pop("sources"), "facts and source outcomes", id="no-sources"),
        pytest.param(nested("acceptance", "tone", "passed"), "facts and source outcomes", id="extra-outcome")])
@title("History rejects an incomplete or inconsistent snapshot even when resealed [{param_id}]")
def test_validate_rejects_resealed_snapshot(corrupt, message):
    row = snapshot()
    corrupt(row)

    with pytest.raises((ValueError, AssertionError), match=message):
        validate_snapshot(seal_snapshot(row))


@title("A failed deterministic outcome is valid history and the prompt checksum excludes the capture marker")
def test_validate_accepts_failed_outcome_and_capture_marker():
    marked = configuration(openAiPrompt=f"{PROMPT}\n[LLM_TESTKIT_CAPTURE:{CAPTURE_ID}]")

    row = snapshot(
            acceptance={
            "facts": "failed",
            "sources": "failed"}, configuration=marked, prompt_sha256=hashlib.sha256(PROMPT.encode()).hexdigest())

    assert validate_snapshot(row) is row


@title("Run ids may use letters of either case, digits, underscores and hyphens")
def test_validate_accepts_safe_run_id():
    row = snapshot(run_id="Nightly_Run-7")

    assert validate_snapshot(row) is row


@title("A recorded snapshot is written once under its run id")
def test_record_snapshot_writes_file(tmp_path):
    row = snapshot()

    path = record_snapshot(tmp_path, row)

    assert path == tmp_path / "a.json"
    assert json.loads(path.read_text()) == row


@title("An identical captured answer is not a new time-series observation")
def test_record_rejects_already_recorded_sample(tmp_path):
    record_snapshot(tmp_path, snapshot())

    with pytest.raises(ValueError, match="already recorded; it is not a new observation"):
        record_snapshot(tmp_path, snapshot(run_id="other"))
    assert sorted(path.name for path in tmp_path.iterdir()) == ["a.json"]


@title("A different captured answer is recorded beside existing history")
def test_record_appends_new_sample(tmp_path):
    record_snapshot(tmp_path, snapshot())

    record_snapshot(tmp_path, snapshot("b"))

    assert sorted(path.name for path in tmp_path.iterdir()) == ["a.json", "b.json"]


@title("Recording refuses to add to history that contains an edited snapshot")
def test_record_validates_existing_history(tmp_path):
    edited = snapshot()
    edited["metrics"]["faithfulness"] = 0.0
    (tmp_path / "a.json").write_text(json.dumps(edited))

    with pytest.raises(ValueError, match="integrity checksum mismatch"):
        record_snapshot(tmp_path, snapshot("b"))


@title("A run id that is already used is never overwritten by another sample")
def test_record_refuses_to_overwrite_run(tmp_path):
    original = record_snapshot(tmp_path, snapshot()).read_text()

    with pytest.raises(FileExistsError):
        record_snapshot(tmp_path, snapshot("b", run_id="a"))
    assert (tmp_path / "a.json").read_text() == original


@pytest.mark.parametrize("run_id", ["../escaped", "/absolute", "nested/file", ".."])
@title("A run id cannot write outside the history directory [{run_id}]")
def test_record_rejects_escaping_run_id(tmp_path, run_id):
    with pytest.raises(ValueError, match="single safe file identifier"):
        record_snapshot(tmp_path / "history", snapshot(run_id=run_id))
    assert not (tmp_path / "escaped.json").exists()


@title("A comparison with unchanged conditions and metrics passes and reports every delta")
def test_comparison_passes():
    baseline, current = snapshot(), snapshot("b")

    report = compare_snapshots(baseline, current)

    assert report == {
            "schema_version": 1,
            "status": "passed",
            "baseline_run": "a",
            "current_run": "b",
            "baseline_sha256": baseline["snapshot_sha256"],
            "current_sha256": current["snapshot_sha256"],
            "maximum_drop": 0.05,
            "deltas": dict.fromkeys(sorted(METRICS), 0.0),
            "dropped_metrics": [],
            "failed_acceptance": [],
            "incomparable_fields": [],
            "generation_model_changed": False,
            "interpretation": report["interpretation"]}
    assert "not statistical drift detection" in report["interpretation"]


@pytest.mark.parametrize(("value", "status"), [
        pytest.param(0.95, "passed", id="at-allowed-drop"),
        pytest.param(0.949, "regression", id="just-beyond"),
        pytest.param(1.0, "passed", id="unchanged")])
@title("The allowed drop is inclusive and small regressions are not rounded away [{param_id}]")
def test_comparison_drop_boundary(value, status):
    report = compare_snapshots(snapshot(), snapshot("b", metrics=dict.fromkeys(METRICS, value)))

    assert report["status"] == status


@title("A metric drop beyond the allowance is a regression that names the metric and its delta")
def test_comparison_reports_dropped_metric():
    current = snapshot("b", metrics=dict.fromkeys(METRICS, 1.0) | {"context_recall": 0.8})

    report = compare_snapshots(snapshot(), current)

    assert (report["status"], report["dropped_metrics"]) == ("regression", ["context_recall"])
    assert report["deltas"]["context_recall"] == pytest.approx(-0.2)


@title("A metric improvement is not a regression")
def test_comparison_allows_improvement():
    baseline = snapshot(metrics=dict.fromkeys(METRICS, 0.5))

    report = compare_snapshots(baseline, snapshot("b"))

    assert (report["status"], report["deltas"]["faithfulness"]) == ("passed", 0.5)


@title("A custom allowed drop is applied and reported")
def test_comparison_custom_maximum_drop():
    current = snapshot("b", metrics=dict.fromkeys(METRICS, 0.8))

    report = compare_snapshots(snapshot(), current, maximum_drop=0.2)

    assert (report["status"], report["maximum_drop"]) == ("passed", 0.2)


@pytest.mark.parametrize("maximum_drop", [pytest.param(-0.1, id="negative"), pytest.param(1.5, id="above-one")])
@title("The allowed drop must itself be a score between 0 and 1 [{param_id}]")
def test_comparison_rejects_invalid_maximum_drop(maximum_drop):
    with pytest.raises(AssertionError, match="finite quality score"):
        compare_snapshots(snapshot(), snapshot("b"), maximum_drop=maximum_drop)


@pytest.mark.parametrize(
        "outcomes", [
        pytest.param({
        "facts": "failed",
        "sources": "passed"}, id="facts"),
        pytest.param({
        "facts": "passed",
        "sources": "failed"}, id="sources")])
@title("A failed deterministic check in the current run is a regression [{param_id}]")
def test_comparison_reports_failed_acceptance(outcomes):
    report = compare_snapshots(snapshot(), snapshot("b", acceptance=outcomes))

    assert (report["status"],
            report["failed_acceptance"]) == ("regression", [k for k, v in outcomes.items() if v == "failed"])


@pytest.mark.parametrize(("changes", "fields"), [
        pytest.param({"case_id": "carryover_limit"}, ["case_id"], id="case"),
        pytest.param({"policy_sha256": "1" * 64}, ["policy_sha256"], id="policy"),
        pytest.param({"dataset_sha256": "1" * 64}, ["dataset_sha256"], id="dataset"),
        pytest.param({"configuration": configuration(openAiPrompt="Changed")}, ["prompt_sha256"], id="prompt"),
        pytest.param({"thinking_mode": "think"}, ["thinking_mode"], id="thinking"),
        pytest.param({"context_parser": "other"}, ["context_parser"], id="parser"),
        pytest.param({"judges": judges(judge_model="other")}, ["judge_sha256"], id="judge"),
        pytest.param({"configuration": configuration(topN=2)}, ["retrieval_configuration"], id="retrieval")])
@title("Changed evaluation conditions make runs incomparable instead of a regression [{param_id}]")
def test_comparison_reports_changed_conditions(changes, fields):
    report = compare_snapshots(snapshot(), snapshot("b", metrics=dict.fromkeys(METRICS, 0.0), **changes))

    assert (report["status"], report["incomparable_fields"]) == ("incomparable", fields)


@title("A changed generation model is compared and flagged, not treated as changed conditions")
def test_comparison_flags_generation_model_change():
    current = snapshot("b", model="new", configuration=configuration(chatModel="new"))

    report = compare_snapshots(snapshot(), current)

    assert (report["status"], report["generation_model_changed"]) == ("passed", True)


@title("A changed model digest alone is flagged as a generation model change")
def test_comparison_flags_model_digest_change():
    report = compare_snapshots(snapshot(), snapshot("b", model_digest="2" * 64))

    assert report["generation_model_changed"] is True


@pytest.mark.parametrize(("current", "status"), [
        pytest.param({}, "duplicate", id="same-sample"),
        pytest.param({"acceptance": {
        "facts": "failed",
        "sources": "passed"}}, "duplicate", id="duplicate-outranks-regression"),
        pytest.param({"thinking_mode": "think"}, "incomparable", id="incomparable-outranks-duplicate")])
@title("A second snapshot of the same captured sample is a duplicate, not a new observation [{param_id}]")
def test_comparison_of_same_sample(current, status):
    assert compare_snapshots(snapshot(), snapshot(**current))["status"] == status


@pytest.mark.parametrize("side", ["baseline", "current"])
@title("Both compared snapshots are validated [{side}]")
def test_comparison_validates_both_snapshots(side):
    edited = snapshot()
    edited["metrics"]["faithfulness"] = 0.0
    rows = {"baseline": snapshot(), "current": snapshot("b")} | {side: edited}

    with pytest.raises(ValueError, match="integrity"):
        compare_snapshots(rows["baseline"], rows["current"])


JUDGE = {"judge_model": "judge", "judge_model_digest": "e" * 64, "judge_configuration": {}, "ragas_version": "test"}


@pytest.fixture
def assembly(tmp_path, monkeypatch):
    """A captured paid-leave sample, judge evidence and a stubbed quality report built from them."""
    sample = make_paid_leave_sample()
    sample["metadata"] = {
            "policy_sha256":
            GOLDEN_DATASET.policy_sha256,
            "model_digest":
            "b" * 64,
            "thinking_mode":
            "default",
            "workspace_configuration":
            configuration(
            chatModel=sample["observation"]["request"]["model"],
            openAiPrompt=f"{PROMPT}\n[LLM_TESTKIT_CAPTURE:{CAPTURE_ID}]")}
    sample_path = tmp_path / "sample.json"
    sample_path.write_text(json.dumps(sample))
    evidence = {name: tmp_path / f"{name}.json" for name in JUDGED}
    for name, path in evidence.items():
        path.write_text(json.dumps(JUDGE | {"judge_model": f"{name}-judge"}))
    report = {
            "status":
            "checks_passed",
            "generation_model":
            sample["observation"]["request"]["model"],
            "sample_sha256":
            hashlib.sha256(sample_path.read_bytes()).hexdigest(),
            "golden_dataset_sha256":
            GOLDEN_DATASET.sha256,
            "evidence_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in evidence.items()},
            "dimensions": [{
            "status": "passed"}, {
            "status": "failed"}, *({
            "metric": metric,
            "details": {
            "value": 0.5}} for metric in sorted(METRICS)), {
            "metric": "answer_relevancy",
            "details": {
            "value": 0.1}}]}
    calls = []

    def build_quality_report(*args, **kwargs):
        calls.append((args, kwargs))
        return report

    monkeypatch.setattr(drift, "build_quality_report", build_quality_report)
    arguments = {
            **evidence, "profile": TEST_DATA / "quality-paid-leave.json",
            "dataset_path": GOLDEN_DATASET_FILE,
            "policy": POLICY_FILE,
            "case_id": "paid_leave"}

    def edit_sample(**metadata):
        sample["metadata"].update(metadata)
        sample_path.write_text(json.dumps(sample))

    return SimpleNamespace(
            make=lambda: make_snapshot(sample_path, **arguments), edit_sample=edit_sample, sample=sample,
            sample_path=sample_path, evidence=evidence, report=report, calls=calls)


@title("A snapshot binds measured evidence to the captured configuration and reviewed dataset")
def test_make_snapshot(assembly):
    row = assembly.make()

    assert validate_snapshot(row) is row
    assert len(row["run_id"]) == 32 and datetime.fromisoformat(row["recorded_at"]).utcoffset().total_seconds() == 0
    assert (row["case_id"], row["model"],
            row["model_digest"]) == ("paid_leave", assembly.report["generation_model"], "b" * 64)
    assert (row["sample_sha256"], row["policy_sha256"], row["dataset_sha256"]) == (
            assembly.report["sample_sha256"], GOLDEN_DATASET.policy_sha256, GOLDEN_DATASET.sha256)
    assert row["configuration"]["openAiPrompt"] == PROMPT
    assert row["prompt_sha256"] == hashlib.sha256(PROMPT.encode()).hexdigest()
    assert (row["thinking_mode"], row["context_parser"]) == ("default", assembly.sample["context_parser"])
    assert row["judges"] == {name: JUDGE | {"judge_model": f"{name}-judge"} for name in JUDGED}
    assert row["evidence_sha256"] == assembly.report["evidence_sha256"]
    assert row["metrics"] == dict.fromkeys(METRICS, 0.5)
    assert row["acceptance"] == {"facts": "passed", "sources": "failed"}


@title("The quality report is built from all supplied evidence, dataset and policy")
def test_make_snapshot_builds_report_from_inputs(assembly):
    assembly.make()

    (args, kwargs), = assembly.calls
    assert args == (assembly.sample_path, assembly.evidence["faithfulness"], TEST_DATA / "quality-paid-leave.json")
    assert kwargs == {
            "correctness_path": assembly.evidence["correctness"],
            "relevance_path": assembly.evidence["relevance"],
            "golden_dataset_path": GOLDEN_DATASET_FILE,
            "policy_file": POLICY_FILE}


@title("Incomplete or invalid quality evidence cannot enter history")
def test_make_snapshot_rejects_error_report(assembly):
    assembly.report["status"] = "error"

    with pytest.raises(ValueError, match="Incomplete or invalid quality evidence cannot enter metric history"):
        assembly.make()


@title("A failed quality report still enters history as an observation")
def test_make_snapshot_accepts_failed_report(assembly):
    assembly.report["status"] = "failed"

    assert assembly.make()["acceptance"]["sources"] == "failed"


@pytest.mark.parametrize("field", ["sample_sha256", "golden_dataset_sha256"])
@title("A sample or dataset that changed after the report was built is rejected [{field}]")
def test_make_snapshot_rejects_changed_sample_or_dataset(assembly, field):
    assembly.report[field] = "0" * 64

    with pytest.raises(ValueError, match="Sample or dataset changed during history assembly"):
        assembly.make()


@title("A sample edited after the report was built is rejected")
def test_make_snapshot_rejects_edited_sample(assembly):
    assembly.edit_sample(thinking_mode="changed")

    with pytest.raises(ValueError, match="Sample or dataset changed during history assembly"):
        assembly.make()


@title("A sample captured against another policy is rejected")
def test_make_snapshot_rejects_other_policy(assembly):
    assembly.edit_sample(policy_sha256="changed")
    assembly.report["sample_sha256"] = hashlib.sha256(assembly.sample_path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="policy_sha256 does not match golden provenance"):
        assembly.make()


@title("A sample without a captured policy checksum is rejected")
def test_make_snapshot_requires_captured_policy(assembly):
    del assembly.sample["metadata"]["policy_sha256"]
    assembly.sample_path.write_text(json.dumps(assembly.sample))
    assembly.report["sample_sha256"] = hashlib.sha256(assembly.sample_path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="^Captured policy does not match reviewed dataset$"):
        assembly.make()


@title("A captured configuration for another chat model is rejected")
def test_make_snapshot_rejects_other_chat_model(assembly):
    assembly.edit_sample(workspace_configuration=configuration(chatModel="other"))
    assembly.report["sample_sha256"] = hashlib.sha256(assembly.sample_path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="Recorded generation configuration mismatch"):
        assembly.make()


@pytest.mark.parametrize("name", JUDGED)
@title("Judge evidence that changed after the report was built is rejected [{name}]")
def test_make_snapshot_rejects_changed_evidence(assembly, name):
    path = assembly.evidence[name]
    path.write_text(path.read_text() + "\n")

    with pytest.raises(ValueError, match=f"^Judge evidence changed during history assembly: {name}$"):
        assembly.make()


@title("A sample for another golden case is rejected")
def test_make_snapshot_rejects_other_case(assembly):
    with pytest.raises(ValueError, match="Sample question/reference does not match the golden case"):
        make_snapshot(
                assembly.sample_path, **assembly.evidence, profile=TEST_DATA / "quality-paid-leave.json",
                dataset_path=GOLDEN_DATASET_FILE, policy=POLICY_FILE, case_id="carryover_limit")


def run_cli(monkeypatch, *args) -> int:
    monkeypatch.setattr(sys, "argv", ["drift", *map(str, args)])
    return drift.main()


@pytest.mark.parametrize(("current", "code", "status"), [
        pytest.param(snapshot("b"), 0, "passed", id="passed"),
        pytest.param(snapshot("b", metrics=dict.fromkeys(METRICS, 0.5)), 1, "regression", id="regression"),
        pytest.param(snapshot("b", thinking_mode="think"), 1, "incomparable", id="incomparable")])
@title("The compare command writes the report and exits non-zero unless the comparison passed [{param_id}]")
def test_cli_compare(tmp_path, monkeypatch, capsys, current, code, status):
    (tmp_path / "base.json").write_text(json.dumps(snapshot()))
    (tmp_path / "current.json").write_text(json.dumps(current))

    exit_code = run_cli(
            monkeypatch, "compare", tmp_path / "base.json", tmp_path / "current.json", "--output",
            tmp_path / "out.json")

    assert exit_code == code
    assert json.loads((tmp_path / "out.json").read_text())["status"] == status
    assert capsys.readouterr().out == f"History comparison: {status}\n"


@title("The compare command passes the allowed drop through")
def test_cli_compare_maximum_drop(tmp_path, monkeypatch):
    (tmp_path / "base.json").write_text(json.dumps(snapshot()))
    (tmp_path / "current.json").write_text(json.dumps(snapshot("b", metrics=dict.fromkeys(METRICS, 0.8))))

    exit_code = run_cli(
            monkeypatch, "compare", tmp_path / "base.json", tmp_path / "current.json", "--maximum-drop", "0.2",
            "--output", tmp_path / "out.json")

    assert (exit_code, json.loads((tmp_path / "out.json").read_text())["maximum_drop"]) == (0, 0.2)


@title("The compare command refuses to overwrite an existing report")
def test_cli_compare_refuses_existing_output(tmp_path, monkeypatch, capsys):
    (tmp_path / "out.json").write_text("{}")

    with pytest.raises(SystemExit) as exit:
        run_cli(monkeypatch, "compare", tmp_path / "a.json", tmp_path / "b.json", "--output", tmp_path / "out.json")

    assert exit.value.code == 2
    assert "Output already exists" in capsys.readouterr().err
    assert (tmp_path / "out.json").read_text() == "{}"


@title("The compare command requires an output path")
def test_cli_compare_requires_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as exit:
        run_cli(monkeypatch, "compare", "a.json", "b.json")

    assert exit.value.code == 2


@title("The record command assembles a snapshot from the given files and prints where it was recorded")
def test_cli_record(tmp_path, monkeypatch, capsys, assembly):
    history = tmp_path / "history"
    history.mkdir()

    exit_code = run_cli(
            monkeypatch, "record", "--sample", assembly.sample_path, "--faithfulness",
            assembly.evidence["faithfulness"], "--correctness", assembly.evidence["correctness"], "--relevance",
            assembly.evidence["relevance"], "--profile", TEST_DATA / "quality-paid-leave.json", "--dataset",
            GOLDEN_DATASET_FILE, "--policy", POLICY_FILE, "--case", "paid_leave", "--history", history)

    recorded, = history.iterdir()
    assert exit_code == 0
    assert capsys.readouterr().out == f"{recorded}\n"
    assert json.loads(recorded.read_text())["case_id"] == "paid_leave"


@title("The record command defaults to the reviewed test data and the reports history directory")
def test_cli_record_defaults(tmp_path, monkeypatch, assembly):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "reports/history").mkdir(parents=True)
    (tmp_path / "test_data").symlink_to(TEST_DATA)

    run_cli(
            monkeypatch, "record", "--sample", assembly.sample_path, "--faithfulness",
            assembly.evidence["faithfulness"], "--correctness", assembly.evidence["correctness"], "--relevance",
            assembly.evidence["relevance"])

    (args, kwargs), = assembly.calls
    assert args[2] == drift.Path("test_data/quality-paid-leave.json")
    assert (kwargs["golden_dataset_path"], kwargs["policy_file"]) == (
            drift.Path("test_data/golden-policy.json"), drift.Path("test_data/company-policy.txt"))
    recorded, = (tmp_path / "reports/history").iterdir()
    assert json.loads(recorded.read_text())["case_id"] == "paid_leave"


RECORD_ARGUMENTS = {
        "--sample": "s.json",
        "--faithfulness": "f.json",
        "--correctness": "c.json",
        "--relevance": "r.json"}


@pytest.mark.parametrize("missing", list(RECORD_ARGUMENTS))
@title("The record command requires the sample and all three judge reports [{missing}]")
def test_cli_record_requires_inputs(tmp_path, monkeypatch, capsys, missing):
    monkeypatch.chdir(tmp_path)
    arguments = [part for flag, value in RECORD_ARGUMENTS.items() if flag != missing for part in (flag, value)]

    with pytest.raises(SystemExit) as exit:
        run_cli(monkeypatch, "record", *arguments)

    assert exit.value.code == 2
    assert f"the following arguments are required: {missing}" in capsys.readouterr().err


@title("A command is required")
def test_cli_requires_command(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as exit:
        run_cli(monkeypatch)

    assert exit.value.code == 2
