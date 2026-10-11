"""Credential redaction: remove the known runtime key from evidence before it is hashed or uploaded."""

import hashlib
import sys
from unittest.mock import MagicMock, Mock

import pytest

from llm_testkit.evaluation import benchmark_runner as runner
from llm_testkit.reporting import redaction
from llm_testkit.reporting.junit import read_junit
from llm_testkit.reporting.redaction import redact_files
from llm_testkit.reporting.steps import title

pytestmark = pytest.mark.unit

SECRET = "DISPOSABLE-TEST-CREDENTIAL"
TRACE = f"Authorization: Bearer {SECRET}\nReadTimeout: request failed\n{SECRET}\n"
JUNIT = (
        f'<testsuite><testcase name="test_golden_policy_answer[paid_leave-qwen3.5:4b]">'
        f'<failure message="ReadTimeout">{SECRET}</failure></testcase></testsuite>')
SAMPLE = '{"answer": "Preserved evidence"}'


@pytest.fixture
def reports(tmp_path):
    """A failed run's log and JUnit containing the key, and a sample without it."""
    directory = tmp_path / "reports"
    directory.mkdir()
    (directory / "generation.log").write_text(TRACE)
    (directory / "generation.xml").write_text(JUNIT)
    (directory / "sample.json").write_text(SAMPLE)
    return directory


@title("Every occurrence of the key is replaced while diagnostics and other files are preserved")
def test_redact_files(reports):
    changed = redact_files(reports.rglob("*"), SECRET)

    assert changed == 2
    assert (reports / "generation.log").read_text() == TRACE.replace(SECRET, "[REDACTED]")
    assert (reports / "generation.xml").read_text() == JUNIT.replace(SECRET, "[REDACTED]")
    assert (reports / "sample.json").read_text() == SAMPLE


@title("Redaction is idempotent")
def test_redaction_is_idempotent(reports):
    redact_files(reports.rglob("*"), SECRET)

    assert redact_files(reports.rglob("*"), SECRET) == 0


@pytest.mark.parametrize("secret", [pytest.param(None, id="none"), pytest.param("", id="empty")])
@title("Without a known key no file is read or changed [{param_id}]")
def test_no_secret_changes_nothing(reports, secret):
    paths = MagicMock()

    assert redact_files(paths, secret) == 0
    paths.__iter__.assert_not_called()
    assert (reports / "generation.log").read_text() == TRACE


@title("Non-ASCII keys are matched by their UTF-8 bytes")
def test_redaction_uses_utf8(tmp_path):
    path = tmp_path / "log.txt"
    path.write_bytes("ключ-секрет and more".encode())

    assert redact_files([path], "ключ-секрет") == 1
    assert path.read_text(encoding="utf-8") == "[REDACTED] and more"


@title("Directories and missing paths are skipped")
def test_redaction_skips_non_files(reports, tmp_path):
    assert redact_files([reports, tmp_path / "absent.log", reports / "generation.log"], SECRET) == 1


@title("A symlink in published evidence is refused rather than modifying an external file")
def test_redaction_refuses_symlink(reports):
    linked = reports / "linked.log"
    linked.symlink_to(reports / "generation.log")

    with pytest.raises(ValueError, match="Refusing a symlink in published evidence"):
        redact_files([linked], SECRET)
    assert (reports / "generation.log").read_text() == TRACE


@title("A dangling symlink is refused too")
def test_redaction_refuses_dangling_symlink(tmp_path):
    linked = tmp_path / "linked.log"
    linked.symlink_to(tmp_path / "absent.log")

    with pytest.raises(ValueError, match="symlink"):
        redact_files([linked], SECRET)


@pytest.fixture
def failed_generation(tmp_path, monkeypatch):
    """A generation subprocess that fails with the key in its log and JUnit, and a JUnit reader that records input."""
    root = tmp_path / "automation"
    directory = tmp_path / "reports"
    directory.mkdir()
    (tmp_path / ".runtime").mkdir()
    (tmp_path / ".runtime/anythingllm-api-key").write_text(SECRET)
    monkeypatch.delenv("ANYTHINGLLM_API_KEY", raising=False)
    monkeypatch.delenv("ANYTHINGLLM_API_KEY_FILE", raising=False)

    def execute(*args, stdout, **kwargs):
        stdout.write(TRACE)
        (directory / "generation.xml").write_text(JUNIT)
        return Mock(returncode=1)

    inspected = []

    def inspect(path):
        result = read_junit(path)
        inspected.append((path.read_bytes(), result.sha256))
        return result

    monkeypatch.setattr(runner.subprocess, "run", execute)
    monkeypatch.setattr(runner, "read_junit", inspect)
    return root, directory, inspected


@title("Reports are redacted before JUnit parsing and checksum calculation, even when generation fails")
def test_generation_redacts_before_hashing(failed_generation):
    root, directory, inspected = failed_generation

    with pytest.raises(ValueError, match="no captured sample"):
        runner.generate_sample(root, directory, "paid_leave", "qwen3.5:4b")

    (content, digest), = inspected
    assert SECRET.encode() not in content and b"ReadTimeout" in content
    assert digest == hashlib.sha256(content).hexdigest()
    assert SECRET not in (directory / "generation.log").read_text()


def run_cli(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["redaction", *map(str, args)])
    return redaction.main()


@title("The command redacts every file under the directory with the key from the secret file")
def test_cli_redacts_directory(reports, tmp_path, monkeypatch, capsys):
    (reports / "nested").mkdir()
    (reports / "nested/trace.log").write_text(SECRET)
    (tmp_path / "key").write_text(f"  {SECRET}\n")

    assert run_cli(monkeypatch, reports, "--secret-file", tmp_path / "key") == 0

    assert capsys.readouterr().out == "Credential redaction completed; changed files: 3\n"
    assert (reports / "nested/trace.log").read_text() == "[REDACTED]"


@title("Without a secret file the command changes nothing")
def test_cli_without_secret_file(reports, tmp_path, monkeypatch, capsys):
    assert run_cli(monkeypatch, reports, "--secret-file", tmp_path / "absent") == 0

    assert capsys.readouterr().out == "Credential redaction completed; changed files: 0\n"
    assert (reports / "generation.log").read_text() == TRACE


@title("The command requires a secret file argument")
def test_cli_requires_secret_file(reports, monkeypatch):
    with pytest.raises(SystemExit) as exit:
        run_cli(monkeypatch, reports)

    assert exit.value.code == 2
