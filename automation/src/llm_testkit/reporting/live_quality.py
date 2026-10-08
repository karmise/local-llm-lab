"""Reported operations for the opt-in captured quality scenario."""

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from requests import Response

from llm_testkit.config import Settings
from llm_testkit.evaluation.faithfulness import evaluate_sample_report
from llm_testkit.observation.evaluation_sample import write_sample
from llm_testkit.reporting.steps import attach_file, step


@step("Generate one answer and capture actual model context")
def generate_captured_answer(
        chat: Callable[[str, str], Response], profile: Mapping[str, Any], sample_path: Path) -> None:
    chat(profile["question"], profile["reference"])
    attach_file(sample_path, name="Captured evaluation sample", media_type="application/json", extension="json")


@step("Run local judge and preserve its evidence")
def evaluate_captured_answer(sample_path: Path, evidence_path: Path, *, settings: Settings, judge_model: str) -> None:
    evidence = evaluate_sample_report(sample_path, settings=settings, judge_model=judge_model)
    write_sample(evidence_path, evidence)
    attach_file(evidence_path, name="Original judge evidence", media_type="application/json", extension="json")
