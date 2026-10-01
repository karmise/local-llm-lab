"""Render independent quality dimensions as Allure steps and attachments."""

import json
from pathlib import Path
from typing import Any

from llm_testkit import assertions
from llm_testkit.reporting.steps import attach_file


def present_quality_report(report: dict[str, Any], *, evidence_path: Path | None = None) -> None:
    import allure

    if evidence_path is not None:
        attach_file(evidence_path, name="Original judge evidence", media_type="application/json", extension="json")
    allure.dynamic.title(f"RAG quality: {report['scenario']}")
    allure.dynamic.feature("RAG quality")
    allure.dynamic.story("Live answer evaluation" if report.get("execution_mode") == "live" else "Saved answer evaluation")
    allure.dynamic.description(report["interpretation"])
    allure.dynamic.parameter("Generation model", report["generation_model"])
    allure.dynamic.parameter("Sample SHA256", report["sample_sha256"])
    for name, value in (
        ("Question", report["question"]), ("Final answer", report["answer"]),
        ("Actual model contexts", report["contexts"]), ("Response sources", report["sources"]),
        ("Run metadata", report["metadata"]), ("Quality summary", report),
    ):
        allure.attach(json.dumps(value, indent=2, ensure_ascii=False), name=name, attachment_type=allure.attachment_type.JSON)
    for dimension in report["dimensions"]:
        try:
            with allure.step(dimension["name"]):
                allure.attach(json.dumps(dimension, indent=2), name="Dimension result", attachment_type=allure.attachment_type.JSON)
                assertions.assert_quality_dimension(dimension)
        except AssertionError:
            # Keep all independent steps visible; fail the test after rendering.
            continue
    assertions.assert_quality_report(report)
