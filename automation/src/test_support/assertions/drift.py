"""Domain expectations for drift unit scenarios."""

import pytest

from llm_testkit.reporting.drift import make_snapshot


def check_snapshot_assembly_outcome(arguments, case, change, path):
    if change != "none":
        with pytest.raises(ValueError):
            make_snapshot(path, **arguments)
    else:
        row = make_snapshot(path, **arguments)
        assert row["configuration"]["openAiPrompt"] == "Policy"
        assert row["case_id"] == case.id
        assert set(row["evidence_sha256"]) == {"faithfulness", "correctness", "relevance"}
