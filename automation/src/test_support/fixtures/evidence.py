"""Optional report labels for reviewed requirements."""

import json

import pytest

from llm_testkit.reporting.steps import traceability_labels


@pytest.fixture(autouse=True)
def report_requirement_labels(request: pytest.FixtureRequest) -> None:
    properties = dict(request.node.user_properties)
    if properties.get("requirement_ids"):
        traceability_labels(
            json.loads(properties["requirement_ids"]),
            json.loads(properties["qualification_phases"]),
        )
