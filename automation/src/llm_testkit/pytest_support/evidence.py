"""Attach reviewed requirement identity to JUnit and optional Allure labels."""

import json
from typing import Any

import pytest

from llm_testkit.qualification.plan import load_plan, matching_requirements

PLAN = pytest.StashKey[dict[str, Any]]()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    try:
        plan = load_plan(config.rootpath / "test_data/qualification-plan.json", config.rootpath)
    except (ValueError, OSError) as error:
        raise pytest.UsageError(f"Invalid qualification plan: {error}") from error
    config.stash[PLAN] = plan
    for item in items:
        rows = matching_requirements(plan, item.nodeid)
        if rows:
            item.user_properties.extend(
                [
                    ("requirement_ids", json.dumps([r["id"] for r in rows])),
                    ("qualification_phases", json.dumps(sorted({r["phase"] for r in rows}))),
                    ("qualification_plan_sha256", plan["sha256"]),
                    ("test_node_id", item.nodeid),
                    (
                        "test_source_sha256",
                        plan["test_source_sha256"][item.nodeid.split("[", 1)[0]],
                    ),
                    ("framework_source_sha256", plan["framework_source_sha256"]),
                ]
            )
