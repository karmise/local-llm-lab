"""Only conversational scenarios use the application's conversational profile."""

import json
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture(scope="session")
def workspace_template(automation_root: Path) -> dict[str, Any]:
    path = automation_root.parent / "config/conversation-workspace.json"
    return json.loads(path.read_text(encoding="utf-8"))
