"""Environment, clients and immutable-on-disk test data."""

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from llm_testkit.clients.anythingllm_client import AnythingLLMClient
from llm_testkit.config import Settings
from llm_testkit.core.http_client import HttpClient
from llm_testkit.datasets.golden import GoldenCase, load_golden_dataset


@pytest.fixture(scope="session")
def automation_root(pytestconfig: pytest.Config) -> Path:
    return pytestconfig.rootpath


@pytest.fixture(scope="session")
def settings(automation_root: Path) -> Settings:
    key_file = automation_root.parent / ".runtime" / "anythingllm-api-key"
    return Settings.from_env(default_api_key_file=key_file)


@pytest.fixture
def http_client(settings: Settings) -> Iterator[HttpClient]:
    with HttpClient(base_url=settings.base_url, timeout=settings.http_timeout) as client:
        yield client


@pytest.fixture
def anythingllm_api(http_client: HttpClient) -> AnythingLLMClient:
    return AnythingLLMClient(http_client)


@pytest.fixture
def authenticated_anythingllm_api(http_client: HttpClient, settings: Settings) -> AnythingLLMClient:
    if not settings.api_key:
        pytest.fail(
            "Developer API key is missing. Configure ANYTHINGLLM_API_KEY or "
            "ANYTHINGLLM_API_KEY_FILE; see docs/step-03-developer-api.md.",
            pytrace=False,
        )
    return AnythingLLMClient(http_client, api_key=settings.api_key)


@pytest.fixture(scope="session")
def workspace_template(automation_root: Path) -> dict[str, Any]:
    configuration_file = automation_root.parent / "config" / "workspace.json"
    return json.loads(configuration_file.read_text(encoding="utf-8"))


@pytest.fixture
def policy_file(automation_root: Path) -> Path:
    return automation_root / "test_data" / "company-policy.txt"


@pytest.fixture(scope="session")
def paid_leave_profile_path(automation_root: Path) -> Path:
    return automation_root / "test_data" / "quality-paid-leave.json"


@pytest.fixture
def paid_leave_profile(paid_leave_profile_path: Path) -> dict[str, Any]:
    return json.loads(paid_leave_profile_path.read_text(encoding="utf-8"))


@pytest.fixture
def missing_policy_profile(automation_root: Path) -> dict[str, Any]:
    path = automation_root / "test_data" / "missing-policy.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def golden_metadata(
    automation_root: Path,
    policy_file: Path,
    golden_case: GoldenCase,
    record_property: Callable[[str, object], None],
) -> None:
    dataset = load_golden_dataset(automation_root / "test_data/golden-policy.json", policy_file)
    if golden_case not in dataset.cases:
        pytest.fail(
            "Golden dataset changed after collection; collect the suite again", pytrace=False
        )
    record_property("golden_dataset_version", dataset.version)
    record_property("golden_dataset_sha256", dataset.sha256)
    record_property("golden_case_id", golden_case.id)
    record_property("golden_category", golden_case.category)
