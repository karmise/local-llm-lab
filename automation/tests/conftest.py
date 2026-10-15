"""Register framework plugins without eagerly loading optional browser/judge packages."""

import pytest

pytest.register_assert_rewrite("assertions")

pytest_plugins = (
        "fixtures.ci", "fixtures.conversation", "pytester", "plugins.options", "plugins.evidence", "fixtures.evidence",
        "fixtures.quality", "fixtures.performance", "fixtures.environment", "fixtures.installation",
        "fixtures.resources", "fixtures.rag")
