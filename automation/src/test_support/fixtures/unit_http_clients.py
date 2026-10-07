"""Fixtures scoped to the associated unit-test module."""

from unittest.mock import Mock

import pytest
import requests


@pytest.fixture
def session(monkeypatch: pytest.MonkeyPatch) -> Mock:
    session = Mock(spec=requests.Session)
    monkeypatch.setattr(requests, "Session", lambda: session)
    return session
