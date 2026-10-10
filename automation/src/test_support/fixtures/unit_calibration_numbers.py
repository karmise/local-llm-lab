"""Prepared numeric claim control scenarios, independent of model calls."""

import pytest

from test_support.builders.calibration import make_numeric_control


@pytest.fixture
def numeric_control():
    return make_numeric_control()
