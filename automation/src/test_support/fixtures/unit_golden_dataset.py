"""Golden-catalog observations supplied separately from assertion steps."""

from collections import Counter

import pytest

from test_support.data.golden_dataset import DATASET


@pytest.fixture
def golden_category_counts() -> Counter[str]:
    return Counter(case.category for case in DATASET.cases)
