"""Scenario data builders and deterministic test doubles."""

from test_support.data.catalog_validation import DATA as DATA
from test_support.data.catalog_validation import DATASET as DATASET
from test_support.data.catalog_validation import LOADERS as LOADERS


def prepare_malformed_catalog_case(change, data, rows_key):
    if change == "root":
        data = []
    elif change == "row":
        data[rows_key][0] = None
    else:
        data[rows_key][0]["forbidden_patterns"] = {"invalid": "["}

    return data


def prepare_nonstring_lookup_step_3(data, field, rows_key):
    (data if field == "baseline" else data[rows_key][0])[field] = []
