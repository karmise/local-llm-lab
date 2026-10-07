import json

import pytest

from test_support.builders.catalog_validation import (
    prepare_malformed_catalog_case,
    set_invalid_lookup_field,
)
from test_support.data.catalog_validation import (
    DATA,
    DUPLICATE_JSON_FIELDS_CATALOG_CASES,
    LOADERS,
    MALFORMED_CATALOG_CATALOG_CHANGE_CASES,
    NONSTRING_LOOKUP_CATALOG_FIELD_CASES,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "catalog,change",
    MALFORMED_CATALOG_CATALOG_CHANGE_CASES,
)
def test_malformed_catalogs_raise_actionable_validation_errors(tmp_path, catalog, change):
    filename, rows_key, load = LOADERS[catalog]
    data = json.loads((DATA / filename).read_text())
    data = prepare_malformed_catalog_case(change, data, rows_key)
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load(path)


@pytest.mark.parametrize("catalog", DUPLICATE_JSON_FIELDS_CATALOG_CASES)
def test_duplicate_json_fields_are_not_silently_overwritten(tmp_path, catalog):
    filename, _, load = LOADERS[catalog]
    raw = (DATA / filename).read_text()
    path = tmp_path / "ambiguous.json"
    path.write_text('{"schema_version": 0,' + raw.lstrip()[1:])
    with pytest.raises(ValueError, match="duplicate JSON field"):
        load(path)


@pytest.mark.parametrize(
    "catalog,field",
    NONSTRING_LOOKUP_CATALOG_FIELD_CASES,
)
def test_nonstring_lookup_fields_raise_validation_errors(tmp_path, catalog, field):
    filename, rows_key, load = LOADERS[catalog]
    data = json.loads((DATA / filename).read_text())
    set_invalid_lookup_field(data, field, rows_key)
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load(path)
