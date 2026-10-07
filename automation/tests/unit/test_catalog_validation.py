import json

import pytest

from test_support.builders.catalog_validation import DATA, LOADERS

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "catalog,change",
    [
        (name, change)
        for name in LOADERS
        for change in ("root", "row", "regex")
        if (name, change) != ("prompts", "regex")
    ],
)
def test_malformed_catalogs_raise_actionable_validation_errors(tmp_path, catalog, change):
    filename, rows_key, load = LOADERS[catalog]
    data = json.loads((DATA / filename).read_text())
    if change == "root":
        data = []
    elif change == "row":
        data[rows_key][0] = None
    else:
        data[rows_key][0]["forbidden_patterns"] = {"invalid": "["}
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load(path)


@pytest.mark.parametrize("catalog", LOADERS)
def test_duplicate_json_fields_are_not_silently_overwritten(tmp_path, catalog):
    filename, _, load = LOADERS[catalog]
    raw = (DATA / filename).read_text()
    path = tmp_path / "ambiguous.json"
    path.write_text('{"schema_version": 0,' + raw.lstrip()[1:])
    with pytest.raises(ValueError, match="duplicate JSON field"):
        load(path)


@pytest.mark.parametrize(
    "catalog,field",
    [
        ("bias", "golden_case_id"),
        ("adversarial", "category"),
        ("adversarial", "golden_case_id"),
        ("prompts", "baseline"),
    ],
)
def test_nonstring_lookup_fields_raise_validation_errors(tmp_path, catalog, field):
    filename, rows_key, load = LOADERS[catalog]
    data = json.loads((DATA / filename).read_text())
    (data if field == "baseline" else data[rows_key][0])[field] = []
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load(path)
