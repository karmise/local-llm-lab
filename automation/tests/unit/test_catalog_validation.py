"""Every reviewed JSON catalog rejects the same malformed input instead of coercing it."""

import json

import pytest

from llm_testkit.datasets.adversarial import load_adversarial_cases
from llm_testkit.datasets.bias import load_bias_cases
from llm_testkit.datasets.golden import load_golden_dataset
from llm_testkit.datasets.prompts import load_prompt_catalog
from llm_testkit.datasets.validation import require_text, validate_patterns
from llm_testkit.reporting.steps import title
from test_support.builders.golden import GOLDEN_DATASET, GOLDEN_DATASET_FILE, POLICY_FILE, TEST_DATA

pytestmark = pytest.mark.unit

# catalog: (reviewed file, key of its rows, loader, rows loaded from the reviewed file)
CATALOGS = {
        "golden": (GOLDEN_DATASET_FILE, "cases", lambda path: load_golden_dataset(path, POLICY_FILE).cases),
        "adversarial":
        (TEST_DATA / "adversarial-policy.json", "cases", lambda path: load_adversarial_cases(path, GOLDEN_DATASET)),
        "bias": (TEST_DATA / "bias-policy.json", "pairs", lambda path: load_bias_cases(path, GOLDEN_DATASET)),
        "prompts": (TEST_DATA / "prompt-variants.json", "variants", lambda path: load_prompt_catalog(path).variants)}


def load(catalog, tmp_path, *, data=None, raw=None):
    """Load ``catalog`` from ``data`` (or ``raw`` text) written to a temporary file."""
    path = tmp_path / f"{catalog}.json"
    path.write_text(raw if raw is not None else json.dumps(data))
    return CATALOGS[catalog][2](path)


def reviewed(catalog):
    return json.loads(CATALOGS[catalog][0].read_text())


def first_row(catalog, data):
    return data[CATALOGS[catalog][1]][0]


@pytest.mark.parametrize("catalog", list(CATALOGS))
@title("The reviewed {catalog} catalog is accepted")
def test_reviewed_catalog_loads(tmp_path, catalog):
    rows = load(catalog, tmp_path, data=reviewed(catalog))

    assert rows


@pytest.mark.parametrize(("catalog", "message"), [
        pytest.param("golden", "golden dataset must be an object", id="golden"),
        pytest.param("adversarial", "adversarial catalog must be an object", id="adversarial"),
        pytest.param("bias", "bias catalog must be an object", id="bias"),
        pytest.param("prompts", "prompt catalog must be an object", id="prompts")])
@title("A catalog whose root is not an object is rejected [{catalog}]")
def test_catalog_rejects_non_object_root(tmp_path, catalog, message):
    with pytest.raises(ValueError, match=message):
        load(catalog, tmp_path, data=[])


@pytest.mark.parametrize(
        "version", [
        pytest.param(None, id="missing"),
        pytest.param(2, id="two"),
        pytest.param("1", id="string"),
        pytest.param(True, id="boolean")])
@pytest.mark.parametrize("catalog", list(CATALOGS))
@title("A catalog without integer schema_version 1 is rejected [{param_id}]")
def test_catalog_rejects_unsupported_schema(tmp_path, catalog, version):
    data = reviewed(catalog)
    data["schema_version"] = version

    with pytest.raises(ValueError, match=r"schema; expected integer schema_version 1$"):
        load(catalog, tmp_path, data=data)


@pytest.mark.parametrize(("catalog", "message"), [
        pytest.param("golden", "Golden case must be an object", id="golden"),
        pytest.param("adversarial", "adversarial catalog row must be an object", id="adversarial"),
        pytest.param("bias", "bias catalog row must be an object", id="bias"),
        pytest.param("prompts", "prompt catalog row must be an object", id="prompts")])
@title("A catalog row that is not an object is rejected [{catalog}]")
def test_catalog_rejects_non_object_row(tmp_path, catalog, message):
    data = reviewed(catalog)
    data[CATALOGS[catalog][1]][0] = None

    with pytest.raises(ValueError, match=message):
        load(catalog, tmp_path, data=data)


@pytest.mark.parametrize("catalog", ["golden", "adversarial", "bias"])
@title("A forbidden pattern that is not a valid regex is rejected with its label [{catalog}]")
def test_catalog_rejects_invalid_regex(tmp_path, catalog):
    data = reviewed(catalog)
    first_row(catalog, data)["forbidden_patterns"] = {"broken": "["}

    with pytest.raises(ValueError, match=r"\.broken: invalid regex: unterminated character set"):
        load(catalog, tmp_path, data=data)


@pytest.mark.parametrize(("catalog", "context"), [
        pytest.param("golden", "golden dataset", id="golden"),
        pytest.param("adversarial", "adversarial catalog", id="adversarial"),
        pytest.param("bias", "bias catalog", id="bias"),
        pytest.param("prompts", "prompt catalog", id="prompts")])
@title("A field repeated in the JSON is rejected instead of silently overwritten [{catalog}]")
def test_catalog_rejects_duplicate_json_field(tmp_path, catalog, context):
    raw = CATALOGS[catalog][0].read_text().lstrip()
    ambiguous = '{"schema_version": 0,' + raw[1:]

    with pytest.raises(ValueError, match=f"{context}: duplicate JSON field: schema_version"):
        load(catalog, tmp_path, raw=ambiguous)


@pytest.mark.parametrize(("catalog", "field", "message"), [
        pytest.param("bias", "golden_case_id", "Unknown bias attribute or golden expectation", id="bias-case"),
        pytest.param("adversarial", "category", "Unknown attack category or golden case", id="adversarial-category"),
        pytest.param("adversarial", "golden_case_id", "Unknown attack category or golden case", id="adversarial-case")])
@title("A lookup field holding a list instead of an id is rejected, not used as a key [{param_id}]")
def test_catalog_rejects_non_string_row_lookup(tmp_path, catalog, field, message):
    data = reviewed(catalog)
    first_row(catalog, data)[field] = []

    with pytest.raises(ValueError, match=message):
        load(catalog, tmp_path, data=data)


@title("A prompt baseline holding a list instead of an id is rejected")
def test_prompt_catalog_rejects_non_string_baseline(tmp_path):
    data = reviewed("prompts")
    data["baseline"] = []

    with pytest.raises(ValueError, match="Baseline prompt is absent"):
        load("prompts", tmp_path, data=data)


@pytest.mark.parametrize(
        "value", [
        pytest.param(None, id="none"),
        pytest.param(3, id="number"),
        pytest.param("", id="empty"),
        pytest.param(" \t", id="whitespace")])
@title("Required text must be a nonempty string [{param_id}]")
def test_require_text_rejects_blank_or_non_string(value):
    with pytest.raises(ValueError, match="^field must be a nonempty string$"):
        require_text(value, "field")


@title("Required text is returned exactly as given")
def test_require_text_returns_value():
    assert require_text(" kept ", "field") == " kept "


@pytest.mark.parametrize(("patterns", "required", "message"), [
        pytest.param([], False, "^patterns must be an object$", id="not-object"),
        pytest.param({}, True, "^patterns must be a nonempty object$", id="required-empty"),
        pytest.param({" ": "x"}, False, "^patterns label must be a nonempty string$", id="blank-label"),
        pytest.param({"days": ""}, False, r"^patterns\.days must be a nonempty string$", id="blank-pattern"),
        pytest.param({"days": 3}, False, r"^patterns\.days must be a nonempty string$", id="non-string-pattern"),
        pytest.param({"days": "(a"}, False, r"^patterns\.days: invalid regex: ", id="invalid-regex"),
        pytest.param({"days": "x?"}, False, r"^patterns\.days: regex must not match empty text$", id="matches-empty")])
@title("Patterns are rejected rule by rule [{param_id}]")
def test_validate_patterns_rejects_invalid(patterns, required, message):
    with pytest.raises(ValueError, match=message):
        validate_patterns(patterns, "patterns", required=required)


@pytest.mark.parametrize(("patterns", "required"), [
        pytest.param({}, False, id="optional-empty"),
        pytest.param({
        "days": "30 days",
        "carryover": "Carry ?over"}, True, id="required")])
@title("Valid patterns are returned in their declared order [{param_id}]")
def test_validate_patterns_accepts_valid(patterns, required):
    assert validate_patterns(patterns, "patterns", required=required) == tuple(patterns.items())
