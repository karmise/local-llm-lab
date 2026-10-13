"""Qualification plan: a reviewed requirement-to-test mapping validated against real test definitions."""

import hashlib
from copy import deepcopy

import pytest

from llm_testkit.qualification.plan import expected_cells, framework_checksum, load_plan, matching_requirements
from llm_testkit.reporting.steps import title
from test_support.builders.qualification import SELECTOR, qualification_lab, reviewed_plan
from test_support.paths import AUTOMATION_ROOT

pytestmark = pytest.mark.unit


@pytest.fixture
def lab(tmp_path):
    return qualification_lab(tmp_path)


@title("The committed plan maps 13 requirements to real tests with complete model and prompt matrices")
def test_committed_plan_matches_real_tests():
    plan = load_plan(AUTOMATION_ROOT / "test_data/qualification-plan.json", AUTOMATION_ROOT)

    requirements = {row["id"]: row for row in plan["requirements"]}
    assert len(requirements) == 13
    assert [len(expected_cells(requirements[i])) for i in ("REQ-GOLDEN", "REQ-PROMPT", "REQ-BIAS")] == [32, 64, 12]
    assert len([row for row in requirements.values() if row["phase"] == "IQ"]) == 3


@title("A loaded plan carries checksums of itself, the framework sources and each mapped test file")
def test_load_plan_adds_checksums(lab):
    plan = lab.plan()

    assert plan["sha256"] == hashlib.sha256(lab.plan_path.read_bytes()).hexdigest()
    assert plan["framework_source_sha256"] == framework_checksum(lab.root)
    assert plan["test_source_sha256"] == {
            SELECTOR: hashlib.sha256((lab.root / "tests/test_example.py").read_bytes()).hexdigest()}
    assert {
            k: v
            for k, v in plan.items()
            if k not in ("sha256", "framework_source_sha256", "test_source_sha256")} == reviewed_plan()


def update(**fields):
    return lambda plan: plan.update(fields)


def requirement(**fields):
    return lambda plan: plan["requirements"][0].update(fields)


@pytest.mark.parametrize(("change", "message"), [
        pytest.param(update(schema_version=True), "supported educational schema", id="boolean-schema"),
        pytest.param(update(schema_version=2), "supported educational schema", id="schema-two"),
        pytest.param(update(educational_only=False), "supported educational schema", id="not-educational"),
        pytest.param(update(version=""), "Plan requires a version", id="blank-version"),
        pytest.param(update(version=1), "Plan requires a version", id="numeric-version"),
        pytest.param(update(data_sha256=[]), "Data baselines must use file names and SHA-256", id="data-not-object"),
        pytest.param(update(data_sha256={"policy.txt": "0" * 63}), "Data baselines must use", id="short-checksum"),
        pytest.param(update(data_sha256={"../outside.txt": "0" * 64}), "Data baselines must use", id="data-escape"),
        pytest.param(update(data_sha256={"./policy.txt": "0" * 64}), "Data baselines must use", id="data-alias"),
        pytest.param(update(data_sha256={"/abs.txt": "0" * 64}), "Data baselines must use", id="data-absolute"),
        pytest.param(update(data_sha256={"a\\b.txt": "0" * 64}), "Data baselines must use", id="data-backslash"),
        pytest.param(
        update(data_sha256={"policy.txt": "0" * 64}),
        "^Plan data baseline changed; review expectations and version: policy.txt$", id="data-changed"),
        pytest.param(update(requirements=[]), "Plan requires requirements", id="no-requirements"),
        pytest.param(update(requirements={}), "Plan requires requirements", id="requirements-not-list"),
        pytest.param(update(requirements=[1]), "Requirements must be objects", id="row-not-object"),
        pytest.param(requirement(id="bad id"), "Invalid or duplicate requirement id", id="invalid-id"),
        pytest.param(
        lambda p: p["requirements"].append(deepcopy(p["requirements"][0])), "Invalid or duplicate requirement id",
        id="duplicate-id"),
        pytest.param(requirement(phase=""), "Unknown qualification phase or risk", id="blank-phase"),
        pytest.param(requirement(phase=[]), "Unknown qualification phase or risk", id="list-phase"),
        pytest.param(requirement(phase="DQ"), "Unknown qualification phase or risk", id="unknown-phase"),
        pytest.param(requirement(risk="critical"), "Unknown qualification phase or risk", id="unknown-risk"), *(
        pytest.param(requirement(**{field: " "}), "Requirement criteria must be explicit", id=f"blank-{field}")
        for field in ("description", "acceptance", "rationale")),
        pytest.param(requirement(tests=[]), "distinct test selectors", id="no-tests"),
        pytest.param(requirement(tests=[1]), "distinct test selectors", id="numeric-test"),
        pytest.param(requirement(tests=[SELECTOR, SELECTOR]), "distinct test selectors", id="duplicate-test"),
        pytest.param(
        requirement(tests=["outside.py::test_example"]), "unparameterized test function selectors", id="outside-tests"),
        pytest.param(requirement(tests=["tests/test_example.txt::test_example"]), "unparameterized", id="not-python"),
        pytest.param(requirement(tests=[SELECTOR + "[first]::x"]), "unparameterized", id="two-separators"),
        pytest.param(
        requirement(tests=["tests/../tests/test_example.py::test_example"]), "unparameterized", id="selector-alias"),
        pytest.param(
        requirement(tests=["tests/test_example.py::test_unknown"]),
        "^Requirement references an unknown test: tests/test_example.py::test_unknown$", id="unknown-test"),
        pytest.param(requirement(axes=[]), "nonempty distinct string values", id="axes-not-object"),
        pytest.param(requirement(axes={"golden_case_id": []}), "nonempty distinct string values", id="empty-axis"),
        pytest.param(
        requirement(axes={"golden_case_id": ["first", "first"]}), "nonempty distinct", id="duplicate-axis-value"),
        pytest.param(requirement(axes={"golden_case_id": [""]}), "nonempty distinct", id="blank-axis-value"),
        pytest.param(requirement(axes={"golden_case_id": "first"}), "nonempty distinct", id="axis-not-list")])
@title("A malformed, stale or unsafe plan is rejected rule by rule [{param_id}]")
def test_load_plan_rejects(lab, change, message):
    plan = reviewed_plan()
    change(plan)
    lab.write_plan(plan)

    with pytest.raises(ValueError, match=message):
        lab.plan()


@title("A plan that is not an object is rejected")
def test_load_plan_rejects_non_object(lab):
    lab.plan_path.write_text("[]")

    with pytest.raises(ValueError, match="supported educational schema"):
        lab.plan()


@title("A test selector that resolves outside the root through a link is rejected")
def test_load_plan_rejects_linked_test_outside_root(lab, tmp_path):
    (tmp_path / "outside.py").write_text("def test_example(): pass\n")
    (lab.root / "tests/linked.py").symlink_to(tmp_path / "outside.py")
    plan = reviewed_plan()
    plan["requirements"][0]["tests"] = ["tests/linked.py::test_example"]
    lab.write_plan(plan)

    with pytest.raises(ValueError, match="^Test selector escapes automation root$"):
        lab.plan()


@title("A data file that resolves outside the data root through a link is rejected")
def test_load_plan_rejects_linked_data_outside_root(lab, tmp_path):
    (tmp_path / "outside.txt").write_bytes(b"x")
    (lab.root / "test_data/linked.txt").symlink_to(tmp_path / "outside.txt")
    plan = reviewed_plan()
    plan["data_sha256"] = {"linked.txt": hashlib.sha256(b"x").hexdigest()}
    lab.write_plan(plan)

    with pytest.raises(ValueError, match="data baseline changed"):
        lab.plan()


@title("Async test functions and requirements without axes or data are valid")
def test_load_plan_accepts_minimal_requirement(lab):
    (lab.root / "tests/test_async.py").write_text("async def test_async():\n    pass\n")
    plan = reviewed_plan()
    del plan["data_sha256"]
    plan["requirements"][0] = {k: v for k, v in plan["requirements"][0].items() if k != "axes"}
    plan["requirements"][0]["tests"] = [SELECTOR, "tests/test_async.py::test_async"]
    plan["requirements"].append(plan["requirements"][0] | {"id": "REQ-IQ-1", "phase": "IQ", "risk": "low"})
    lab.write_plan(plan)

    loaded = lab.plan()

    assert sorted(loaded["test_source_sha256"]) == ["tests/test_async.py::test_async", SELECTOR]


@title("Only module-level functions count as tests; nested and class methods do not")
def test_load_plan_ignores_nested_functions(lab):
    (lab.root / "tests/test_example.py"
     ).write_text("class TestX:\n    def test_method(self): pass\n\ndef test_example():\n    def test_inner(): pass\n")
    plan = reviewed_plan()
    plan["requirements"][0]["tests"] = ["tests/test_example.py::test_inner"]
    lab.write_plan(plan)

    with pytest.raises(ValueError, match="unknown test"):
        lab.plan()


@title("A data root other than the automation tree's can be given")
def test_load_plan_with_data_root(lab, tmp_path):
    other = tmp_path / "data"
    other.mkdir()
    (other / "policy.txt").write_bytes(b"Six days\n")
    (lab.root / "test_data/policy.txt").write_text("changed")

    assert load_plan(lab.plan_path, lab.root, data_root=other)["version"] == "example-v1"


@title("Framework provenance covers sources, lock files, project settings and nested fixtures, nothing else")
def test_framework_checksum_sources(lab):
    before = framework_checksum(lab.root)
    (lab.root / "src/notes.md").write_text("ignored")
    (lab.root / "tests/test_other.py").write_text("ignored")
    unchanged = framework_checksum(lab.root)
    changes = []
    for path, content in (("src/hook.cjs", "x"), ("requirements-dev.lock", "x"), ("pyproject.toml", "x"),
            ("tests/ui/conftest.py", "x"), ("src/runtime.py", "VERSION = 2\n")):
        (lab.root / path).parent.mkdir(parents=True, exist_ok=True)
        (lab.root / path).write_text(content)
        changes.append(framework_checksum(lab.root))

    assert unchanged == before
    assert len({before, *changes}) == 6


@title("Framework provenance includes file names, so a rename changes it")
def test_framework_checksum_includes_names(lab):
    before = framework_checksum(lab.root)
    (lab.root / "src/runtime.py").rename(lab.root / "src/renamed.py")

    assert framework_checksum(lab.root) != before


@pytest.mark.parametrize(("node", "properties", "matched"), [
        pytest.param(SELECTOR + "[first]", None, True, id="parameterized-node"),
        pytest.param(SELECTOR, {"golden_case_id": "first"}, True, id="matching-axis"),
        pytest.param(SELECTOR, {"golden_case_id": "other"}, False, id="other-axis"),
        pytest.param(SELECTOR, {}, False, id="missing-axis"),
        pytest.param("tests/other.py::test_other", None, False, id="other-test")])
@title("Requirements are matched by test selector and, when given, declared axis values [{param_id}]")
def test_matching_requirements(lab, node, properties, matched):
    requirements = matching_requirements(lab.plan(), node, properties)

    assert [row["id"] for row in requirements] == (["REQ-EXAMPLE"] if matched else [])


@title("Expected cells are every selector crossed with every combination of axis values, axes sorted by name")
def test_expected_cells():
    row = {"tests": ["a::t", "b::t"], "axes": {"model": ["m1", "m2"], "case": ["c1"]}}

    assert expected_cells(row) == [("a::t", (("case", "c1"), ("model", "m1"))),
            ("a::t", (("case", "c1"), ("model", "m2"))), ("b::t", (("case", "c1"), ("model", "m1"))),
            ("b::t", (("case", "c1"), ("model", "m2")))]


@title("A requirement without axes expects one cell per selector")
def test_expected_cells_without_axes():
    assert expected_cells({"tests": ["a::t"]}) == [("a::t", ())]
