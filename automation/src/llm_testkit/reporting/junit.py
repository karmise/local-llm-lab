"""Read one immutable JUnit input, retaining phase outcomes and metadata conflicts."""

import hashlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

_OUTCOMES = {"failure": "failed", "skipped": "skipped", "error": "error"}
_PRIORITY = {"passed": 0, "failed": 1, "skipped": 2, "error": 3}


@dataclass
class TestCaseResult:
    classname: str
    name: str
    properties: dict[str, str] = field(default_factory=dict)
    conflicts: set[str] = field(default_factory=set)
    outcomes: set[str] = field(default_factory=set)
    details: list[dict[str, str]] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.conflicts:
            return "error"
        return max(self.outcomes or {"passed"}, key=_PRIORITY.__getitem__)


@dataclass
class JUnitReport:
    sha256: str
    cases: dict[tuple[str, str], TestCaseResult]


def read_junit(path: Path, *, identity_property: str | None = None) -> JUnitReport:
    raw = path.read_bytes()
    entries: dict[tuple[str, str], TestCaseResult] = {}
    for row in ET.fromstring(raw).iter("testcase"):
        classname, name = row.get("classname", ""), row.get("name")
        if name is None:
            raise ValueError("JUnit testcase requires a name")
        # Keep properties as a sequence: a dict would hide conflicts within a single element.
        properties = []
        for prop in row.findall("./properties/property"):
            key = prop.get("name")
            if not key:
                raise ValueError("JUnit property requires a name")
            properties.append((key, prop.get("value", "")))
        identity = (classname, name)
        if identity_property:
            node = next((v for k, v in properties if k == identity_property and v), None)
            if node:
                identity = ("", node)
        entry = entries.setdefault(identity, TestCaseResult(classname, name))
        for key, value in properties:
            if key in entry.properties and entry.properties[key] != value:
                entry.conflicts.add(key)
            else:
                entry.properties[key] = value
        for tag, outcome in _OUTCOMES.items():
            for child in row.findall(tag):
                entry.outcomes.add(outcome)
                entry.details.append({"kind": outcome, "message": child.get("message", ""), "text": child.text or ""})
    return JUnitReport(hashlib.sha256(raw).hexdigest(), entries)
