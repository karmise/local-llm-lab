"""Write small JUnit XML reports of test runs with properties and outcomes."""

import xml.etree.ElementTree as ET
from pathlib import Path


def write_junit(path: Path, runs: list[dict], *, classname: str) -> Path:
    """Each run is {"name", "properties", "outcome"} and may override ``classname``.

    An outcome is failure, error, skipped or None for a pass.
    """
    root = ET.Element("testsuites")
    suite = ET.SubElement(root, "testsuite")
    for run in runs:
        row = ET.SubElement(suite, "testcase", classname=run.get("classname", classname), name=run["name"])
        properties = ET.SubElement(row, "properties")
        for key, value in run["properties"].items():
            ET.SubElement(properties, "property", name=key, value=value)
        if run["outcome"]:
            ET.SubElement(row, run["outcome"])
    ET.ElementTree(root).write(path)
    return path
