"""Summarize repeated RAG outcomes without treating repeats as retries."""

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

METADATA_FIELDS = (
    "model_digest",
    "policy_sha256",
    "workspace_configuration",
    "thinking_mode",
)


def summarize_report(path: Path) -> list[dict[str, Any]]:
    entries: dict[tuple[str, str], dict[str, Any]] = {}
    for case in ET.parse(path).getroot().iter("testcase"):
        key = (case.attrib.get("classname", ""), case.attrib["name"])
        entry = entries.setdefault(
            key,
            {
                "name": key[1],
                "classname": key[0],
                "properties": {},
                "failed": False,
                "errored": False,
                "skipped": False,
                "metadata_conflict": False,
            },
        )
        properties = {
            prop.attrib["name"]: prop.attrib.get("value", "")
            for prop in case.findall("./properties/property")
        }
        for field, value in properties.items():
            if field in entry["properties"] and entry["properties"][field] != value:
                entry["metadata_conflict"] = True
            entry["properties"][field] = value
        entry["failed"] |= case.find("failure") is not None
        entry["errored"] |= case.find("error") is not None
        entry["skipped"] |= case.find("skipped") is not None

    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in entries.values():
        properties = entry["properties"]
        name = entry["name"]
        fallback = (
            re.search(r"\[(.+?)(?:-run-(\d+))?\]$", name)
            if entry["classname"].endswith("test_rag")
            else None
        )
        model = properties.get("generation_model") or (fallback.group(1) if fallback else None)
        if model is None:
            continue
        scenario = name.split("[", 1)[0]
        golden_case = properties.get("golden_case_id")
        if golden_case:
            scenario += f"[{golden_case}]"
        if properties.get("bias_pair_id"):
            scenario += f"[bias={properties['bias_pair_id']}:{properties.get('bias_variant_id', 'missing')}]"
        if properties.get("adversarial_case_id"):
            scenario += f"[attack={properties['adversarial_case_id']}]"
        if properties.get("prompt_id"):
            scenario += f"[prompt={properties['prompt_id']}]"
        group = groups.setdefault(
            (scenario, model),
            {
                "scenario": scenario,
                "model": model,
                "runs": 0,
                "passed": 0,
                "failed": 0,
                "errored": 0,
                "skipped": 0,
                "metadata_complete": True,
                "fingerprints": set(),
            },
        )
        group["runs"] += 1
        failed = entry["failed"]
        errored = entry["errored"]
        skipped = entry["skipped"]
        group["failed"] += int(failed)
        group["errored"] += int(errored)
        group["skipped"] += int(skipped)
        group["passed"] += int(not (failed or errored or skipped))
        complete = not entry["metadata_conflict"] and all(
            properties.get(field) for field in METADATA_FIELDS
        )
        if golden_case:
            complete = complete and bool(properties.get("golden_dataset_sha256"))
        if properties.get("bias_pair_id"):
            complete = (
                complete
                and bool(properties.get("bias_catalog_sha256"))
                and bool(properties.get("bias_variant_id"))
            )
        if properties.get("adversarial_case_id"):
            complete = complete and bool(properties.get("adversarial_catalog_sha256"))
        if properties.get("prompt_id"):
            complete = complete and bool(properties.get("prompt_sha256"))
        group["metadata_complete"] = group["metadata_complete"] and complete
        if complete:
            configuration = json.dumps(
                json.loads(properties["workspace_configuration"]), sort_keys=True
            )
            group["fingerprints"].add(
                (
                    properties["model_digest"],
                    properties["policy_sha256"],
                    configuration,
                    properties["thinking_mode"],
                    properties.get("golden_dataset_sha256", ""),
                    properties.get("prompt_sha256", ""),
                    properties.get("adversarial_catalog_sha256", ""),
                    properties.get("bias_catalog_sha256", ""),
                )
            )

    results = []
    for group in groups.values():
        fingerprints = group.pop("fingerprints")
        group["configuration_consistent"] = group["metadata_complete"] and len(fingerprints) == 1
        group["mixed_pass_fail_observed"] = group["passed"] > 0 and group["failed"] > 0
        results.append(group)
    return sorted(results, key=lambda row: (row["scenario"], row["model"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = summarize_report(args.report)
    if not results:
        parser.error("The report contains no identifiable RAG cases")
    content = json.dumps(results, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
    print(content, end="")


if __name__ == "__main__":
    main()
