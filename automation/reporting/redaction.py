"""Remove known runtime credentials before evidence is hashed or uploaded."""

import argparse
from collections.abc import Iterable
from pathlib import Path


def redact_files(paths: Iterable[Path], secret: str | None) -> int:
    """Preserve diagnostics while replacing every occurrence of the known key."""
    if not secret:
        return 0
    key = secret.encode("utf-8")
    changed = 0
    for path in paths:
        if path.is_symlink():
            raise ValueError("Refusing a symlink in published evidence")
        if not path.is_file():
            continue
        original = path.read_bytes()
        cleaned = original.replace(key, b"[REDACTED]")
        if cleaned != original:
            path.write_bytes(cleaned)
            changed += 1
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--secret-file", type=Path, required=True)
    args = parser.parse_args()
    secret = args.secret_file.read_text(encoding="utf-8").strip() if args.secret_file.is_file() else None
    changed = redact_files(args.directory.rglob("*"), secret)
    print(f"Credential redaction completed; changed files: {changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
