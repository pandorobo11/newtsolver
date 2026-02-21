#!/usr/bin/env python3
"""Bump [project].version in pyproject.toml using SemVer."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


def parse_semver(text: str) -> tuple[int, int, int]:
    """Parse SemVer string into integer tuple."""
    m = SEMVER_RE.fullmatch(text.strip())
    if not m:
        raise ValueError(f"Invalid SemVer: '{text}'")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def bump(current: str, spec: str) -> str:
    """Return next SemVer based on bump spec."""
    major, minor, patch = parse_semver(current)
    if spec == "major":
        return f"{major + 1}.0.0"
    if spec == "minor":
        return f"{major}.{minor + 1}.0"
    if spec == "patch":
        return f"{major}.{minor}.{patch + 1}"
    parse_semver(spec)
    return spec


def update_pyproject_version(pyproject_path: Path, new_version: str) -> tuple[str, str]:
    """Replace [project].version while preserving surrounding content."""
    lines = pyproject_path.read_text(encoding="utf-8").splitlines()
    in_project = False
    old_version = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_project = stripped == "[project]"
            continue
        if in_project and stripped.startswith("version"):
            m = re.match(r'^(\s*version\s*=\s*")([^"]+)(".*)$', line)
            if not m:
                raise RuntimeError("Failed to parse [project].version line.")
            old_version = m.group(2)
            lines[i] = f'{m.group(1)}{new_version}{m.group(3)}'
            break
    if old_version is None:
        raise RuntimeError("Could not find [project].version in pyproject.toml.")
    pyproject_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return old_version, new_version


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bump newtsolver version in pyproject.toml.",
    )
    parser.add_argument(
        "target",
        help="major|minor|patch or explicit SemVer X.Y.Z",
    )
    parser.add_argument(
        "--pyproject",
        default="pyproject.toml",
        help="Path to pyproject.toml",
    )
    args = parser.parse_args(argv)

    pyproject_path = Path(args.pyproject)
    if not pyproject_path.exists():
        print(f"[ERROR] File not found: {pyproject_path}", file=sys.stderr)
        return 1

    text = pyproject_path.read_text(encoding="utf-8")
    m = re.search(r'(?ms)^\[project\].*?^\s*version\s*=\s*"([^"]+)"', text)
    if not m:
        print("[ERROR] Could not read current [project].version", file=sys.stderr)
        return 1

    current = m.group(1).strip()
    try:
        next_version = bump(current, args.target.strip())
        old_version, written_version = update_pyproject_version(pyproject_path, next_version)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    print(f"{old_version} -> {written_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
