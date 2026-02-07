# Release Guide

This file defines the release operation for `fmfsolver`.

## Policy

- Versioning uses SemVer: `MAJOR.MINOR.PATCH`.
- `pyproject.toml` is the single source of truth for the package version.
- Version is bumped only for release commits.
- Git tag must match version using `vX.Y.Z`.

## Release Steps

1. Ensure working tree is clean.
2. Run tests:
   ```bash
   .venv/bin/python -m unittest discover -s tests -p 'test_*.py'
   ```
3. Bump version in `pyproject.toml`:
   ```bash
   python scripts/bump_version.py patch
   ```
   or:
   ```bash
   python scripts/bump_version.py minor
   python scripts/bump_version.py major
   python scripts/bump_version.py 1.2.3
   ```
4. Commit:
   ```bash
   git add pyproject.toml
   git commit -m "Release vX.Y.Z"
   ```
5. Create tag and push:
   ```bash
   git tag vX.Y.Z
   git push
   git push origin vX.Y.Z
   ```
6. Create GitHub Release from the tag and add release notes.

## Notes

- CI checks that tag `vX.Y.Z` matches `pyproject.toml` version.
- If you need a prerelease, use a separate branch strategy; this repository currently tracks stable SemVer releases in `pyproject.toml`.
