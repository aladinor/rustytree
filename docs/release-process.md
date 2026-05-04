# Release process

> **Status:** no releases have shipped yet. The full PyPI publish, hotfix,
> and yank workflows will be defined when the first `0.1.0` release PR is
> opened. This doc captures the minimum we need to get there.

## Versioning

Follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Pre-1.0 (`0.x.y`): `MINOR` may include breaking changes; `PATCH` should not.

## Release checklist (pre-1.0, local-only)

1. Confirm `main` is green. Look at `[Unreleased]` in
   [`CHANGELOG.md`](../CHANGELOG.md) and pick a version.
2. Open a `chore/release-vX.Y.Z` branch and bump:
   - `Cargo.toml` `version`
   - `pyproject.toml` `version`
   - `python/rustytree/__init__.py` `__version__`
3. Roll the CHANGELOG: rename `## [Unreleased]` to
   `## [X.Y.Z] - YYYY-MM-DD`, add a fresh empty `[Unreleased]` above, update
   the link refs at the bottom.
4. Run the validation chain locally (see [`contributing.md`](contributing.md)).
5. Open a release PR; merge.
6. Tag and push:
   ```bash
   git checkout main && git pull --ff-only
   git tag -s vX.Y.Z -m "Release vX.Y.Z"
   git push origin vX.Y.Z
   ```
7. Cut a GitHub release with `gh release create vX.Y.Z --notes-from-tag`.

## After v0.1.0 ships

This doc grows to cover PyPI publish (`maturin upload`), hotfix branching
off tags, and yanking. Until then, anything beyond a local tag is YAGNI.
