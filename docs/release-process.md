# Release process

Releases are tag-driven: pushing a `vX.Y.Z` tag triggers
[`.github/workflows/release.yml`](../.github/workflows/release.yml),
which builds wheels (manylinux x86_64 + manylinux aarch64 + macOS
arm64, CPython 3.12 + 3.13), builds an sdist, and attaches
everything to a GitHub Release with auto-generated notes.

PyPI publishing is enabled via trusted-publishing OIDC; every
`vX.Y.Z` tag publishes to PyPI alongside the GitHub Release. See
**Enable PyPI publishing** below for the one-time setup that's
already been done.

## Versioning

Follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Pre-1.0 (`0.x.y`): `MINOR` may include breaking changes; `PATCH`
should not.

## Cutting a release

1. Confirm `main` is green and look at `[Unreleased]` in
   [`CHANGELOG.md`](../CHANGELOG.md) to pick a version.
2. Open a `chore/release-vX.Y.Z` branch and bump the version in
   four places (must stay in sync):
   - `Cargo.toml` `[package].version`
   - `pyproject.toml` `[project].version`
   - `python/rustytree/__init__.py` `__version__`
   - `tests/test_phase1_scaffold.py::test_package_imports_with_version`
     — the hardcoded `assert rustytree.__version__ == "X.Y.Z"`. It
     stays hardcoded on purpose: reading the version from
     `pyproject.toml` would tautologically compare the package
     against itself, whereas a literal asserts the bump actually
     landed everywhere it had to.
3. Roll the CHANGELOG: rename `## [Unreleased]` to
   `## [X.Y.Z] - YYYY-MM-DD`, add a fresh empty `## [Unreleased]`
   block above it, and update the link refs at the bottom of the
   file.
4. Run the validation chain locally
   (see [`contributing.md`](contributing.md)) and merge the
   release-prep PR.
5. Tag and push from `main`:
   ```bash
   git checkout main && git pull --ff-only
   git tag -s vX.Y.Z -m "Release vX.Y.Z"
   git push origin vX.Y.Z
   ```
6. Watch the Release workflow run. Once green, edit the auto-
   generated GitHub Release notes if useful (the auto notes mostly
   list the merged PRs since the last tag).

## What the workflow builds

- **Wheels**: 6 cells (linux x86_64 + linux aarch64 + macOS arm64) ×
  (CPython 3.12 + 3.13). manylinux tag is `auto` — `maturin-action`
  selects the most compatible PEP 600 tag the build container
  supports (`manylinux_2_28` today). The aarch64 cell cross-compiles
  inside the manylinux container via QEMU and takes ~2× the wall
  time of the native x86_64 cell.
- **Sdist**: one universal `.tar.gz`. Lets people on platforms we
  don't ship wheels for (Windows, macOS Intel, linux musl) build
  from source via `pip install rustytree --no-binary rustytree`.

The `release` job is gated on `startsWith(github.ref, 'refs/tags/v')`,
so you can `workflow_dispatch` the build matrix from a release-prep
branch without cutting an actual release — useful for dry-running the
matrix before the real tag push.

## Enable PyPI publishing

PyPI publishing requires trusted-publishing setup. One-time:

1. Create the project on PyPI (`rustytree`).
2. Configure trusted publishing for the GitHub repo
   `aladinor/rustytree`, the workflow file `release.yml`, and the
   environment name `pypi`. See
   [PyPI's docs](https://docs.pypi.org/trusted-publishers/).
3. Add a `pypi` environment in repo Settings → Environments
   (no secrets required — trusted publishing uses OIDC).
4. Uncomment the `pypi-publish` job in `release.yml`.

After that, every `vX.Y.Z` tag push publishes to PyPI in addition to
the GitHub Release.

## Hotfix releases

For a `0.x.y → 0.x.(y+1)` hotfix:

1. Create `hotfix/vX.Y.(Y+1)` from the previous release tag (not
   `main`), apply the fix, follow the cut-a-release flow above.
2. Cherry-pick the fix back onto `main` if it's not already there.

## After a release

- Yanking: if a release is broken, `pip` users can avoid it via
  `gh release edit vX.Y.Z --draft` (hides the release) and, once
  PyPI is enabled, `pypi yank --version X.Y.Z` (server-side mark).
  Don't `git tag -d` a pushed tag — it makes downstream installs
  inscrutable.
- Bug reports against a released version: tag them with
  `regression-X.Y.Z` for triage.
