# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Until the first tagged release, every PR appends to `[Unreleased]`. On
release, that section is renamed to `[x.y.z] - YYYY-MM-DD` and a fresh
`[Unreleased]` block is started.

## [Unreleased]

### Added

- Async metadata walk for a single Zarr v3 group on local filesystem ([#7]):
  `src/store.rs` builds an `AsyncReadableListableStorage` from a path via
  `zarrs_object_store` + `LocalFileSystem`. `src/walk.rs::open_single`
  opens the group, lists its child arrays, captures dims/dtype/shape/attrs,
  and returns a `NodeData`. `_rustytree.open_datatree(path, *, group=None)`
  is now wired end-to-end and returns a Python dict
  `{"path", "attrs", "vars": [{"name", "dims", "dtype", "shape", "attrs"}]}`.
  `file://` URLs accepted; other URL schemes (s3://, gs://, …) raise
  `NotImplementedError` with a clear message — the icechunk + remote
  `object_store` dispatch lands in the next PR.
- 8 new pytest tests against a synthetic 2-array Zarr v3 fixture
  (`tests/test_walk.py`).
- Async runtime + error scaffolding ([#6]): `src/runtime.rs` exposes a
  process-global `tokio` multi-threaded runtime via `OnceLock`;
  `src/error.rs` defines `RustytreeError` (thiserror enum) with a
  `From<RustytreeError> for PyErr` mapping each variant to the right
  Python exception (`OSError`, `KeyError`, `ValueError`, `RuntimeError`).
- Project scaffold ([#1]): maturin-built PyO3 cdylib registered as an
  xarray backend; `xr.open_datatree(engine="rustytree")` resolves to a
  stub raising `NotImplementedError` until the walk is implemented.
- 9-test pytest suite locking the plugin-discovery contract ([#1]).
- `[lints.clippy]` policy: `pedantic = warn`, `perf = deny`,
  `unwrap_used = warn` ([#1]).
- `CHANGELOG.md`, `README.md`, and `docs/` folder ([#4]):
  `architecture.md`, `usage.md`, `contributing.md`, `release-process.md`.
- CI workflow ([#5]): `.github/workflows/ci.yml` runs `cargo fmt`,
  `cargo clippy -D warnings`, `cargo test`, `maturin develop`, and
  `pytest` against Python 3.12 on every push to `main` and every pull
  request.

### Changed

- CI speedup ([#8]): the "install dev extras" step previously ran
  `uv pip install -e ".[dev]"`, which rebuilt the cdylib on top of the
  `maturin develop` step, doubling the per-matrix-cell build time. The
  step now installs the `[dev]` deps directly (`pytest`, `pytest-cov`,
  `zarr>=3.0`, `ruff`) so `maturin develop` is the only build. ~80s
  saved per cell.
- Dependency additions ([#7]): `zarrs = "0.22"` (with `async`),
  `zarrs_storage = "0.4"` (with `async`), `zarrs_object_store = "0.6"`
  (with `fs`), `serde_json = "1"`. icechunk + remote `object_store`
  backends remain deferred to the icechunk-dispatch PR.
- Dependency bump ([#6]): `pyo3` 0.22 → 0.28. Removes the
  `unsafe_op_in_unsafe_fn = "allow"` workaround that PyO3 0.22 needed
  under edition 2024. Adds `tokio` (rt-multi-thread + sync), `futures`,
  and `thiserror` to support the upcoming hierarchy walk.
- Cargo features restructured ([#6]): `extension-module` is now a named
  feature (default-on) instead of being hard-coded into the `pyo3` dep.
  This lets `cargo test` reuse the same wheel-style link configuration
  the runtime uses, so unit tests don't need `LD_LIBRARY_PATH`.
- Toolchain bumped ([#3]): `rust-version` 1.75 → 1.91.1, `edition`
  2021 → 2024 (matches `icechunk`'s MSRV ahead of the next milestone).
- Python floor raised ([#5]): `requires-python` `>=3.10` → `>=3.12` to
  match `zarr>=3.0`'s upstream Python requirement (3.0.x and 3.1.x
  versions are yanked or require Python ≥ 3.11; 3.2.0 requires 3.12).
  Classifiers updated; Python 3.10 / 3.11 entries removed.
- Project license switched from `MIT OR Apache-2.0` to
  `AGPL-3.0-or-later` ([#5]). If you use, modify, or run rustytree —
  including over a network as part of a hosted service — you must make
  the corresponding source code available under the same license to
  anyone interacting with it. See `LICENSE` section 13 for the
  network-use clause. The previous `LICENSE-MIT` and `LICENSE-APACHE`
  files are removed.

[Unreleased]: https://github.com/aladinor/rustytree/compare/...HEAD
[#1]: https://github.com/aladinor/rustytree/pull/1
[#3]: https://github.com/aladinor/rustytree/pull/3
[#4]: https://github.com/aladinor/rustytree/pull/4
[#5]: https://github.com/aladinor/rustytree/pull/5
[#6]: https://github.com/aladinor/rustytree/pull/6
[#7]: https://github.com/aladinor/rustytree/pull/7
[#8]: https://github.com/aladinor/rustytree/pull/8
