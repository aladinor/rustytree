# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Until the first tagged release, every PR appends to `[Unreleased]`. On
release, that section is renamed to `[x.y.z] - YYYY-MM-DD` and a fresh
`[Unreleased]` block is started.

## [Unreleased]

### Added

- Recursive multi-node walk ([#12]): `_rustytree.open_datatree` now
  returns a Python `dict[str, NodeData]` keyed by absolute group path
  rather than a single root `NodeData`. The walk discovers every
  descendant group beneath the requested root with `try_join_all` over
  sibling subtrees (a recursive async function pinned via `Box::pin`),
  then fans out per-group metadata fetches with a second `try_join_all`.
  A `tokio::sync::Semaphore` (default 32 permits, exposed as the
  `max_concurrency=` kwarg) caps in-flight `Group::async_open` calls so
  cold-cache object-store walks don't open more sockets than the client
  can pipeline. Results are pre-order so `xr.DataTree.from_dict`
  consumers see parents before children. Collapses the per-group
  `Repository::open` + per-group PyO3 boundary crossing that previously
  forced callers to loop in Python — one `Repository::open` covers the
  whole walk. Validated against the local KLOT-xradar icechunk repo
  (12 groups): 81.5 ms recursive walk vs 213.3 ms `xr.open_datatree(...,
  engine="zarr")` warm-cache → **2.62×**. Validated against
  `s3://nexrad-arco/KLOT` (107 groups, anon, cold-cache): 1.56 s
  recursive walk vs 50.4 s xarray → **32.4×**. New `multilevel_zarr_store`
  pytest fixture (4 groups, 3 levels) plus 6 new tests in
  `tests/test_walk.py` covering all-groups discovery, per-node
  variables, attrs round-trip, subtree-rooted walks, and
  `max_concurrency=` kwarg plumbing.
- icechunk-on-S3 ([#11]): `_rustytree.open_datatree` now opens icechunk
  repositories that live on S3 (e.g. the public `s3://nexrad-arco/KLOT`).
  A single HEAD on `<prefix>/repo` distinguishes icechunk vs vanilla
  layouts; icechunk paths route through `icechunk::storage::new_s3_storage`
  + `Repository::open` + `readonly_session(BranchTipRef)` →
  `AsyncIcechunkStore`. Same `storage_options` keys as the vanilla path
  (`region`, `endpoint`, `access_key_id`, `secret_access_key`,
  `session_token`, `allow_http`, `skip_signature` / `anon`); icechunk's
  `S3Options` and `S3Credentials` are constructed from them. Network-
  gated pytest smoke `test_open_nexrad_arco_klot_anon_s3` (opt-in via
  `RUSTYTREE_S3_SMOKE=1`). Validated end-to-end against the real
  `s3://nexrad-arco/KLOT` (107 groups; root opens in ~760 ms).
- S3 support for vanilla Zarr v3 ([#10]): `_rustytree.open_datatree`
  accepts `s3://bucket` and `s3://bucket/prefix` URLs, building the store
  via `object_store::aws::AmazonS3Builder` (wrapped in
  `zarrs_object_store::AsyncObjectStore`). New `storage_options=` kwarg
  threads fsspec/xarray-style credentials through (`region`, `endpoint`,
  `access_key_id`, `secret_access_key`, `session_token`, `allow_http`,
  `skip_signature` / `anon`). Unknown keys are rejected so typos surface
  as a clear `ValueError`. New module `src/url.rs` parses the input into
  a `StoreSpec` enum so future schemes (`gs://`, `az://`, `http(s)://`,
  icechunk-on-S3) can be added without touching the dispatch site.
  Cargo: `zarrs_object_store` gains the `aws` feature. 9 new cargo
  unit tests for URL parsing + 4 new pytest tests for the dispatch.
- icechunk dispatch on local filesystem ([#9]): `_rustytree.open_datatree`
  now auto-detects an on-disk icechunk repository (presence of a `repo`
  manifest file + `snapshots/` directory) and routes through
  `icechunk::Repository::open` + `readonly_session(BranchTipRef(branch))`,
  wrapping the resulting session as `zarrs_icechunk::AsyncIcechunkStore`.
  Vanilla Zarr v3 directories continue to go through `zarrs_object_store`.
  New `branch=` kwarg defaults to `"main"`. Same `NodeData` shape comes
  back from either path. New module `src/icechunk_store.rs`. Cargo deps:
  `icechunk = "2"`, `zarrs_icechunk = "0.5"`. 6 new pytest tests in
  `tests/test_icechunk.py` against a local icechunk fixture, plus a
  KTWX-repo smoke test that skips when the repo isn't present locally.
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
[#9]: https://github.com/aladinor/rustytree/pull/9
[#10]: https://github.com/aladinor/rustytree/pull/10
[#11]: https://github.com/aladinor/rustytree/pull/11
[#12]: https://github.com/aladinor/rustytree/pull/12
