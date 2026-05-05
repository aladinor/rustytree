# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Until the first tagged release, every PR appends to `[Unreleased]`. On
release, that section is renamed to `[x.y.z] - YYYY-MM-DD` and a fresh
`[Unreleased]` block is started.

## [Unreleased]

### Changed

- icechunk snapshot fast-path for metadata walk ([#19]).
  `walk_recursive` is now a dispatcher: for icechunk inputs it routes
  to a new `walk_icechunk_session_snapshot` that reads
  `Session::list_nodes(...)` (in-memory snapshot tree, no I/O after
  the user-side `Repository::open`) and projects each `NodeSnapshot`
  into `NodeData` / `VarMeta` directly. zarrs's `Array` objects are
  constructed via `Array::new_with_metadata` (synchronous, no I/O)
  using the `user_data` payload from the snapshot, parallelised
  across parent groups via `spawn_blocking + try_join_all` so
  per-array codec/chunk-grid construction overlaps. Vanilla Zarr v3
  inputs continue to use the existing zarrs-based walker.

  Profiled on `s3://nexrad-arco/KLOT` (107 groups + 1300 arrays,
  cold-cache, release):

  ```
              before #19    after #19
  metadata walk:  ~1500 ms     356 ms   (-76%)
  eager_phase:    ~1170 ms    ~1170 ms   (unchanged — separate bottleneck)
  python decode:   455 ms      455 ms
  TOTAL:          ~2100 ms    ~2000 ms
  ```

  The 1.1s metadata-walk speedup is the structural win. The
  eager_phase (243 coord-chunk reads through `AsyncIcechunkStore`)
  remains the dominant cost at ~1.17s; tracking as a follow-up
  (Phase 4/part 3) — bypassing `AsyncIcechunkStore::get` to read
  chunks directly via `Session::get_chunk` would eliminate the
  per-call Store-wrapper rebuild + `RwLock<Session>` contention.

  Refactor surface:
  - `IcechunkBundle { session, store }` replaces the bare
    `AsyncReadableListableStorage` for icechunk inputs — both handles
    needed (session for the fast-path metadata walk, store for lazy
    chunk reads).
  - `WalkSource::{Icechunk, Vanilla}` enum gates the walker dispatch.
  - `bundle_from_session_bytes` replaces the old
    `store_from_session_bytes`; same FFI shape, returns the bundle.
  - All 66 existing tests pass unchanged (parity with the old
    walker is byte-identical from a `NodeData`/`VarMeta` standpoint).

- PR #17 audit-driven cleanup ([#18]): three small follow-ups
  surfaced by retrospectively running the audit chain against the
  merged PR #17 diff. None block correctness; bundled here so the
  full-development-path audit workflow stays honest.
  - Drop redundant `bytes(... .as_bytes())` wrap in
    `_to_rust_source` — `PySession.as_bytes()` already returns
    `bytes`; the outer wrap was dead code.
  - Add typed `RustytreeError::IcechunkSession(SessionError)`
    variant (`#[from]` from `icechunk::session::SessionError`).
    `store_from_session_bytes` now propagates with `?` instead of
    formatting through `RustytreeError::Other(String)`. Maps to
    `ValueError` on the Python side (caller-supplied bytes were
    bad), grouped with `InvalidInput` to keep the existing
    contract.
  - New `tests/test_to_rust_source.py` (7 tests) exercises every
    input shape: `icechunk.Session`, `icechunk.IcechunkStore`, str,
    `pathlib.Path`, an arbitrary object that fall-back-coerces via
    `str(...)`, and the path where `icechunk` isn't importable. Plus
    a Rust unit test (`display_icechunk_session_wraps_underlying`)
    confirming the typed error variant carries the underlying
    `SessionError` and renders with the `"icechunk session: "`
    prefix.

- **Breaking**: drop the `s3://` URL icechunk dispatch ([#16]).
  `xr.open_datatree("s3://bucket/prefix", engine="rustytree")` no
  longer auto-detects icechunk repos — users must construct the
  icechunk `Session` themselves and pass `session.store`. This
  matches xarray's stock `engine="zarr"` interface exactly and gives
  users full control over branch / credentials / cache config (which
  the URL dispatch hard-coded). Migration:
  ```python
  # before
  dt = xr.open_datatree(
      "s3://nexrad-arco/KLOT", engine="rustytree",
      storage_options={"region": "us-east-1", "anon": True},
  )
  # after
  import icechunk
  storage = icechunk.s3_storage(
      bucket="nexrad-arco", prefix="KLOT",
      region="us-east-1", anonymous=True,
  )
  repo = icechunk.Repository.open(storage)
  session = repo.readonly_session("main")
  dt = xr.open_datatree(session.store, engine="rustytree")
  ```
  Local-FS path strings (`/path/to/repo`) still work for both
  vanilla Zarr v3 and icechunk (auto-detected via the
  `<root>/repo` + `<root>/snapshots/` heuristic). `s3://` URLs
  still work for **vanilla Zarr v3** stores. Cross-extension
  Session unwrap goes via `PySession.as_bytes()` →
  `icechunk::session::Session::from_bytes` (msgpack); both crates
  link the same `icechunk = "2"` so the format matches. Wall time on
  `s3://nexrad-arco/KLOT` (107 groups, anonymous, cold-cache,
  release build): **2.6 s** via the new path
  (`Repository.open` ~300 ms user-side + 2.3 s rustytree). The
  `tokio::join!` parallel-probe optimization ([#13]) is removed
  along with the URL dispatch — pinpointed dead code now that
  `s3_is_icechunk` no longer runs.

### Added

- `Variable.encoding["chunks"]` and `["preferred_chunks"]` ([#16]).
  Every `Variable` produced by the entrypoint now carries the on-disk
  chunk shape on `encoding`, so `xr.open_datatree(..., chunks={})`
  produces dask arrays with the correct chunk grid (multi-chunk along
  chunked dims) rather than a single big chunk per array. New
  `ZarrsArrayHandle.chunks` getter (Rust) exposes the chunk shape via
  `array.chunk_shape(&[0; N])`. New `tests/test_chunks.py` (4 tests)
  verifies handle-side chunks, encoding round-trip, end-to-end
  `chunks={}` correctness on a fixture with non-trivial chunking, and
  values round-trip via dask `.compute()`.

### Added

- `RustytreeBackendEntrypoint.open_datatree` + `open_dataset` end-to-end
  ([#15]). The headline `xr.open_datatree(URL, engine="rustytree", ...)`
  API now returns a real `xr.DataTree` instead of raising
  `NotImplementedError`. Validated against `s3://nexrad-arco/KLOT`
  (107 groups, anonymous icechunk, cold-cache release build):
  **2,071 ms** rustytree vs 47,608 ms `engine="zarr"` → **23×**
  end-to-end speedup, with full structural + value parity checked
  across all 107 nodes. CF decoding (`decode_times`, `mask_and_scale`,
  `decode_coords`, etc.), pandas indexes (`.sel(vcp_time="...")`
  works), and scalar/1-D coord values all match `engine="zarr"`.
  Three structural pieces close the gap:
  1. **Parallel eager fanout for self-named dim coords** (`src/walk.rs`
     Phase C). After the existing discover + open phases, the walk
     identifies `var.dims == [var.name]` 1-D coords across every node,
     fan-outs `array.async_retrieve_array_subset_elements::<T>` via
     `try_join_all` against the same tokio runtime + 32-permit
     semaphore, and stuffs the results into a new
     `EagerElements` enum on `VarMeta`. xarray's
     `_maybe_create_default_indexes` post-pass then finds resident
     numpy data instead of triggering N×serial RTTs through our lazy
     backend. Skips coords > 1 M elements (size cap). New
     `src/dtype_dispatch.rs` module exports a
     `for_each_supported_dtype!` macro shared between `read_subset`
     and the eager fanout to avoid 11-arm drift.
  2. **Metadata-only datetime dtype inference** (Python monkey-patch
     in `_metadata_only_datetime_dtype`). xarray's
     `_decode_cf_datetime_dtype` peeks `arr[0]` and `arr[-1]` per CF
     time variable to call `decode_cf_datetime(example, units, ...)`
     and return `result.dtype`. We swap the function for the duration
     of `decode_cf_variables`: synthesise `np.array([0, 0])` with the
     variable's int dtype, run the same `decode_cf_datetime` call, and
     return the resulting dtype — same answer, no chunk read. Falls
     back to the original peeking implementation on exception so
     malformed-units error reporting stays accurate. Defensively
     guarded — if xarray ever moves/renames the function (e.g. when
     PR #11304 lands upstream), the patch becomes a no-op rather than
     raising. To be removed once rustytree's xarray floor moves past
     PR #11304.
  3. **Sparse-chunk handling in `read_subset`** (`src/array.rs`).
     zarrs's `async_retrieve_chunk_subset_opt` slow-path bypasses the
     `fill_value` fallback when a chunk doesn't exist in storage —
     critical for icechunk arrays where chunks are sparse. Workaround:
     align every read request to chunk-grid boundaries, which forces
     zarrs's per-chunk fast path (`async_retrieve_chunk_opt`) that
     does handle missing chunks. Slice the chunk-aligned result down
     to the requested range via a new `slice_nd` helper.

  Other entrypoint plumbing: re-roots the returned `DataTree` at the
  `group=` argument to match xarray's subtree contract; `open_dataset`
  delegates to `open_datatree` and pulls `tree.dataset`. Six new
  `tests/test_eager_fetch.py` tests cover the predicate, data
  round-trip, oversized fallback, and end-to-end "no extra reads via
  the lazy backend" assertion. Nine new `tests/test_backend_entrypoint.py`
  tests verify parity vs `engine="zarr"` on the tiny + multilevel +
  icechunk fixtures using `xr.testing.assert_identical`. Stale
  `NotImplementedError("Phase 4")` stub removed; Phase 1 scaffold
  tests trimmed.
  The entrypoint calls `_rustytree.open_datatree(...)` for the
  metadata, then for each group: wraps every var's `ZarrsArrayHandle`
  as `LazilyIndexedArray(RustyBackendArray(handle))`, builds raw
  `xr.Variable`s, runs `xr.conventions.decode_cf_variables(...)` over
  them (so `mask_and_scale` / `decode_times` / `decode_coords` /
  `decode_timedelta` / `concat_characters` / `use_cftime` /
  `drop_variables` behave the same as `engine="zarr"`), splits decoded
- Lazy `ZarrsArrayHandle` + `RustyBackendArray` for chunk reads ([#14]):
  every var dict produced by `_rustytree.open_datatree(...)` now carries
  a `"handle"` key holding a `ZarrsArrayHandle` PyO3 class. The handle
  wraps the already-opened `zarrs::Array` (no extra opens — the walk
  was already opening every array to populate `VarMeta`) and exposes
  `shape`, `dtype` (NumPy-canonical string), and
  `read_subset(ranges) -> numpy.ndarray` that runs
  `runtime.block_on(array.async_retrieve_array_subset_elements::<T>(...))`
  with the GIL released via `Python::detach`. Dispatch covers `bool`,
  `int{8,16,32,64}`, `uint{8,16,32,64}`, and `float{32,64}`; less
  common dtypes raise `NotImplementedError` naming what's missing.
  New `python/rustytree/_array.py` defines `RustyBackendArray(BackendArray)`
  using `xarray.core.indexing.explicit_indexing_adapter` at
  `IndexingSupport.BASIC`, translating xarray-shaped indexers into
  `(start, stop)` ranges per axis (with axis-squeeze for integer
  indexers, including negative ones). Cargo: `numpy = "0.28"` to back
  the `PyArray1::from_vec` zero-copy handoff. New `tests/test_lazy.py`
  (11 tests) covers handle presence, shape/dtype round-trip, full-array
  reads matching `xr.open_zarr`, basic slicing, integer indexing,
  negative indexing, and the unsupported-dtype error path. This PR
  unblocks Phase 5 — `xr.open_datatree(URL, engine="rustytree")` can
  now be wired by wrapping each var's handle as a `RustyBackendArray`
  and handing the dict to `datatree_from_dict_with_io_cleanup`.

### Changed

- Pipeline the S3 icechunk-vs-vanilla probe with `Repository::open`
  ([#13]): the auto-detect HEAD on `<prefix>/repo` previously ran
  sequentially before the icechunk open, paying a full TLS + DNS + TCP
  handshake (~260 ms cold-cache against AWS) on top of the actual repo
  open (~302 ms). They now run concurrently via `tokio::join!`; if the
  probe rules out icechunk we drop the in-flight icechunk open and fall
  back to vanilla. Wall = max(probe, icechunk_open) instead of probe +
  icechunk_open. Cold-cache profile against `s3://nexrad-arco/KLOT`
  (107 groups, release build): **843 ms → 563 ms (33% faster)**.
  Vanilla S3 stores pay one wasted icechunk open whose error is
  discarded — acceptable cost for preserving the auto-detect contract.

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
[#13]: https://github.com/aladinor/rustytree/pull/13
[#14]: https://github.com/aladinor/rustytree/pull/14
[#15]: https://github.com/aladinor/rustytree/pull/15
[#16]: https://github.com/aladinor/rustytree/pull/16
[#17]: https://github.com/aladinor/rustytree/pull/17
[#18]: https://github.com/aladinor/rustytree/pull/18
[#19]: https://github.com/aladinor/rustytree/pull/19
