# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Until the first tagged release, every PR appends to `[Unreleased]`. On
release, that section is renamed to `[x.y.z] - YYYY-MM-DD` and a fresh
`[Unreleased]` block is started.

## [Unreleased]

### Added

- `numcodecs.zlib` codec support ([#41]). Enables the `zlib` feature on
  the `zarrs` dependency so rustytree can decode arrays whose codec
  pipeline uses the non-standard `numcodecs.`-namespace codecs
  zarr-python writes — e.g. the public `earthmover-public/goes-16`
  arraylake dataset, which chains `bytes → numcodecs.shuffle →
  numcodecs.zlib`. zarrs's `zlib` codec is documented byte-compatible
  with zarr-python's; `numcodecs.shuffle` is always compiled, and
  `gzip` / `blosc` / `zstd` / `crc32c` already come from zarrs's
  default features. Verified bit-for-bit against zarr-python on a
  GOES-16 `CMI_C01` slice. New non-gated `tests/test_codecs.py` writes
  the shuffle+zlib pipeline to a local store and checks the decode
  round-trips.

### Fixed

- Open arraylake / Earthmover icechunk sessions ([#41], fixes #40).
  `xr.open_datatree(session.store, engine="rustytree")` raised
  `ValueError: icechunk session: unknown error: unknown variant
  PythonCredentialsFetcher, there are no variants` for sessions
  created via arraylake. Such sessions store S3 credentials as
  `Refreshable(Arc<dyn S3CredentialsFetcher>)` whose concrete fetcher
  is `#[typetag::serde]`-registered only inside icechunk-python's
  cdylib; rustytree links the vanilla `icechunk` crate, whose fetcher
  registry is empty, so `Session::from_bytes` couldn't resolve the tag.
  `xr.open_zarr` is unaffected because it uses the live in-process
  store and never round-trips through bytes. New `src/py_credentials.rs`
  re-registers `PythonCredentialsFetcher` (S3 / GCS / Azure) inside
  rustytree's cdylib, mirroring icechunk-python's `get()`: serve the
  scattered `initial` static credentials while fresh, else acquire the
  GIL and re-run the embedded pickled credential callable to refresh —
  so the common case and `scatter_initial_credentials=False` /
  mid-walk credential expiry all keep working. `typetag` is pinned
  `=0.2.21` to share icechunk's `inventory` registry. A Python-side
  friendly error (`_rust_open_or_explain` in `backend.py`) is a safety
  net for a future icechunk-python that renames the fetcher. CI now
  builds tests with `--no-default-features` so PyO3 links libpython:
  the inventory-retained fetcher impls reference `Python::attach`, so
  their libpython symbols can no longer be dead-code-eliminated from
  the test binary.

## [0.2.1] - 2026-05-23

### Fixed

- Compatibility with icechunk 2.0.5: pin Rust-side `icechunk` crate to
  `=2.0.5` and bump Python-side floor to `>=2.0.5`. Fixes #37.
  The `Session::from_bytes` msgpack format changed between 2.0.4 and
  2.0.5; the 0.2.0 wheel (built against 2.0.4) could not deserialise
  session bytes produced by icechunk-python 2.0.5.

## [0.2.0] - 2026-05-08

### Changed

- `include_ancestor_coords=True` against an icechunk Session/Store now
  serialises the session snapshot exactly once, instead of once per
  ancestor. Fixes #34 — addresses the audit finding from #33 that
  each `self.open_dataset(...)` call in the ancestor-merge loop was
  re-running `session.as_bytes()`. `_to_rust_source` short-circuits
  on `bytes` input, and `open_datatree` threads its already-serialised
  `source` through the loop. Behaviour-only refactor; measured saving
  is small (~8 µs per depth-2 call against local KLOT, snapshot
  size 600 B), so this is mostly a code-hygiene fix.

### Added

- `include_ancestor_coords=True` (default) on
  `xr.open_datatree(..., engine="rustytree", group=...)` for literal
  group paths. Promotes ancestor group datasets into the new root so
  `latitude`/`longitude`/`altitude` (and any other ancestor-level
  coords/vars) are present after a subtree open — matches what users
  get from a full-tree open + slice with
  `inherit="all_coords"`. Glob mode (`group="*/sweep_0"`) is unchanged
  (already correct via `_filter_by_glob`). Set to `False` to keep the
  pre-flag orphaned-subtree behavior.

- Logo + README header. New `assets/logo.png` (transparent icon) and a
  light/dark banner pair (`assets/logo-banner-{light,dark}.png` plus
  SVG sources). README replaces the `# rustytree` heading with a
  `<picture>` tag that auto-switches between light and dark variants
  via `prefers-color-scheme`. Palette aligns with the AtmoScale brand
  (cyan `#00D4FF`, deep navy `#0F1724`) with rust-orange tip accents
  on the chunked-cube tree icon. All text passes WCAG AAA on its
  intended background.

- Status badges in the README: CI status, PyPI version, supported
  Python versions, and license — all clickable to the relevant page.

### Fixed

- Accept `zarr_format` and `consolidated` kwargs in
  `xr.open_datatree(..., engine="rustytree")` and
  `xr.open_dataset(..., engine="rustytree")`. Previously the entrypoint
  raised `TypeError: unexpected keyword argument` because the kwargs
  weren't declared. They're now declared on both methods and validated:
  v3-implying values (`zarr_format=3` / `None`, `consolidated=False` /
  `None`) pass through silently; v2-implying values (`zarr_format=2`
  or any non-3 int, `consolidated=True`) raise
  `NotImplementedError` pointing the user at `engine="zarr"`. rustytree
  currently supports Zarr v3 only; the icechunk snapshot plays the
  consolidated-metadata role for icechunk repos.

## [0.1.0] - 2026-05-05

First tagged release. The engine is end-to-end usable against
icechunk + vanilla Zarr v3 stores via
`xr.open_datatree(engine="rustytree")` and
`xr.open_dataset(engine="rustytree")`, with the per-PR breakdown
below.

### Added

- Tag-driven release workflow ([#27]).
  `.github/workflows/release.yml` builds wheels (manylinux x86_64 +
  manylinux aarch64 + macOS arm64, CPython 3.12 + 3.13) plus an
  sdist via `PyO3/maturin-action`, then attaches everything to a
  GitHub Release on `vX.Y.Z` tag push. `workflow_dispatch` runs
  the same matrix without cutting a release. PyPI publishing is
  gated behind a documented flip-on (trusted publishing).
  `docs/release-process.md` rewritten to describe the workflow,
  cut-a-release flow, hotfix branching, and yank/release-edit
  guidance.

- Glob `group=` filter for `open_datatree` ([#23], [#25]).
  `xr.open_datatree(session.store, engine="rustytree",
  group="/*/sweep_0")` returns a DataTree filtered to matches
  plus their ancestors, mirroring xarray PR #11302's
  `PurePosixPath.match` semantics. [#23] implemented post-walk
  Python filtering; [#25] pushed a conservative prefix predicate
  (`src/glob.rs`) into the Rust walk to prune subtrees that
  can't match — `~2.4 s → ~1.4 s (~1.8×)` on KLOT. The
  predicate handles `*` patterns natively and falls back to
  "no prune" for `?` / `[` (Python post-filter still applies
  them correctly). Globs in `open_dataset` raise
  `NotImplementedError` (single-Dataset return shape can't
  accommodate multi-match).

- Non-recursive walk for `open_dataset(group=literal)` ([#22]).
  Single-Dataset opens skip walking siblings/descendants:
  `xr.open_dataset(session.store, engine="rustytree",
  group="/VCP-12/sweep_0")` lands at ~1.3 s vs ~2.0 s full-tree
  (~1.7× on KLOT). `walk_recursive` and both walkers gain a
  `recursive: bool` parameter; vanilla skips `discover_paths`,
  icechunk filters the snapshot to root + immediate-array
  children. Glob patterns still walk recursively (the glob
  pre-discovery prune in [#25] covers those).

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
  1. **Parallel eager fanout for self-named dim coords** (eager-
     fetch step in `src/walk.rs`). After the existing discover +
     open phases, the walk
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
  icechunk fixtures using `xr.testing.assert_identical`.

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
  negative indexing, and the unsupported-dtype error path.

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

- icechunk-on-S3 ([#11]): `_rustytree.open_datatree` opens icechunk
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
  `s3://nexrad-arco/KLOT` (107 groups; root opens in ~760 ms). _The
  `s3://` URL icechunk dispatch was subsequently removed in [#16] —
  see Changed below._

- S3 support for vanilla Zarr v3 ([#10]): `_rustytree.open_datatree`
  accepts `s3://bucket` and `s3://bucket/prefix` URLs, building the store
  via `object_store::aws::AmazonS3Builder` (wrapped in
  `zarrs_object_store::AsyncObjectStore`). New `storage_options=` kwarg
  threads fsspec/xarray-style credentials through (`region`, `endpoint`,
  `access_key_id`, `secret_access_key`, `session_token`, `allow_http`,
  `skip_signature` / `anon`). Unknown keys are rejected so typos surface
  as a clear `ValueError`. New module `src/url.rs` parses the input into
  a `StoreSpec` enum so future schemes (`gs://`, `az://`, `http(s)://`)
  can be added without touching the dispatch site. Cargo:
  `zarrs_object_store` gains the `aws` feature. 9 new cargo unit tests
  for URL parsing + 4 new pytest tests for the dispatch.

- icechunk dispatch on local filesystem ([#9]): `_rustytree.open_datatree`
  auto-detects an on-disk icechunk repository (presence of a `repo`
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
  is wired end-to-end and returns a Python dict
  `{"path", "attrs", "vars": [{"name", "dims", "dtype", "shape", "attrs"}]}`.
  `file://` URLs accepted; other URL schemes (s3://, gs://, …) raise
  `NotImplementedError` with a clear message. 8 new pytest tests against
  a synthetic 2-array Zarr v3 fixture (`tests/test_walk.py`).

- Async runtime + error scaffolding ([#6]): `src/runtime.rs` exposes a
  process-global `tokio` multi-threaded runtime via `OnceLock`;
  `src/error.rs` defines `RustytreeError` (thiserror enum) with a
  `From<RustytreeError> for PyErr` mapping each variant to the right
  Python exception (`OSError`, `KeyError`, `ValueError`, `RuntimeError`).

- CI workflow ([#5]): `.github/workflows/ci.yml` runs `cargo fmt`,
  `cargo clippy -D warnings`, `cargo test`, `maturin develop`, and
  `pytest` against Python 3.12 on every push to `main` and every pull
  request.

- `CHANGELOG.md`, `README.md`, and `docs/` folder ([#4]):
  `architecture.md`, `usage.md`, `contributing.md`, `release-process.md`.

- Project scaffold ([#1]): maturin-built PyO3 cdylib registered as an
  xarray backend; `xr.open_datatree(engine="rustytree")` resolves to a
  stub raising `NotImplementedError` until the walk is implemented.
  9-test pytest suite locking the plugin-discovery contract. Cargo
  `[lints.clippy]` policy: `pedantic = warn`, `perf = deny`,
  `unwrap_used = warn`.

### Changed

- Distribution name on PyPI is `rustytree-xarray` (atmoscale org).
  The canonical `rustytree` is blocked by an unrelated dormant
  package (`rusty-tree`, last released 2022); PyPI's name
  normalisation collapses `-`/`_`/case so the bare name isn't
  available without a PEP 541 reclaim. The import name stays
  `rustytree` everywhere — `import rustytree`,
  `engine="rustytree"`, the Python package directory, the Rust
  crate. Only `pip install` changes.

- Switch the manylinux aarch64 wheel cell to `ubuntu-24.04-arm`
  (native AWS Graviton runner, free for public repos) instead of
  cross-compiling via QEMU on `ubuntu-latest`. The QEMU path failed
  in `ring`'s build script (its pre-generated ARM assembly bails
  if the C preprocessor doesn't define `__ARM_ARCH`, which the
  manylinux cross-gcc doesn't). Native build sidesteps the issue
  and is ~3× faster wall-time.

- Enable PyPI trusted publishing in the release workflow. The
  `pypi-publish` job runs on every `vX.Y.Z` tag push after the
  wheel + sdist matrix completes, using GitHub OIDC against the
  `atmoscale/rustytree-xarray` trusted publisher and the `pypi`
  environment (deployment-tag rule `v*.*.*`). No long-lived API
  tokens.

- Project license switched from `AGPL-3.0-or-later` to `Apache-2.0`
  for the v0.1.0 release. The previous AGPL choice (set in [#5])
  is replaced wholesale; the `LICENSE` file is the standard Apache
  License 2.0 text, and `Cargo.toml` / `pyproject.toml` license
  fields and classifiers are updated to match. Apache-2.0 is the
  conventional license in the scientific-Python and Rust
  communities (xarray, zarr, numpy, polars, tokio, hyper) and
  removes the network-use compliance burden the AGPL imposed for
  hosted-service users. README License section trimmed to a
  one-line reference.

- Documentation refresh for release prep ([#26], [#20]). README
  adds a concrete "Example" section with the full icechunk + KLOT
  walkthrough, plus examples for non-recursive single-Dataset
  open and the glob radar workflow. Quick start collapses to
  `uv sync --extra dev` (one command — replaces the previous
  `uv venv` + `uv pip install` + `maturin develop` flow).
  `docs/usage.md` full rewrite — drops the stale
  `NotImplementedError` claim (true at PR #11; not since PR #15)
  and documents the actual surface: input shapes, glob vs literal
  `group=`, lazy reads, typed error table, performance digest.
  `docs/architecture.md` staleness pass + new "Glob filtering"
  section. `notebooks/README.md` removed (consolidated into the
  main README).

- Refactor per-node Dataset construction via `_RustyDataStore`
  shim ([#21]). Inline `decode_cf_variables` + manual data/coord
  split replaced by a thin `AbstractDataStore` adapter delegated
  to xarray's `StoreBackendEntrypoint.open_dataset`. Net -23
  lines in `backend.py`; inherits `set_close` and group-level
  `encoding` plumbing for free, future-proof against new CF
  decode kwargs xarray adds.

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
  _Subsequently removed in [#16] along with the `s3://` URL icechunk
  dispatch._

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

- Dependency bump + Cargo features restructured ([#6]): `pyo3` 0.22 →
  0.28 — removes the `unsafe_op_in_unsafe_fn = "allow"` workaround that
  PyO3 0.22 needed under edition 2024. Adds `tokio` (rt-multi-thread +
  sync), `futures`, and `thiserror` to support the upcoming hierarchy
  walk. `extension-module` is now a named feature (default-on) instead
  of being hard-coded into the `pyo3` dep — lets `cargo test` reuse
  the same wheel-style link configuration the runtime uses, so unit
  tests don't need `LD_LIBRARY_PATH`.

- Python floor raised + initial license set ([#5]):
  `requires-python` `>=3.10` → `>=3.12` to match `zarr>=3.0`'s
  upstream Python requirement (3.0.x and 3.1.x versions are yanked or
  require Python ≥ 3.11; 3.2.0 requires 3.12). Classifiers updated;
  Python 3.10 / 3.11 entries removed. Initial project license set to
  `AGPL-3.0-or-later` (subsequently swapped to `Apache-2.0` for the
  v0.1.0 release — see License entry above).

- Toolchain bumped ([#3]): `rust-version` 1.75 → 1.91.1, `edition`
  2021 → 2024 (matches `icechunk`'s MSRV ahead of the next milestone).

### Fixed

- Missing literal `group=` raises rather than returning empty
  ([#25]). `xr.open_datatree(session.store, group="VCP/sweep_0")`
  (typo'd path on icechunk) silently returned an empty tree —
  `Session::list_nodes(missing)` returns an empty iterator (vanilla
  raises via `Group::async_open`). Now raises `KeyError` for
  literal paths; globs remain exempt (empty match is valid for
  them per PR #11302).

- Glob pattern `parse` collapses `//` runs ([#25]). A pattern
  with consecutive slashes (`/foo//bar`) previously created an
  empty literal segment that nothing could match → false
  negative on the Rust prune (silent data-loss).
  `PurePosixPath` coalesces `//`; the parse now does too.

- Normalize literal `group=` paths to absolute ([#24]).
  `xr.open_dataset(URL, engine="rustytree",
  group="VCP-12/sweep_0")` was raising `ValueError: invalid
  input: icechunk: invalid root path: path must start with a /
  character`. Pre-existing bug — icechunk's path validator
  strictly requires a leading slash; xarray's convention is
  absolute paths but users commonly drop the slash. New
  `_normalize_literal_group` helper called at both entrypoints,
  routing literal paths through `pathlib.PurePosixPath` so it
  also collapses `//` runs and strips trailing `/`. Globs are
  intentionally not normalized — relative-vs-absolute glob
  semantics differ in `PurePosixPath.match`.

[Unreleased]: https://github.com/aladinor/rustytree/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/aladinor/rustytree/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/aladinor/rustytree/releases/tag/v0.1.0
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
[#18]: https://github.com/aladinor/rustytree/pull/18
[#19]: https://github.com/aladinor/rustytree/pull/19
[#20]: https://github.com/aladinor/rustytree/pull/20
[#21]: https://github.com/aladinor/rustytree/pull/21
[#22]: https://github.com/aladinor/rustytree/pull/22
[#23]: https://github.com/aladinor/rustytree/pull/23
[#24]: https://github.com/aladinor/rustytree/pull/24
[#25]: https://github.com/aladinor/rustytree/pull/25
[#26]: https://github.com/aladinor/rustytree/pull/26
[#27]: https://github.com/aladinor/rustytree/pull/27
[#41]: https://github.com/aladinor/rustytree/pull/41
