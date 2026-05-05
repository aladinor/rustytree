# Architecture

## Why a Rust backend

`xr.open_datatree(engine="zarr")` is sequential at every level that matters
for object-storage latency:

- Per-group decode: `open_groups_as_dict` decodes one group at a time
  (xarray issue #10579, PR #10742).
- Per-coordinate index creation: one round-trip per coord.
- Per-time-variable dtype inference: two extra reads per time variable for
  CF datetime decoding (issue #11303, PR #11304).
- No subtree pruning: `group=` is a literal path, not a glob pattern
  (issue #11196, PR #11302).

xarray's PRs above all fight Python-side: `asyncio.gather` + thread-pool
wrappers + `max_concurrency` semaphores reentering the GIL. `rustytree`
collapses that stack by owning the read path in Rust: one tokio runtime,
one FFI crossing with the GIL released, and `try_join_all` over async store
operations under the hood. The recursive multi-node walk is in place today
(`_rustytree.open_datatree(...)` returns `dict[str, NodeData]` keyed by
absolute group path); lazy chunk reads + `RustyBackendArray` ship in a
follow-up PR.

## Primary target: icechunk-backed Zarr v3

The dominant use case is icechunk-backed Zarr v3. icechunk is **not** a
regular Zarr v3 store — chunks live at content-addressed hashed paths and
metadata lives inline in snapshot files, not at canonical `c/0/0` keys.
Reading icechunk requires icechunk's storage layer.

The official Rust adapter [`zarrs_icechunk`](https://github.com/zarrs/zarrs_icechunk)
exposes `AsyncIcechunkStore::new(session)` returning an
`AsyncReadableStorageTraits` impl. `rustytree` will land on
`icechunk = "2"`, `zarrs = "0.22"`, `zarrs_icechunk = "0.5"` in the next
implementation milestone.

icechunk reframes the bottleneck profile:

- **Snapshot file IS the consolidated metadata.** One GET fetches every
  group's `zarr.json` inline. Earthmover explicitly recommends
  `consolidated=False` against icechunk because the snapshot already plays
  that role.
- The N-groups × RTT problem (#10579) shrinks to N-groups × *parse* —
  parallelizable in Rust without GIL ping-pong.
- Per-coord round-trips and CF datetime decoding round-trips should remain
  real wins (to be confirmed by the eventual benchmark suite).

## Polymorphic engine

`engine="rustytree"` resolves both icechunk and vanilla Zarr v3 stores
polymorphically at the store boundary, the same way `engine="zarr"` does
today (icechunk's Python `IcechunkStore` implements zarr-python's Store
protocol).

| `filename_or_obj` | Rust store impl |
| --- | --- |
| Local path to an icechunk repo (auto-detected) | `Repository::open` → `readonly_session` → `AsyncIcechunkStore` |
| `s3://` URL to an icechunk repo (auto-detected) | `icechunk::storage::new_s3_storage` → `Repository::open` → `AsyncIcechunkStore` |
| Local or `s3://` URL to a plain Zarr v3 store | `zarrs_object_store::AsyncObjectStore` |
| Python `IcechunkStore` / `Session` (later) | unwrap to icechunk `Session` → `AsyncIcechunkStore` |
| `MutableMapping` / fsspec object (later) | fall back to `zarrs_object_store` |

Detection cues:
- **Local**: `<root>/repo` file + `<root>/snapshots/` directory → icechunk.
- **S3**: one HEAD on `<prefix>/repo` → 200 means icechunk, 404 means vanilla Zarr v3.
- **GCS / Azure / HTTP**: same HEAD-probe pattern; lands with the next remote-stores PR.

## Module map (target layout)

Today `src/lib.rs`, `src/runtime.rs`, `src/error.rs`, `src/store.rs`,
`src/icechunk_store.rs`, `src/url.rs`, `src/node.rs`, `src/walk.rs`, and
`python/rustytree/{__init__.py, backend.py}` exist. The remaining modules
in the layout below (`src/glob.rs`, `src/array.rs`,
`python/rustytree/_array.py`, `python/rustytree/_decode.py`) land in
follow-up PRs.

```
rustytree/
├── Cargo.toml                # rust-version = "1.91.1", edition = "2024"
├── pyproject.toml            # maturin build, xarray.backends entry point
├── CHANGELOG.md              # Keep a Changelog
├── README.md
├── docs/                     # this folder
├── src/                      # Rust sources
│   ├── lib.rs                # PyO3 module: registers open_datatree, open_dataset
│   ├── runtime.rs            # Shared tokio multi-thread runtime (OnceLock)
│   ├── store.rs              # filename_or_obj → Arc<dyn AsyncReadableListableStorageTraits>
│   ├── icechunk_store.rs     # icechunk dispatch: unwrap Python Session OR Repository::open
│   ├── walk.rs               # Async hierarchy walk + glob prune
│   ├── node.rs               # NodeData: per-group metadata snapshot
│   ├── array.rs              # ZarrsArrayHandle: lazy chunk reads via zarrs
│   ├── error.rs              # RustytreeError (thiserror) + impl From → PyErr
│   └── glob.rs               # group= glob matching during walk
├── python/rustytree/         # Python package
│   ├── __init__.py
│   ├── backend.py            # RustytreeBackendEntrypoint
│   ├── _array.py             # RustyBackendArray (xarray BackendArray adapter)
│   └── _decode.py            # NodeData → xr.Dataset, CF decode handoff
├── tests/
└── benchmarks/
```

## Concurrency model

- One process-global `tokio::runtime::Runtime` (multi-thread, default workers)
  created in `runtime.rs` via `OnceLock`. Reused for both the discovery walk
  and lazy chunk reads.
- A `tokio::sync::Semaphore` caps in-flight `Group::async_open` /
  `Array::async_open` requests during the walk. Default permits = **32**.
  Exposed as `max_concurrency=` kwarg on `xr.open_datatree`.
- The whole walk is one `runtime.block_on(...)` from the PyO3 boundary;
  the GIL is held only for the final dict marshalling.

## Open path

```text
Python: xr.open_datatree(filename_or_obj, engine="rustytree", ...)
   │
   ▼  (xarray plugin loader → entry point)
RustytreeBackendEntrypoint.open_datatree           [python/rustytree/backend.py]
   │
   ▼  (one PyO3 call, GIL released around block_on)
_rustytree.open_datatree(path, *, group, branch, storage_options, max_concurrency)
   │
   ▼  Rust: runtime.block_on(...)
1. store build:
     • probe HEAD <prefix>/repo  ──┐
     • Repository::open           ──┼─── tokio::join! (PR #13)
     • readonly_session              │
2. walk::walk_recursive(store, root, semaphore)    → Vec<NodeData>
     • discover_paths (recursive Box::pin + try_join_all over siblings)
     • per-group open_single (Group::async_open + per-array fan-out)
3. Build PyDict[path → {path, attrs, vars: [{name, dims, dtype, shape, attrs, handle}]}]
   │
   ▼  Back in Python (RustytreeBackendEntrypoint, PR #15)
4. For each (path, node):
     • Wrap each var's `handle` as RustyBackendArray
     • LazilyIndexedArray(RustyBackendArray(handle))
     • Variable(dims, data, attrs)
     • xr.conventions.decode_cf_variables(...)
     • Split into data_vars / coord_vars
     • xr.Dataset(data_vars, coords, attrs)
5. datatree_from_dict_with_io_cleanup(groups)      → xr.DataTree
```

When the caller passes `group=/foo`, the rust walk surfaces absolute
paths (`/foo`, `/foo/bar`, …) and the entrypoint re-roots them at `/`
to match xarray's subtree contract.

## Performance targets

To be validated by the benchmark suite (`benchmarks/bench_open_datatree.py`,
not yet implemented):

- ≥ 3× cold-cache speedup vs. `engine="zarr"` on a local icechunk DataTree.
- ≥ 5× warm-cache speedup on the same.
- ≥ 5× speedup on a 100-group vanilla Zarr v3 store on remote S3.

### Current measurements

| Target | Engine | Wall time | Speedup |
| --- | --- | --- | --- |
| KLOT-xradar (local icechunk, 12 groups, warm-cache) | `xr.open_datatree(..., engine="zarr", consolidated=False)` | 213.3 ms | — |
| KLOT-xradar (local icechunk, 12 groups, warm-cache) | `_rustytree.open_datatree(...)` (recursive walk) | 81.5 ms | **2.62×** |
| `s3://nexrad-arco/KLOT` (anon S3 icechunk, 107 groups, cold-cache) | `xr.open_datatree(session.store, engine="zarr", consolidated=False)` | 47,608 ms | — |
| `s3://nexrad-arco/KLOT` (anon S3 icechunk, 107 groups, cold-cache, release + parallel probe) | `_rustytree.open_datatree(...)` ([#13]) | 569 ms | 84× |
| `s3://nexrad-arco/KLOT` (anon S3 icechunk, 107 groups, cold-cache, release + Phase 5 entrypoint + eager dim coords + metadata-only datetime) | `xr.open_datatree(URL, engine="rustytree", ...)` ([#15]) | **2,071 ms** | **23×** |

The Phase 5 entrypoint number (2,071 ms) is the headline-API number
through `xr.open_datatree(engine="rustytree", ...)`, including
xarray's CF decoding, default-index construction, and per-node
`Dataset` assembly. Structural + value parity vs `engine="zarr"` is
verified across all 107 nodes (paths, dims, scalar coords, attrs,
`vcp_time` `DatetimeIndex` with `.sel(...)` working).

The S3 win has four stacked drivers:

- **Cross-group parallelism**: xarray opens 107 groups sequentially
  through `IcechunkStore`'s `SyncMixin`, paying ~470 ms per group for
  the metadata round-trip. The recursive walk fans them out through one
  tokio runtime with `try_join_all` and a 32-permit semaphore, holding
  ~32 in-flight metadata GETs at a time.
- **Probe pipelining ([#13])**: the icechunk-vs-vanilla auto-detect
  HEAD on `<prefix>/repo` (~260 ms cold-cache TLS+DNS+TCP) now races
  alongside `Repository::open` instead of running before it.
- **Parallel eager fanout for self-named dim coords ([#15])**:
  xarray's `_maybe_create_default_indexes` post-pass reads every
  self-named 1-D coord on every node to build pandas Index objects.
  Left lazy that's N×serial RTTs through our backend; pre-fetched in
  parallel from Rust (gated on the same 32-permit semaphore) it's
  resident memory by the time xarray touches it.
- **Metadata-only datetime dtype inference ([#15])**: a context-
  manager monkey-patch around `decode_cf_variables` swaps xarray's
  `_decode_cf_datetime_dtype` for a version that synthesises example
  values instead of peeking `arr[0]` and `arr[-1]`. Mirrors xarray
  PR #11304's idea; removable once that lands upstream.

Per-array eager opens (today's `open_single` opens every array to
populate `VarMeta`) are still the ceiling on further wins on the
warm-cache path — the lazy `BackendArray` PR (#14) removes that cost
from the lazy chunk-read side; the icechunk snapshot fast path
(Phase 4 / part 2) would close it on the open side too.

## Lazy chunk reads

Implemented in [#14]. Every var dict produced by `_rustytree.open_datatree(...)`
carries a `handle` field — a `ZarrsArrayHandle` PyO3 class wrapping the
opened `zarrs::Array`. The walk was already paying for the
`Array::async_open` calls (to populate `VarMeta`); the handle just
keeps the resulting `Arc<Array>` alive instead of dropping it.

`ZarrsArrayHandle` exposes:

- `shape`, `dtype` (canonical NumPy string)
- `read_subset(ranges)` where `ranges = list[tuple[int, int]]` (one
  `(start, stop)` per dim, exclusive stop). Runs
  `runtime.block_on(array.async_retrieve_array_subset_elements::<T>(...))`
  with the GIL released via `Python::detach`. Returns a 1-D NumPy
  array of the matching primitive type.

Python-side, `RustyBackendArray(BackendArray)` adapts xarray's indexing
protocol to the handle:

- Advertises `IndexingSupport.BASIC` (slice + scalar-int per axis).
  Fancy indexing flows through xarray's
  `explicit_indexing_adapter` which decomposes it into a sequence of
  basic reads.
- `_raw_indexing_method` translates xarray's tuple-of-(slice|int) into
  `(start, stop)` ranges, calls `handle.read_subset(ranges)`, reshapes,
  and squeezes axes selected by integer indexers.

Phase 5 (BackendEntrypoint wiring) is what makes `xr.open_datatree(URL,
engine="rustytree")` actually return a real `DataTree` — it wraps each
var's handle as a `RustyBackendArray`, then in
`LazilyIndexedArray(...)`, then folds it into a per-group `xr.Dataset`
that runs through `xr.conventions.decode_cf_variables`.
