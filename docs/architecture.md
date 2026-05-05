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
one FFI crossing with the GIL released, and `try_join_all` over async
store operations under the hood. The recursive multi-node walk, lazy
chunk reads via `RustyBackendArray`, the icechunk snapshot fast-path,
glob `group=` filtering with pre-discovery prune, and non-recursive
single-Dataset opens are all shipped — see [`CHANGELOG.md`](../CHANGELOG.md)
for the per-PR breakdown.

## Primary target: icechunk-backed Zarr v3

The dominant use case is icechunk-backed Zarr v3. icechunk is **not** a
regular Zarr v3 store — chunks live at content-addressed hashed paths and
metadata lives inline in snapshot files, not at canonical `c/0/0` keys.
Reading icechunk requires icechunk's storage layer.

The official Rust adapter [`zarrs_icechunk`](https://github.com/zarrs/zarrs_icechunk)
exposes `AsyncIcechunkStore::new(session)` returning an
`AsyncReadableStorageTraits` impl. `rustytree` pins `icechunk = "2"`,
`zarrs = "0.22"`, `zarrs_icechunk = "0.5"`.

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
today. The input contract is **symmetric with
xarray's stock `engine="zarr"`**: users construct an icechunk
`Session` themselves and pass `session.store`. URL dispatch for
*remote* icechunk has been removed — users have full control over
branch / credentials / cache via the icechunk Python API.

| `filename_or_obj` | Rust store impl |
| --- | --- |
| `icechunk.IcechunkStore` (i.e. `session.store`) | `PySession.as_bytes()` → `Session::from_bytes` → `AsyncIcechunkStore` |
| `icechunk.Session` directly | same path; `session._session.as_bytes()` |
| Local path to an icechunk repo (auto-detected) | `Repository::open` → `readonly_session` → `AsyncIcechunkStore` |
| Local path / `s3://` URL to a plain Zarr v3 store | `zarrs_object_store::AsyncObjectStore` |

Detection cues:
- **Python object**: `isinstance(obj, icechunk.{Session,IcechunkStore})` → bytes-roundtrip path.
- **Local string**: `<root>/repo` file + `<root>/snapshots/` directory → icechunk; otherwise vanilla Zarr v3.
- **`s3://` string**: vanilla Zarr v3 only. Remote icechunk requires an explicit `Session`.
- **GCS / Azure / HTTP**: not yet supported.

The cross-extension Session unwrap goes through `PySession.as_bytes()`
(msgpack-serde) and `icechunk::session::Session::from_bytes()`. Both
crates link the same `icechunk = "2"` so the format matches. PyO3 type
extraction can't reach across cdylibs, so the bytes path is the only
way to hand a live Session from icechunk-python to rustytree without
re-opening the repo.

## Module map (target layout)

```
rustytree/
├── Cargo.toml                # rust-version = "1.91.1", edition = "2024"
├── pyproject.toml            # maturin build, xarray.backends entry point
├── CHANGELOG.md              # Keep a Changelog
├── README.md
├── docs/                     # this folder
├── src/                      # Rust sources
│   ├── lib.rs                # PyO3 module: registers open_datatree
│   ├── runtime.rs            # Shared tokio multi-thread runtime (OnceLock)
│   ├── store.rs              # filename_or_obj → AsyncReadableListableStorage
│   ├── icechunk_store.rs     # icechunk dispatch: unwrap Python Session, Repository::open
│   ├── url.rs                # Parse `s3://` / path inputs
│   ├── walk.rs               # Async hierarchy walk + glob prune
│   ├── node.rs               # NodeData / VarMeta: per-group metadata snapshot
│   ├── array.rs              # ZarrsArrayHandle: lazy chunk reads via zarrs
│   ├── glob.rs               # group= glob predicate (pre-discovery prune)
│   ├── dtype_dispatch.rs     # for_each_supported_dtype! macro
│   └── error.rs              # RustytreeError (thiserror) + impl From → PyErr
├── python/rustytree/         # Python package
│   ├── __init__.py
│   ├── backend.py            # RustytreeBackendEntrypoint + _RustyDataStore shim
│   └── _array.py             # RustyBackendArray (xarray BackendArray adapter)
├── tests/
└── notebooks/                # KLOT demo + future tutorials
```

The Python decode path is inline in `backend.py` via the
`_RustyDataStore(AbstractDataStore)` shim — `StoreBackendEntrypoint`
handles the CF decode + data/coord split + `set_close` / `encoding`
plumbing, so there's no separate `_decode.py`.

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
   │  • Detect glob in `group=` (PurePosixPath chars `* ? [`)
   │  • Normalise literal paths via PurePosixPath (collapse `//`,
   │    strip trailing `/`, prepend leading `/` if missing)
   │  • Translate Session/IcechunkStore inputs to bytes via
   │    PySession.as_bytes() — cross-extension handoff
   │
   ▼  (one PyO3 call, GIL released around `Python::detach` + block_on)
_rustytree.open_datatree(source, *, group, branch, storage_options,
                         max_concurrency, recursive, glob)
   │
   ▼  Rust: runtime.block_on(...)
1. Build WalkSource:
     • bytes      → bundle_from_session_bytes (Session::from_bytes)
     • local path → auto-detect icechunk vs vanilla
     • s3:// URL  → vanilla S3 (icechunk-on-S3 requires a Session)
2. walk::walk_recursive(source, root, max_concurrency, recursive, glob)
     • Icechunk: in-memory snapshot walk via Session::list_nodes
       (one parse per group from `user_data`)
     • Vanilla: discover_paths (recursive Box::pin + try_join_all),
       gated on a 32-permit semaphore
     • Glob predicate prunes subtrees that can't match (vanilla:
       skips Group::async_open; icechunk: skips
       Array::new_with_metadata)
     • recursive=False short-circuits to a single NodeData
3. eager_phase: parallel try_join_all over 1-D self-named coords
     (gated on the same semaphore)
4. Build PyDict[path → {path, attrs, vars: [{name, dims, dtype,
   shape, attrs, handle, eager_data?}]}]
   │
   ▼  Back in Python (RustytreeBackendEntrypoint)
5. If glob → _filter_by_glob(tree, pattern) — PurePosixPath.match
   plus auto-include ancestors
6. If literal `group=` not in tree → KeyError
7. For each (path, node):
     • _RustyDataStore(node) — AbstractDataStore shim
     • StoreBackendEntrypoint.open_dataset(store, ...)
        → decode_cf_variables, data/coord split, set_close, encoding
8. datatree_from_dict_with_io_cleanup(groups)        → xr.DataTree
```

When the caller passes a literal `group=/foo`, the rust walk
surfaces absolute paths (`/foo`, `/foo/bar`, …) and the entrypoint
re-roots them at `/` to match xarray's subtree contract. Glob
results stay rooted at `/` — matched paths can span siblings without
a common non-root prefix.

## Glob `group=` filtering

Glob detection mirrors xarray PR #11302's `_is_glob_pattern`: any of
`*`, `?`, `[` in `group=` triggers glob handling. Matching uses
`pathlib.PurePosixPath.match` as the source of truth, with every
ancestor of a matched path auto-included so `DataTree.from_dict` sees
a well-formed hierarchy.

The Rust `GlobPredicate` (`src/glob.rs`) is a *conservative* prefix
predicate: it parses absolute `*`-only patterns into per-segment
matchers and prunes subtrees that can't match before the walk
descends into them. Patterns containing `?` or `[` (which our hand-
rolled per-segment matcher doesn't fully implement) fall back to
"no prune" — the Python post-filter still applies them correctly,
just without the structural speedup.

The `//`-collapse rule applies on both sides: `parse` filters empty
pattern segments to mirror `PurePosixPath`'s coalescing of `//`
runs, and `_normalize_literal_group` routes literal paths through
`PurePosixPath` so canonical-form mismatches can't produce a missing-
key lookup.

## Performance

Headline measurements against `s3://nexrad-arco/KLOT` (anonymous,
107-node radar DataTree) via the public-API entrypoint
`xr.open_datatree(session.store, engine="rustytree", ...)`. Cold-
cache from a US-East home connection unless noted. The benchmark
suite (`benchmarks/`) is a planned follow-up.

| Workload | `engine="rustytree"` | `engine="zarr"` | Speedup |
| --- | --- | --- | --- |
| Full tree (107 nodes) | ~2.0 s | ~50 s | **~25×** |
| Glob `/*/sweep_0` (15 matched) | ~1.4 s | n/a* | — |
| Single-Dataset `open_dataset(group="/VCP-12/sweep_0")` | ~1.3 s | n/a* | — |

\* Glob filtering and non-recursive single-Dataset opens are
rustytree-specific surface (xarray PR #11302's glob support is
not yet upstream); the comparison is against the full-tree open
that would otherwise be required.

The S3 win has multiple stacked drivers, each shipped as its own PR
(see `CHANGELOG.md`):

- **Cross-group parallelism** ([#12]): xarray opens 107 groups
  sequentially through `IcechunkStore`'s `SyncMixin`, paying ~470 ms
  per group. The recursive walk fans them out through one tokio
  runtime with `try_join_all` and a 32-permit semaphore.
- **Probe pipelining** ([#13]): the icechunk-vs-vanilla auto-detect
  HEAD on `<prefix>/repo` races alongside `Repository::open` rather
  than running before it.
- **Parallel eager fanout for self-named dim coords** ([#15]):
  xarray's `_maybe_create_default_indexes` post-pass reads every
  self-named 1-D coord on every node to build pandas Index objects.
  Pre-fetched in parallel from Rust (gated on the same semaphore)
  it's resident memory by the time xarray touches it.
- **Metadata-only datetime dtype inference** ([#15]): a context-
  manager monkey-patch swaps xarray's `_decode_cf_datetime_dtype`
  for a version that synthesises example values instead of peeking
  `arr[0]` and `arr[-1]`. Mirrors xarray PR #11304's idea;
  removable once that lands upstream.
- **icechunk snapshot fast path** ([#19]): for icechunk inputs,
  metadata is read from the in-memory snapshot rather than via
  per-node `Group::async_open` / `Array::async_open` round-trips —
  ~1.5 s → ~350 ms on KLOT.
- **Non-recursive walk for `open_dataset(group=literal)`** ([#22]):
  skip walking siblings/descendants when the user wants only one
  group as a flat Dataset.
- **Glob pre-discovery prune** ([#25]): glob `group=` patterns
  compile to a conservative prefix predicate; subtrees that can't
  match are skipped before `Group::async_open` (vanilla) or
  `Array::new_with_metadata` (icechunk). On KLOT glob, ~1.8× over
  the post-walk-only filter.

## Lazy chunk reads

Every var dict produced by `_rustytree.open_datatree(...)` carries a
`handle` field — a `ZarrsArrayHandle` PyO3 class wrapping the opened
`zarrs::Array`. The walk was already paying for the `Array::async_open`
calls (to populate `VarMeta`); the handle keeps the resulting
`Arc<Array>` alive instead of dropping it.

`ZarrsArrayHandle` exposes:

- `shape`, `dtype` (canonical NumPy string), `chunks`
- `read_subset(ranges)` where `ranges = list[tuple[int, int]]` (one
  `(start, stop)` per dim, exclusive stop). Runs
  `runtime.block_on(array.async_retrieve_array_subset_elements::<T>(...))`
  with the GIL released via `Python::detach`. Returns a 1-D NumPy
  array of the matching primitive type.

Python-side, `RustyBackendArray(BackendArray)` adapts xarray's
indexing protocol to the handle:

- Advertises `IndexingSupport.BASIC` (slice + scalar-int per axis).
  Fancy indexing flows through xarray's
  `explicit_indexing_adapter` which decomposes it into a sequence of
  basic reads.
- `_raw_indexing_method` translates xarray's tuple-of-(slice|int)
  into `(start, stop)` ranges, calls `handle.read_subset(ranges)`,
  reshapes, and squeezes axes selected by integer indexers.

The `RustytreeBackendEntrypoint` builds a `_RustyDataStore` shim
around each `NodeData` and delegates to xarray's
`StoreBackendEntrypoint.open_dataset`, which runs the standard
`decode_cf_variables` + data/coord split + encoding/`set_close`
plumbing — no decode logic is reimplemented in rustytree.
