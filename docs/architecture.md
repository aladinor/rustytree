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
will collapse that stack by owning the read path in Rust: one tokio runtime,
one FFI crossing with the GIL released, and `try_join_all` over async store
operations under the hood. Phase 1 / Phase 1.5 ship only the plugin scaffold;
the walk lands in Phase 2.

## Primary target: icechunk-backed Zarr v3

The dominant use case is icechunk-backed Zarr v3. icechunk is **not** a
regular Zarr v3 store — chunks live at content-addressed hashed paths and
metadata lives inline in snapshot files, not at canonical `c/0/0` keys.
Reading icechunk requires icechunk's storage layer.

The official Rust adapter [`zarrs_icechunk`](https://github.com/zarrs/zarrs_icechunk)
exposes `AsyncIcechunkStore::new(session)` returning an
`AsyncReadableStorageTraits` impl. `rustytree` will land on
`icechunk = "2"`, `zarrs = "0.22"`, `zarrs_icechunk = "0.5"` in Phase 2.

icechunk reframes the bottleneck profile:

- **Snapshot file IS the consolidated metadata.** One GET fetches every
  group's `zarr.json` inline. Earthmover explicitly recommends
  `consolidated=False` against icechunk because the snapshot already plays
  that role.
- The N-groups × RTT problem (#10579) shrinks to N-groups × *parse* —
  parallelizable in Rust without GIL ping-pong.
- Per-coord round-trips and CF datetime decoding round-trips should remain
  real wins (to be confirmed by Phase 9 benchmarks).

## Polymorphic engine

`engine="rustytree"` resolves both icechunk and vanilla Zarr v3 stores
polymorphically at the store boundary, the same way `engine="zarr"` does
today (icechunk's Python `IcechunkStore` implements zarr-python's Store
protocol).

| `filename_or_obj`                              | Rust store impl                              |
| --- | --- |
| Python `IcechunkStore` / `Session`             | unwrap to icechunk `Session` → `AsyncIcechunkStore` |
| Path/URL pointing at an icechunk repo          | `Repository::open` → `readonly_session` → `AsyncIcechunkStore` |
| Path/URL pointing at a plain Zarr v3 store     | `zarrs_object_store::AsyncObjectStore` |
| `MutableMapping` / fsspec object               | fall back to `zarrs_object_store` |

Detection cues for paths: presence of `<root>/refs/` and `<root>/snapshots/`
(icechunk layout), or an `icechunk://` URL prefix as explicit override.

## Module map (post-Phase 2)

Today only `src/lib.rs` and `python/rustytree/{__init__.py, backend.py}`
exist. The layout below is the Phase 2 target.

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
Python: xr.open_datatree(filename_or_obj, engine="rustytree", branch="main", ...)
   │
   ▼  (xarray plugin loader → entry point)
RustytreeBackendEntrypoint.open_datatree
   │
   ├──── Inspect filename_or_obj type at the Python boundary
   │
   ▼  (one PyO3 call, GIL released around block_on)
_rustytree.open_datatree(...)
   │
   ▼  Rust: runtime.block_on(walk::open_datatree(...))
1. store::build_store(...)                       → Arc<dyn AsyncReadable…>
2. walk::recursive(store, root, semaphore)       → Vec<NodeData>
   (try_join_all over list_dir + Group::async_open + Array::async_open)
3. For each NodeData: parallel fetch of 1D self-named coordinate chunks
4. Build PyDict { path: PyNodeData {...} }
   │
   ▼  Back in Python
5. _decode.node_to_dataset(node) for each path
6. datatree_from_dict_with_io_cleanup(groups_dict)  → xr.DataTree
```

## Performance targets

Validated by Phase 9 benchmarks (`benchmarks/bench_open_datatree.py`):

- ≥ 3× cold-cache speedup vs. `engine="zarr"` on a local icechunk DataTree.
- ≥ 5× warm-cache speedup on the same.
- ≥ 5× speedup on a 100-group vanilla Zarr v3 store on remote S3.

## Lazy chunk reads

`RustyBackendArray(BackendArray)` (Python) holds a `ZarrsArrayHandle` (PyO3
class). Its `_raw_indexing_method` releases the GIL, calls `handle.read_subset(...)`
which runs `runtime.block_on(array.async_retrieve_array_subset(...))` and
returns a NumPy ndarray. xarray wraps it in `LazilyIndexedArray` automatically —
same lazy semantics as `engine="zarr"`.
