# Usage

`rustytree` registers as a backend for `xr.open_datatree` /
`xr.open_dataset` via xarray's plugin discovery. After install,
`engine="rustytree"` is available with no further configuration.

## Install

The simplest path while we're pre-PyPI is `uv sync` from a clone:

```bash
git clone https://github.com/aladinor/rustytree
cd rustytree
uv sync --extra dev
```

`uv sync` runs maturin's PEP 517 hook to compile the Rust extension
and installs the Python deps in one step. Requirements: Rust 1.91.1+
(matches `Cargo.toml` `rust-version`), Python 3.12+.

For development iterations on the Rust core, rebuild in-place after
edits:

```bash
.venv/bin/maturin develop --release
```

Verify the plugin registered:

```bash
.venv/bin/python -c "import xarray as xr; assert 'rustytree' in xr.backends.list_engines()"
```

## Inputs

`engine="rustytree"` accepts the same `filename_or_obj` shapes as
`engine="zarr"`, plus the icechunk Session/Store handoff:

```python
import icechunk
import xarray as xr

# (a) Pre-opened icechunk Session — pass session.store. Same shape
#     as engine="zarr" against icechunk.
storage = icechunk.local_filesystem_storage("/path/to/repo")
repo = icechunk.Repository.open(storage)
session = repo.readonly_session("main")
dt = xr.open_datatree(session.store, engine="rustytree")

# (b) Local-FS icechunk repo by path — convenience for zero-credential
#     dev. Auto-detected via the `<root>/repo` + `<root>/snapshots/`
#     layout heuristic.
dt = xr.open_datatree("/path/to/repo", engine="rustytree", branch="main")

# (c) Vanilla Zarr v3 (path or s3:// URL).
dt = xr.open_datatree("s3://bucket/store.zarr", engine="rustytree",
                      storage_options={"region": "us-east-1"})

# (d) Pre-constructed icechunk Session object (not session.store).
dt = xr.open_datatree(session, engine="rustytree")
```

For remote icechunk repos (S3, GCS, Azure), construct the
`icechunk.Repository` and `Session` yourself and pass `session.store`
as in (a). rustytree does **not** auto-detect remote icechunk URLs —
the user controls the credentials, branch, and cache config via
icechunk's own API.

## `group=` patterns

`group=` accepts a literal path, a glob pattern, or `None` (root).

```python
# Literal subtree — Rust skips walking siblings/descendants outside
# this path. Leading slash is optional (we normalise).
dt = xr.open_datatree(session.store, engine="rustytree",
                      group="/VCP-12")

# Glob (xarray PR #11302 semantics, via PurePosixPath.match):
#   *  zero or more non-slash chars
#   ?  one char
#   [...]  character class
# Matched paths plus their ancestors are kept so the result tree is
# well-formed.
dt = xr.open_datatree(session.store, engine="rustytree",
                      group="/*/sweep_0")        # one sweep per VCP
dt = xr.open_datatree(session.store, engine="rustytree",
                      group="*/sweep_[01]")      # sweep_0 and sweep_1
```

`xr.open_dataset(..., group=)` accepts only **literal** paths — a
single Dataset is the wrong return shape for a multi-match glob.
Passing a glob to `open_dataset` raises `NotImplementedError` pointing
at `open_datatree`.

## Lazy reads

Variables come back as `LazilyIndexedArray(RustyBackendArray)` —
chunks are not fetched until xarray asks for them (e.g.
`ds.foo.values`, `ds.compute()`, indexing, dask scheduling). The
exception: 1-D self-named dim coords are **eagerly** fetched during
the walk (Phase C) so xarray's `_maybe_create_default_indexes`
doesn't fan out N serial chunk reads on open.

## Errors

| Condition | Error |
|---|---|
| `group=` literal path not present in the store | `KeyError` |
| `open_dataset(group=)` with a glob pattern | `NotImplementedError` |
| `s3://` URL with malformed `storage_options` (unknown key) | `ValueError` |
| `bytes` input that's not a valid icechunk session blob | `ValueError` |
| Filesystem / network I/O failure | `OSError` |
| Other Rust-side failure (catch-all) | `RuntimeError` |

Glob patterns that match no paths return a DataTree with just an
empty root node — empty match is valid for globs (mirrors xarray PR
#11302).

## Performance

The win profile depends on backend and workload:

- **icechunk on S3**: dominant win on metadata. icechunk's snapshot
  is fetched once and parsed in Rust; the per-group Python `SyncMixin`
  + GIL ping-pong of `engine="zarr"` is gone. Headline KLOT
  cold-cache: full tree (107 nodes) ≈ 2 s; glob `/*/sweep_0`
  (15 nodes) ≈ 1.4 s. `engine="zarr"` on the same workload is ~50 s.

- **vanilla Zarr v3 on object storage**: discovery walk runs
  concurrently in async; `engine="zarr"`'s sequential per-group open
  is the bottleneck we collapse.

- **Lazy chunk reads** (post-open data fetches) inherit zarrs's
  internal chunk-decode parallelism and release the GIL inside
  `block_on`, so concurrent loads from a thread pool actually
  overlap.

The full performance table and methodology will land with Phase 9
(benchmarks).

## Run the test suite

```bash
.venv/bin/pytest tests/        # Python integration tests
cargo test --lib               # Rust unit tests
```
