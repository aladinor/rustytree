# Usage

> **Status:** Phase 1 / Phase 1.5 shipped — plugin discovery only. The real
> hierarchy walk, lazy reads, and decoding land in Phase 2 onward. Calls to
> `xr.open_datatree(..., engine="rustytree")` raise `NotImplementedError`
> for now. Track progress in [`CHANGELOG.md`](../CHANGELOG.md).

## Install (development)

`rustytree` builds as a Python wheel via [maturin](https://www.maturin.rs/).
Requirements:

- Rust 1.91.1+ (matches `Cargo.toml` `rust-version`)
- Python 3.10+
- `uv` (recommended) or `pip` + a virtualenv

```bash
git clone https://github.com/aladinor/rustytree
cd rustytree
uv venv
uv pip install --python .venv/bin/python maturin
.venv/bin/maturin develop                                          # builds + installs the wheel
uv pip install --python .venv/bin/python -e ".[dev]"               # pulls pytest + zarr + ruff
.venv/bin/python -c "import xarray; assert 'rustytree' in xarray.backends.list_engines()"
```

## Run the test suite

```bash
.venv/bin/pytest tests/        # 9 tests as of Phase 1
cargo test                     # Rust unit tests (none yet — Phase 2 adds them)
```

## Planned API (Phase 2+)

Once Phase 2 lands, the engine will be polymorphic over icechunk and vanilla
Zarr v3 — same call shape as `engine="zarr"` works today:

```python
import xarray as xr
import icechunk

# (a) Pre-opened icechunk session
storage = icechunk.local_filesystem_storage("/path/to/repo")
repo = icechunk.Repository.open(storage)
session = repo.readonly_session(branch="main")
dt = xr.open_datatree(session.store, engine="rustytree")

# (b) Cold-open icechunk by path/URL
dt = xr.open_datatree("/path/to/repo", engine="rustytree", branch="main")

# (c) Vanilla Zarr v3 (no icechunk)
dt = xr.open_datatree("s3://bucket/store.zarr", engine="rustytree",
                      storage_options={"region": "us-east-1"})

# (d) Glob-filtered partial open (Phase 8)
dt = xr.open_datatree("/path/to/repo", engine="rustytree",
                      group="*/sweep_0")  # only lowest-elevation sweeps
```

## Performance

Goals and target speedups vs. `engine="zarr"` are documented in
[`architecture.md`](architecture.md); benchmarks land in Phase 9.
