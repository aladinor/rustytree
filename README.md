<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/logo-banner-dark.png">
  <img src="assets/logo-banner-light.png" alt="rustytree — Rust-backed xarray DataTree backend" width="720">
</picture>

[![CI](https://github.com/aladinor/rustytree/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/aladinor/rustytree/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/rustytree-xarray.svg)](https://pypi.org/project/rustytree-xarray/)
[![Python versions](https://img.shields.io/pypi/pyversions/rustytree-xarray.svg)](https://pypi.org/project/rustytree-xarray/)
[![License](https://img.shields.io/pypi/l/rustytree-xarray.svg)](https://github.com/aladinor/rustytree/blob/main/LICENSE)

`rustytree` registers as `xr.open_datatree(engine="rustytree")` and walks Zarr
v3 hierarchies — both icechunk-backed and vanilla — concurrently in async
across one FFI crossing. The goal is a drop-in faster replacement for
`xr.open_datatree(engine="zarr")`, especially on icechunk repos served from
object storage where xarray's current per-group sequential decoding is the
dominant cost.

## Status

**Alpha.** `xr.open_datatree(..., engine="rustytree")` works end-to-end
against both icechunk repositories (pass `session.store`) and vanilla
Zarr v3 stores (path or `s3://` URL). Lazy chunk reads via
`RustyBackendArray`, CF decoding (incl. metadata-only datetime dtype
inference), the recursive multi-node walk, glob `group=` filtering, and
non-recursive single-Dataset opens are all in. See
[`CHANGELOG.md`](CHANGELOG.md) for the per-PR breakdown.

## Compatibility

| Surface | Supported | Not supported |
|---|---|---|
| **Zarr format** | v3 only | v2 — passing `zarr_format=2` or `consolidated=True` raises `NotImplementedError`; use stock `engine="zarr"` for v2 stores |
| **icechunk** | `icechunk>=2.0` (the current major) | older icechunk releases |
| **Python** | 3.12, 3.13 | older versions |
| **Platforms** | manylinux x86_64, manylinux aarch64, macOS arm64 (wheels); other platforms via sdist | Windows / macOS Intel / linux musl wheels (build from source via sdist) |

## Install

Install from PyPI (the import name stays `rustytree`; the
distribution name on PyPI is `rustytree-xarray` because `rustytree`
collides with an unrelated dormant package):

```bash
pip install rustytree-xarray
```

Or install from source per **Quick start** below.

## Example

Open the public anonymous NEXRAD KLOT icechunk repo on S3:

```python
import icechunk
import xarray as xr

storage = icechunk.s3_storage(
    bucket="nexrad-arco", prefix="KLOT",
    region="us-east-1", anonymous=True,
)
session = icechunk.Repository.open(storage).readonly_session("main")

# Open the full DataTree (107 nodes — every VCP × sweep combination).
dt = xr.open_datatree(session.store, engine="rustytree")

# Or grab one specific sweep as a flat Dataset (skips walking siblings):
ds = xr.open_dataset(session.store, engine="rustytree",
                     group="/VCP-12/sweep_0")

# Or apply a glob pattern. "Give me sweep_0 from every VCP" is the
# canonical radar workflow — returns a tree filtered to those matches,
# with the VCP container groups auto-included as ancestors.
sweeps_0 = xr.open_datatree(session.store, engine="rustytree",
                            group="/*/sweep_0")
```

`engine="rustytree"` is a drop-in replacement for `engine="zarr"` —
same `xr.open_datatree` / `xr.open_dataset` entry points, same
`storage_options` / `decode_*` kwargs. The full demo (with side-by-side
timings against `engine="zarr"`) is in
[`notebooks/klot_demo.ipynb`](notebooks/klot_demo.ipynb).

## Quick start

`rustytree` is a Rust extension built via [maturin](https://www.maturin.rs/);
[`uv sync`](https://docs.astral.sh/uv/) handles the venv, Python deps,
and the Rust build (via maturin's PEP 517 hook) in one command.

```bash
# Clone + enter the repo
git clone https://github.com/aladinor/rustytree.git
cd rustytree

# Install everything: venv, deps, and the compiled Rust extension.
uv sync --extra dev

# Confirm the install
.venv/bin/pytest tests/

# Launch Jupyter and open the demo notebook
.venv/bin/jupyter lab notebooks/klot_demo.ipynb
```

The notebooks and tests don't require AWS credentials — KLOT is read
anonymously via icechunk's anonymous S3 path. Network speed determines
cold-cache timings (expect ~1–3 s for `engine="rustytree"` on KLOT
from a home connection; ~50 s for `engine="zarr"` on the same).

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — what the project is and why
  the Rust backend wins.
- [`docs/usage.md`](docs/usage.md) — how to install, run the test suite, and
  the API surface (inputs, `group=` patterns, lazy reads, errors).
- [`docs/contributing.md`](docs/contributing.md) — branching, commit
  conventions, validation gates, per-PR audit.
- [`docs/release-process.md`](docs/release-process.md) — versioning, tag,
  CHANGELOG roll.

## License

[Apache License 2.0](LICENSE) (Apache-2.0).

## Acknowledgements

Built on [`zarrs`](https://github.com/zarrs/zarrs) +
[`zarrs_icechunk`](https://github.com/zarrs/zarrs_icechunk) +
[`icechunk`](https://github.com/earth-mover/icechunk).
Sibling project [`radish`](https://github.com/aladinor/radish) proved the
PyO3 + xarray-backend pattern. The bottlenecks this project sets out to
collapse are tracked in xarray PRs
[#10742](https://github.com/pydata/xarray/pull/10742),
[#11304](https://github.com/pydata/xarray/pull/11304), and
[#11302](https://github.com/pydata/xarray/pull/11302).
