# rustytree

> Rust-backed xarray DataTree backend for fast Zarr (incl. icechunk) access from object storage.

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
inference), and the recursive multi-node walk are all in. See
[`CHANGELOG.md`](CHANGELOG.md) for the per-PR breakdown.

Quickest way to see it in action: open `notebooks/klot_demo.ipynb`,
which compares `engine="zarr"` and `engine="rustytree"` against the
public anonymous NEXRAD KLOT icechunk repo on S3.

## Quick start

`rustytree` is a Rust extension built via [maturin](https://www.maturin.rs/);
the [`uv`](https://docs.astral.sh/uv/) workflow below sets up a Python venv
that contains the compiled extension plus the Python deps the notebooks
need.

```bash
# Clone + enter the repo
git clone https://github.com/aladinor/rustytree.git
cd rustytree

# Create a Python 3.12+ venv and install dev deps + maturin
uv venv --python 3.12
uv pip install --python .venv/bin/python maturin pytest pytest-cov \
    'zarr>=3.0' 'icechunk>=2.0' 'xarray>=2024.10' jupyter

# Build the Rust extension into the venv (release for benchmarks;
# omit `--release` for faster iterations during development).
.venv/bin/maturin develop --release

# Run the test suite to confirm the install works
.venv/bin/pytest tests/

# Launch Jupyter and open the demo notebook
.venv/bin/jupyter lab notebooks/klot_demo.ipynb
```

The notebooks and tests don't require AWS credentials — KLOT is read
anonymously via icechunk's anonymous S3 path. Network speed determines
cold-cache timings (expect ~2–5 s for `engine="rustytree"` on KLOT
from a home connection; ~50 s for `engine="zarr"` on the same).

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — what the project is and why
  the Rust backend wins.
- [`docs/usage.md`](docs/usage.md) — how to build it, run the test suite, and
  the planned API surface.
- [`docs/contributing.md`](docs/contributing.md) — branching, commit
  conventions, validation gates, per-PR audit.
- [`docs/release-process.md`](docs/release-process.md) — versioning, tag,
  CHANGELOG roll.

## License

[GNU Affero General Public License v3.0 or later](LICENSE) (AGPL-3.0-or-later).

If you use, modify, or run this software — including over a network as part
of a hosted service — you must make the corresponding source code available
under the same license to anyone interacting with it. See section 13 of the
LICENSE for the network-use clause. If you need different terms (e.g. for
proprietary embedding), open an issue to discuss.

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
