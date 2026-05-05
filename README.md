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
inference), the recursive multi-node walk, glob `group=` filtering, and
non-recursive single-Dataset opens are all in. See
[`CHANGELOG.md`](CHANGELOG.md) for the per-PR breakdown.

Not on PyPI yet — install from source per **Quick start** below. Once
the first wheel ships, the install reduces to `pip install rustytree`.

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

# Or apply a glob pattern — the radar workflow that motivated Phase 8:
# "give me sweep_0 from every VCP" returns a tree filtered to those
# matches, with the VCP container groups auto-included as ancestors.
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
