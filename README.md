# rustytree

> Rust-backed xarray DataTree backend for fast Zarr (incl. icechunk) access from object storage.

`rustytree` registers as `xr.open_datatree(engine="rustytree")` and walks Zarr
v3 hierarchies — both icechunk-backed and vanilla — concurrently in async
across one FFI crossing. The goal is a drop-in faster replacement for
`xr.open_datatree(engine="zarr")`, especially on icechunk repos served from
object storage where xarray's current per-group sequential decoding is the
dominant cost.

## Status

**Pre-alpha.** Phase 1 + Phase 1.5 shipped — only plugin discovery is wired
up; calling `xr.open_datatree(..., engine="rustytree")` raises
`NotImplementedError` until Phase 2 lands the real walk.

Track progress in [`CHANGELOG.md`](CHANGELOG.md). Architecture and design
choices live in [`docs/architecture.md`](docs/architecture.md).

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — what the project is and why
  the Rust backend wins.
- [`docs/usage.md`](docs/usage.md) — how to build it, run the test suite, and
  the planned API surface.
- [`docs/contributing.md`](docs/contributing.md) — branching, commit
  conventions, audit chain, validation gates.
- [`docs/release-process.md`](docs/release-process.md) — versioning,
  CHANGELOG roll, tag + PyPI workflow.

## License

Dual-licensed under [MIT](LICENSE-MIT) **or** [Apache-2.0](LICENSE-APACHE).
You may choose either.

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
