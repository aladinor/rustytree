# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Until the first tagged release, every PR appends to `[Unreleased]`. On
release, that section is renamed to `[x.y.z] - YYYY-MM-DD` and a fresh
`[Unreleased]` block is started.

## [Unreleased]

### Added

- Project scaffold ([#1]): maturin-built PyO3 cdylib registered as an
  xarray backend; `xr.open_datatree(engine="rustytree")` resolves to a
  stub raising `NotImplementedError` until the walk is implemented.
- 9-test pytest suite locking the plugin-discovery contract ([#1]).
- `[lints.clippy]` policy: `pedantic = warn`, `perf = deny`,
  `unwrap_used = warn` ([#1]).
- `CHANGELOG.md`, `README.md`, and `docs/` folder ([#4]):
  `architecture.md`, `usage.md`, `contributing.md`, `release-process.md`.
- CI workflow ([#5]): `.github/workflows/ci.yml` runs `cargo fmt`,
  `cargo clippy -D warnings`, `cargo test`, `maturin develop`, and
  `pytest` against Python 3.10 and 3.12 on every push to `main` and every
  pull request.

### Changed

- Toolchain bumped ([#3]): `rust-version` 1.75 → 1.91.1, `edition`
  2021 → 2024 (matches `icechunk`'s MSRV ahead of the next milestone).
- Project license switched from `MIT OR Apache-2.0` to
  `AGPL-3.0-or-later` ([#5]). If you use, modify, or run rustytree —
  including over a network as part of a hosted service — you must make
  the corresponding source code available under the same license to
  anyone interacting with it. See `LICENSE` section 13 for the
  network-use clause. The previous `LICENSE-MIT` and `LICENSE-APACHE`
  files are removed.

[Unreleased]: https://github.com/aladinor/rustytree/compare/...HEAD
[#1]: https://github.com/aladinor/rustytree/pull/1
[#3]: https://github.com/aladinor/rustytree/pull/3
[#4]: https://github.com/aladinor/rustytree/pull/4
[#5]: https://github.com/aladinor/rustytree/pull/5
