# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Until the first tagged release, every PR appends to `[Unreleased]`. On
release, that section is renamed to `[x.y.z] - YYYY-MM-DD` and a fresh
`[Unreleased]` block is started.

## [Unreleased]

### Added

- Project scaffold ([#1], Phase 1): maturin-built PyO3 cdylib registered as
  an xarray backend; `xr.open_datatree(engine="rustytree")` resolves to a
  stub raising `NotImplementedError` until Phase 2.
- 9-test pytest suite locking the plugin-discovery contract.
- `[lints.clippy]` policy: `pedantic = warn`, `perf = deny`, `unwrap_used = warn`.
- `CHANGELOG.md` (this file), `README.md`, and `docs/` folder ([#4], Phase 1.6):
  `architecture.md`, `usage.md`, `contributing.md`, `release-process.md`.
- `LICENSE-MIT` and `LICENSE-APACHE` ([#4]).

### Changed

- Toolchain bumped ([#3], Phase 1.5): `rust-version` 1.75 → 1.91.1,
  `edition` 2021 → 2024 (icechunk's MSRV ahead of Phase 2).

[Unreleased]: https://github.com/aladinor/rustytree/compare/...HEAD
[#1]: https://github.com/aladinor/rustytree/pull/1
[#3]: https://github.com/aladinor/rustytree/pull/3
[#4]: https://github.com/aladinor/rustytree/pull/4
