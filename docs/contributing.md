# Contributing

This is a young project, so the workflow is still shaping up — open an issue
first if you're planning a non-trivial change.

## Local setup

See [`usage.md`](usage.md) for build prerequisites and the install commands.

## Validation gates

Every PR must pass these before review:

```bash
cargo fmt --check
cargo clippy --all-targets --all-features --locked -- -D warnings
cargo test
.venv/bin/maturin develop
.venv/bin/pytest tests/
```

CI runs the same chain on every push to `main` and every pull request
([`.github/workflows/ci.yml`](../.github/workflows/ci.yml)) against Python
3.10 and 3.12.

## Branching and PR workflow

- One feature per branch. Names: `feat/<topic>`, `fix/<topic>`,
  `chore/<topic>`, `docs/<topic>`, `ci/<topic>`. No commits to `main` after
  the inaugural commit.
- [Conventional Commits](https://www.conventionalcommits.org/) for messages
  (`feat:`, `fix:`, `chore:`, `docs:`, `refactor:`, `test:`, `perf:`,
  `build:`, `ci:`). Subject ≤ 72 chars; body wraps at 80.
- Sign your commits where possible.
- One PR = one logical change. Toolchain bumps, refactors, and feature work
  each get their own PR.
- PR base is always `main` unless deliberately stacking on another open PR
  (and even then, retarget to `main` before merge).

## Per-PR audit

For any PR touching Rust source or build configuration:

1. Confirm the validation gates above all pass locally.
2. Re-read the diff for: unjustified `unsafe`, swallowed errors, redundant
   clones, premature abstractions, dead code.
3. If you used AI tooling to draft the PR, surface the review findings in a
   PR comment and apply or rebut each one.

A missing audit comment is a request-changes item on substantive Rust PRs.

## CHANGELOG

Every PR appends an entry under `[Unreleased]` in
[`CHANGELOG.md`](../CHANGELOG.md) using
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) subsections
(`Added` / `Changed` / `Deprecated` / `Removed` / `Fixed` / `Security`).
A missing entry is a request-changes item.

## Releases

See [`release-process.md`](release-process.md).

## Issue triage

- Bug reports: include the `xr.open_datatree(...)` call shape, store layout
  (icechunk vs vanilla, v2 vs v3, local vs remote), Python/Rust versions,
  and the full traceback.
- Feature requests: link to the bottleneck issue/PR in upstream `pydata/xarray`
  if relevant; benchmarks help the case.
