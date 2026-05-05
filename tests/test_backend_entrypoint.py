"""End-to-end parity tests for `xr.open_datatree(..., engine="rustytree")`.

These exercise the path that real users hit:

  - xarray plugin loader resolves `engine="rustytree"` to
    `RustytreeBackendEntrypoint` (registered via the `xarray.backends`
    entry point).
  - The entrypoint calls `_rustytree.open_datatree(...)`, wraps each
    var's `ZarrsArrayHandle` as a lazy `RustyBackendArray`, runs CF
    decoding, and assembles a real `xr.DataTree`.
  - `xr.testing.assert_identical` against `engine="zarr"` proves
    structural and value parity (dims, coords, attrs, dtypes, data).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import xarray as xr


# ---- vanilla Zarr v3 ----


def test_open_datatree_engine_rustytree_resolves(tiny_zarr_store: Path) -> None:
    dt = xr.open_datatree(str(tiny_zarr_store), engine="rustytree")
    assert isinstance(dt, xr.DataTree)


def test_root_dataset_matches_zarr(tiny_zarr_store: Path) -> None:
    rusty = xr.open_datatree(str(tiny_zarr_store), engine="rustytree")
    zarr_dt = xr.open_datatree(str(tiny_zarr_store), engine="zarr", consolidated=False)
    xr.testing.assert_identical(rusty.dataset, zarr_dt.dataset)


def test_data_round_trip_root(tiny_zarr_store: Path) -> None:
    """Materialise data and confirm element-for-element parity."""
    rusty = xr.open_datatree(str(tiny_zarr_store), engine="rustytree")
    zarr_dt = xr.open_datatree(str(tiny_zarr_store), engine="zarr", consolidated=False)
    np.testing.assert_array_equal(rusty["temp"].values, zarr_dt["temp"].values)
    np.testing.assert_array_equal(rusty["mask"].values, zarr_dt["mask"].values)


def test_multilevel_tree_matches_zarr(multilevel_zarr_store: Path) -> None:
    rusty = xr.open_datatree(str(multilevel_zarr_store), engine="rustytree")
    zarr_dt = xr.open_datatree(str(multilevel_zarr_store), engine="zarr", consolidated=False)
    # Identical paths.
    assert {n.path for n in rusty.subtree} == {n.path for n in zarr_dt.subtree}
    # Per-node parity.
    for node in rusty.subtree:
        xr.testing.assert_identical(node.dataset, zarr_dt[node.path].dataset)


def test_multilevel_data_round_trip(multilevel_zarr_store: Path) -> None:
    rusty = xr.open_datatree(str(multilevel_zarr_store), engine="rustytree")
    zarr_dt = xr.open_datatree(str(multilevel_zarr_store), engine="zarr", consolidated=False)
    for path in ("/", "/volume_a", "/volume_a/sweep_0", "/volume_a/sweep_1"):
        for var_name in rusty[path].data_vars:
            np.testing.assert_array_equal(
                rusty[path][var_name].values,
                zarr_dt[path][var_name].values,
                err_msg=f"{path}/{var_name}",
            )


def test_subtree_via_group_kwarg(multilevel_zarr_store: Path) -> None:
    sub = xr.open_datatree(
        str(multilevel_zarr_store), engine="rustytree", group="/volume_a"
    )
    assert {n.path for n in sub.subtree} == {"/", "/sweep_0", "/sweep_1"}


# ---- icechunk on local FS ----


def test_icechunk_root_matches_zarr(tiny_icechunk_repo: Path) -> None:
    """The icechunk path is auto-detected; engine="rustytree" should
    produce the same root Dataset that engine="zarr" produces against
    the icechunk session's store."""
    import icechunk

    rusty = xr.open_datatree(str(tiny_icechunk_repo), engine="rustytree")

    storage = icechunk.local_filesystem_storage(str(tiny_icechunk_repo))
    repo = icechunk.Repository.open(storage)
    session = repo.readonly_session("main")
    zarr_dt = xr.open_datatree(session.store, engine="zarr", consolidated=False)

    xr.testing.assert_identical(rusty.dataset, zarr_dt.dataset)


# ---- lazy semantics ----


def test_open_does_not_fetch_chunks(tiny_zarr_store: Path) -> None:
    """Opening the tree should be metadata-only. We can't easily count
    chunk fetches against a `LocalFileSystem` store without a counting
    wrapper, so this test verifies the indirect signal: each var's
    underlying data chain ends in a `RustyBackendArray`. xarray nests
    `LazilyIndexedArray` inside `MemoryCachedArray` and
    `CopyOnWriteArray`, so we walk the `.array` chain to find the
    `RustyBackendArray` at the bottom — its presence proves the lazy
    adapter is wired and chunks haven't been materialised.
    """
    from rustytree._array import RustyBackendArray

    dt = xr.open_datatree(str(tiny_zarr_store), engine="rustytree")
    for var in dt.dataset.data_vars.values():
        node = var.variable._data
        # Walk the `.array` chain (each lazy wrapper holds the next
        # layer in `.array`) until we find the backend array.
        while not isinstance(node, RustyBackendArray):
            assert hasattr(node, "array"), (
                f"data_var {var.name}: expected RustyBackendArray in chain, "
                f"got {type(node).__name__} with no .array"
            )
            node = node.array
        # If we got here, the chain ends in our adapter — i.e. lazy.


def test_unsupported_scheme_propagates_clearly() -> None:
    with pytest.raises(ValueError, match="unsupported URL scheme"):
        xr.open_datatree("gs://bucket/store.zarr", engine="rustytree")


# ---- _RustyDataStore shim ----


def test_rusty_data_store_satisfies_abstract_data_store(tiny_zarr_store: Path) -> None:
    """Construct `_RustyDataStore` directly from a Rust-walked node and
    confirm it honours the `AbstractDataStore` contract that
    `StoreBackendEntrypoint.open_dataset` relies on. Catches regressions
    in the shim independently of the CF-decode parity tests.
    """
    from xarray.backends.common import AbstractDataStore

    from rustytree._rustytree import open_datatree as _rust_open
    from rustytree.backend import _RustyDataStore

    tree = _rust_open(str(tiny_zarr_store))
    store = _RustyDataStore(tree["/"])

    assert isinstance(store, AbstractDataStore)

    # `load()` returns (variables, attrs) — the contract
    # `StoreBackendEntrypoint.open_dataset` calls into.
    variables, attrs = store.load()
    assert set(variables) == {"temp", "mask"}
    for var in variables.values():
        # Per-var encoding from the shim's `get_variables` flows through.
        assert "chunks" in var.encoding
        assert "preferred_chunks" in var.encoding
    assert isinstance(attrs, type(attrs))  # FrozenDict-shaped

    # Defaults inherited from `AbstractDataStore`: empty encoding, no-op close.
    assert store.get_encoding() == {}
    assert store.close() is None


# ---- non-recursive walk (Phase 5/part 3) ----


def test_open_dataset_literal_group_skips_descendants(
    multilevel_zarr_store: Path,
) -> None:
    """`open_dataset(group=literal_path)` must not walk descendants of
    that path — they'd be discarded anyway. Verify by calling Rust
    directly with `recursive=False` and asserting the result has
    exactly one node, then by exercising the entrypoint and confirming
    parity with engine="zarr".
    """
    from rustytree._rustytree import open_datatree as _rust_open

    # Direct Rust call: recursive=False must return only the requested node.
    tree = _rust_open(
        str(multilevel_zarr_store), group="/volume_a", recursive=False
    )
    assert list(tree.keys()) == ["/volume_a"], tree.keys()

    # End-to-end: open_dataset must return the same Dataset as engine="zarr".
    rusty = xr.open_dataset(
        str(multilevel_zarr_store), engine="rustytree", group="/volume_a"
    )
    zarr_ds = xr.open_dataset(
        str(multilevel_zarr_store),
        engine="zarr",
        consolidated=False,
        group="/volume_a",
    )
    xr.testing.assert_identical(rusty, zarr_ds)


def test_open_dataset_root_skips_descendants(multilevel_zarr_store: Path) -> None:
    """`open_dataset(URL)` (no group) returns root's Dataset; with
    Phase 5/part 3 it should not pay for the descendants either.
    Same parity check.
    """
    rusty = xr.open_dataset(str(multilevel_zarr_store), engine="rustytree")
    zarr_ds = xr.open_dataset(
        str(multilevel_zarr_store), engine="zarr", consolidated=False
    )
    xr.testing.assert_identical(rusty, zarr_ds)


def test_open_dataset_leaf_group(multilevel_zarr_store: Path) -> None:
    """`open_dataset(group=deep_leaf)` is the headline use case for
    Phase 5/part 3 — opens just that one leaf, no siblings."""
    rusty = xr.open_dataset(
        str(multilevel_zarr_store),
        engine="rustytree",
        group="/volume_a/sweep_0",
    )
    zarr_ds = xr.open_dataset(
        str(multilevel_zarr_store),
        engine="zarr",
        consolidated=False,
        group="/volume_a/sweep_0",
    )
    xr.testing.assert_identical(rusty, zarr_ds)


def test_open_datatree_still_walks_recursively(
    multilevel_zarr_store: Path,
) -> None:
    """`open_datatree` is the tree-shaped API and must still recurse —
    Phase 5/part 3 is scoped to `open_dataset` only."""
    dt = xr.open_datatree(str(multilevel_zarr_store), engine="rustytree")
    # Multilevel fixture produces 3+ nodes; if recursion were disabled
    # we'd get only the root.
    assert sum(1 for _ in dt.subtree) > 1


def test_open_dataset_icechunk_literal_group(tiny_icechunk_repo: Path) -> None:
    """Exercise the icechunk `recursive=false` filter path: open a
    single group from an icechunk repo via `open_dataset` and assert
    parity with `engine="zarr"` over the same session.store. Without
    this the new `walk_icechunk_session_snapshot` filter (group
    path-equality + array `parent_of` predicate) would be untested.
    """
    import icechunk

    rusty = xr.open_dataset(str(tiny_icechunk_repo), engine="rustytree")

    storage = icechunk.local_filesystem_storage(str(tiny_icechunk_repo))
    repo = icechunk.Repository.open(storage)
    session = repo.readonly_session("main")
    zarr_ds = xr.open_dataset(session.store, engine="zarr", consolidated=False)

    xr.testing.assert_identical(rusty, zarr_ds)


def test_open_dataset_glob_group_rejects() -> None:
    """Glob `group=` patterns aren't meaningful for `open_dataset`
    (multi-match, single-Dataset return). The entrypoint should raise
    NotImplementedError pointing the user at `open_datatree` rather
    than KeyError'ing through the literal-key lookup."""
    with pytest.raises(NotImplementedError, match="open_datatree"):
        xr.open_dataset(
            "/nonexistent",  # never reached; glob check is first
            engine="rustytree",
            group="*/sweep_0",
        )


# ---- glob group= filter (Phase 8) ----


@pytest.mark.parametrize(
    "pattern, expected",
    [
        # Single match → matched leaf + its ancestors.
        ("/*/sweep_0", ["/", "/volume_a", "/volume_a/sweep_0"]),
        # Multi-match siblings.
        (
            "/*/sweep_*",
            ["/", "/volume_a", "/volume_a/sweep_0", "/volume_a/sweep_1"],
        ),
    ],
)
def test_glob_group_matches(
    multilevel_zarr_store: Path, pattern: str, expected: list[str]
) -> None:
    """Glob filter result = matched paths + their ancestors so
    `DataTree.from_dict` sees a well-formed hierarchy. Mirrors xarray
    PR #11302's semantics."""
    dt = xr.open_datatree(
        str(multilevel_zarr_store), engine="rustytree", group=pattern
    )
    assert sorted(n.path for n in dt.subtree) == expected


def test_glob_group_no_matches_returns_empty(multilevel_zarr_store: Path) -> None:
    """A glob with no matches returns a DataTree with just an empty
    root node. Mirrors xarray PR #11302 behaviour."""
    dt = xr.open_datatree(
        str(multilevel_zarr_store), engine="rustytree", group="/*/sweep_99"
    )
    paths = [n.path for n in dt.subtree]
    assert paths == ["/"], paths
    assert len(dt.dataset.data_vars) == 0


def test_glob_data_round_trip(multilevel_zarr_store: Path) -> None:
    """Filtered tree's data should match the stock zarr engine's view
    of the same paths — confirms we're not mangling node contents."""
    rusty = xr.open_datatree(
        str(multilevel_zarr_store), engine="rustytree", group="/*/sweep_0"
    )
    zarr_dt = xr.open_datatree(
        str(multilevel_zarr_store), engine="zarr", consolidated=False
    )
    for path in ("/volume_a/sweep_0",):
        xr.testing.assert_identical(rusty[path].dataset, zarr_dt[path].dataset)


def test_glob_group_icechunk(multilevel_icechunk_repo: Path) -> None:
    """Glob filtering must work against the icechunk snapshot walker
    too — the filter is post-walk in Python today, so this exercises
    that path. Locks the parity now so a future Rust-side icechunk
    glob optimisation can't regress silently.
    """
    dt = xr.open_datatree(
        str(multilevel_icechunk_repo), engine="rustytree", group="/*/sweep_0"
    )
    paths = sorted(n.path for n in dt.subtree)
    assert paths == ["/", "/volume_a", "/volume_a/sweep_0"], paths


def test_glob_group_relative_pattern(multilevel_zarr_store: Path) -> None:
    """A relative pattern (no leading `/`) matches any path suffix.
    `*/sweep_0` should match `/volume_a/sweep_0` (the leading-slash
    component is unnamed). Documents the semantics inherited from
    `PurePosixPath.match` so a future tightening of validation
    surfaces here first.
    """
    dt = xr.open_datatree(
        str(multilevel_zarr_store), engine="rustytree", group="*/sweep_0"
    )
    paths = sorted(n.path for n in dt.subtree)
    assert "/volume_a/sweep_0" in paths, paths


# ---- KTWX smoke (network-free) ----

KTWX_PATH = Path("/home/alfonso-ladino/python/raw2zarr/zarr/KTWX")


@pytest.mark.skipif(
    not KTWX_PATH.exists() or os.environ.get("RUSTYTREE_SKIP_KTWX") == "1",
    reason="KTWX repo not present",
)
def test_ktwx_open_via_engine_returns_datatree() -> None:
    """Smoke test that engine="rustytree" works end-to-end on the user's
    actual radar repo. We don't assert on contents (schema may evolve);
    only that the call succeeds and returns a real `DataTree`.
    """
    dt = xr.open_datatree(str(KTWX_PATH), engine="rustytree")
    assert isinstance(dt, xr.DataTree)
    # At least the root should be present.
    assert dt.path == "/"
