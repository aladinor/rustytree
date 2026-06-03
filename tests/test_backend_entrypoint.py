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


def test_get_variables_decodes_base64_fill_value() -> None:
    """A `_FillValue` carried in the Zarr attributes as the base64 raw
    fill-value wire form (as some virtual/icechunk stores emit) must be
    decoded to a numeric value, mirroring xarray's zarr backend
    """
    import base64
    import struct

    from rustytree.backend import _RustyDataStore

    fill = 1e20
    # FillValueCoder encodes float fills as little-endian float64 bytes.
    b64 = base64.standard_b64encode(struct.pack("<d", fill)).decode()

    class _StubHandle:
        chunks = (2,)
        dtype = "float32"

    node = {
        "attrs": {},
        "vars": [
            {
                "name": "v",
                "dims": ["x"],
                "data": np.array([1.0, 2.0], dtype="float32"),
                "handle": _StubHandle(),
                "attrs": {"_FillValue": b64, "missing_value": fill},
            }
        ],
    }

    variables = _RustyDataStore(node).get_variables()
    fv = variables["v"].attrs["_FillValue"]
    assert isinstance(fv, float) and fv == fill
    # numeric `missing_value` is left untouched
    assert variables["v"].attrs["missing_value"] == fill


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


def test_literal_group_no_leading_slash_icechunk(
    multilevel_icechunk_repo: Path,
) -> None:
    """xarray's convention is absolute group paths, but users commonly
    write them without a leading slash (``group="volume_a/sweep_0"``).
    icechunk's path validator strictly requires a leading ``/`` and was
    raising ``invalid root path`` until we normalized literal paths on
    the Python side. Regression test exercises the icechunk fast-path
    specifically — vanilla Zarr v3 accepts relative paths silently, so
    a vanilla test would not catch the original failure mode.
    """
    import icechunk

    rusty = xr.open_dataset(
        str(multilevel_icechunk_repo),
        engine="rustytree",
        group="volume_a/sweep_0",
    )
    storage = icechunk.local_filesystem_storage(str(multilevel_icechunk_repo))
    repo = icechunk.Repository.open(storage)
    session = repo.readonly_session("main")
    zarr_ds = xr.open_dataset(
        session.store,
        engine="zarr",
        consolidated=False,
        group="/volume_a/sweep_0",
    )
    xr.testing.assert_identical(rusty, zarr_ds)


# ---- xarray-compat kwargs: zarr_format / consolidated ----


@pytest.mark.parametrize(
    "kwargs",
    [
        {},                                     # neither passed
        {"zarr_format": None},                  # explicit None
        {"zarr_format": 3},                     # explicit v3
        {"consolidated": None},                 # explicit None
        {"consolidated": False},                # the v3 path users commonly pass
        {"zarr_format": 3, "consolidated": False},  # both together
    ],
)
def test_v3_compatible_kwargs_pass_through(
    multilevel_zarr_store: Path, kwargs: dict
) -> None:
    """xarray's stock `engine="zarr"` accepts `zarr_format` and
    `consolidated` kwargs. We accept them too for call-site
    compatibility — anything that implies Zarr v3 (or unspecified)
    must pass through silently and produce the same result as the
    no-kwarg call.
    """
    dt = xr.open_datatree(
        str(multilevel_zarr_store), engine="rustytree", **kwargs
    )
    assert isinstance(dt, xr.DataTree)
    assert sum(1 for _ in dt.subtree) > 1


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"zarr_format": 2}, "Zarr v3 only"),
        # Any non-3 zarr_format is rejected — covers hypothetical v4+
        # as well, not just v2. Keeps the "v3 only" contract explicit.
        ({"zarr_format": 4}, "Zarr v3 only"),
        ({"consolidated": True}, "consolidated metadata"),
        # Both v2-implying kwargs together: zarr_format check fires first.
        ({"zarr_format": 2, "consolidated": True}, "Zarr v3 only"),
    ],
)
def test_v2_implying_kwargs_rejected(
    multilevel_zarr_store: Path, kwargs: dict, match: str
) -> None:
    """`zarr_format=2` and `consolidated=True` both imply the Zarr v2
    path, which rustytree does not support. The entrypoint must
    reject them with a clear `NotImplementedError` pointing the user
    at `engine="zarr"`."""
    with pytest.raises(NotImplementedError, match=match):
        xr.open_datatree(
            str(multilevel_zarr_store), engine="rustytree", **kwargs
        )


def test_v2_implying_kwargs_rejected_open_dataset(
    multilevel_zarr_store: Path,
) -> None:
    """Same v2 rejection applies to `open_dataset`."""
    with pytest.raises(NotImplementedError, match="Zarr v3 only"):
        xr.open_dataset(
            str(multilevel_zarr_store),
            engine="rustytree",
            zarr_format=2,
            group="/volume_a",
        )


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


def test_glob_prune_drops_nonmatching_subtree_vanilla(
    multilevel_zarr_store: Path,
) -> None:
    """Phase 8/2: the `glob=` kwarg passes a conservative prefix
    predicate into the Rust walk to skip subtrees that can't match.
    The dict returned by Rust (BEFORE Python's `_filter_by_glob`)
    should already be smaller than the unfiltered walk — proves the
    prune fired rather than the Python post-filter doing all the
    work.
    """
    from rustytree._rustytree import open_datatree as _rust_open

    full = _rust_open(str(multilevel_zarr_store))
    pruned = _rust_open(str(multilevel_zarr_store), glob="/*/sweep_0")
    assert "/volume_a/sweep_1" in full
    assert "/volume_a/sweep_1" not in pruned
    assert "/volume_a/sweep_0" in pruned


def test_glob_prune_drops_nonmatching_subtree_icechunk(
    multilevel_icechunk_repo: Path,
) -> None:
    """Same prune-fired check for the icechunk snapshot walker."""
    from rustytree._rustytree import open_datatree as _rust_open

    full = _rust_open(str(multilevel_icechunk_repo))
    pruned = _rust_open(str(multilevel_icechunk_repo), glob="/*/sweep_0")
    assert "/volume_a/sweep_1" in full
    assert "/volume_a/sweep_1" not in pruned
    assert "/volume_a/sweep_0" in pruned


@pytest.mark.parametrize(
    "pattern",
    [
        "/*/sweep_0",
        "/*/sweep_*",
        "/volume_*/sweep_0",
        "/*/*",
        # Pathological pattern with `//` — covers the parse-side bug
        # where empty pattern segments would otherwise produce a Rust
        # false negative. `PurePosixPath` coalesces `//`, so we must
        # too.
        "/volume_a//sweep_0",
    ],
)
def test_glob_prune_preserves_python_filter_truth(
    multilevel_zarr_store: Path, pattern: str
) -> None:
    """Parity sweep: for any pattern, the Rust prune (`glob=`) must
    not drop a path that Python's `_filter_by_glob` would have
    accepted. Concretely: `_filter_by_glob(rust_full_walk, pattern)`
    must equal `_filter_by_glob(rust_pruned_walk, pattern)`. False
    negatives in the Rust prune (silent data loss) would surface as
    a diff between these two sets.
    """
    from rustytree._rustytree import open_datatree as _rust_open
    from rustytree.backend import _filter_by_glob

    full = _rust_open(str(multilevel_zarr_store))
    pruned = _rust_open(str(multilevel_zarr_store), glob=pattern)

    assert _filter_by_glob(full, pattern).keys() == _filter_by_glob(
        pruned, pattern
    ).keys()


@pytest.mark.parametrize(
    "pattern",
    [
        "/*/sweep_0",
        "/*/sweep_*",
        "/volume_*/sweep_0",
        "/*/*",
        "/volume_a//sweep_0",
    ],
)
def test_glob_prune_preserves_python_filter_truth_icechunk(
    multilevel_icechunk_repo: Path, pattern: str
) -> None:
    """Same parity sweep as the vanilla version, against the icechunk
    snapshot walker. The icechunk path applies the predicate at a
    different call site (snapshot filter rather than `discover_paths`
    recursion), so it could regress independently.
    """
    from rustytree._rustytree import open_datatree as _rust_open
    from rustytree.backend import _filter_by_glob

    full = _rust_open(str(multilevel_icechunk_repo))
    pruned = _rust_open(str(multilevel_icechunk_repo), glob=pattern)

    assert _filter_by_glob(full, pattern).keys() == _filter_by_glob(
        pruned, pattern
    ).keys()


def test_open_datatree_empty_group_treated_as_root(
    multilevel_zarr_store: Path,
) -> None:
    """``group=""`` is a corner-case that occasionally shows up in
    user code; xarray's convention is to treat it as root. Without
    normalisation it reaches Rust as the empty string and surfaces
    as a path-validation error. Pin the "treat as root" behaviour."""
    dt = xr.open_datatree(str(multilevel_zarr_store), engine="rustytree", group="")
    expected = xr.open_datatree(str(multilevel_zarr_store), engine="rustytree")
    assert {n.path for n in dt.subtree} == {n.path for n in expected.subtree}


def test_open_datatree_literal_group_collapses_double_slash(
    multilevel_zarr_store: Path,
) -> None:
    """A literal `group="/volume_a//sweep_0"` should resolve the same
    as `/volume_a/sweep_0`. The `//` collapse fix in Rust's
    `GlobPredicate::parse` covered globs only; without the matching
    Python-side `PurePosixPath` canonicalisation, the user-supplied
    path stays as `/volume_a//sweep_0` while Rust marshals nodes
    with the canonical form, missing the lookup → `KeyError`. End-
    to-end smoke caught this; this regression test pins the fix.
    """
    dt = xr.open_datatree(
        str(multilevel_zarr_store),
        engine="rustytree",
        group="/volume_a//sweep_0",
    )
    paths = sorted(n.path for n in dt.subtree)
    # `_reroot` strips `/volume_a/sweep_0` so the leaf lands at "/".
    assert paths == ["/"], paths


def test_open_datatree_literal_group_strips_trailing_slash(
    multilevel_zarr_store: Path,
) -> None:
    """`group="/volume_a/"` (trailing slash) should resolve the same
    as `/volume_a`. PurePosixPath strips trailing slashes."""
    dt = xr.open_datatree(
        str(multilevel_zarr_store), engine="rustytree", group="/volume_a/"
    )
    paths = sorted(n.path for n in dt.subtree)
    assert paths == ["/", "/sweep_0", "/sweep_1"], paths


def test_glob_group_character_class(multilevel_zarr_store: Path) -> None:
    """`PurePosixPath.match` supports `[...]` character classes; the
    fixture has `sweep_0` and `sweep_1` so `[01]` should match both.
    Our Rust prune bails to "no prune" for patterns containing `[`
    (the Python filter remains authoritative), so this test exercises
    the fallback path."""
    dt = xr.open_datatree(
        str(multilevel_zarr_store), engine="rustytree", group="*/sweep_[01]"
    )
    paths = sorted(n.path for n in dt.subtree)
    assert paths == [
        "/",
        "/volume_a",
        "/volume_a/sweep_0",
        "/volume_a/sweep_1",
    ], paths


def test_glob_group_question_mark(multilevel_zarr_store: Path) -> None:
    """`PurePosixPath.match` supports `?` as single-char wildcard.
    Same Rust-prune bail-out as `[...]` — exercises the fallback."""
    dt = xr.open_datatree(
        str(multilevel_zarr_store), engine="rustytree", group="*/sweep_?"
    )
    paths = sorted(n.path for n in dt.subtree)
    assert paths == [
        "/",
        "/volume_a",
        "/volume_a/sweep_0",
        "/volume_a/sweep_1",
    ], paths


def test_open_datatree_missing_literal_group_raises_icechunk(
    multilevel_icechunk_repo: Path,
) -> None:
    """A literal `group=` that doesn't exist should raise rather than
    silently return an empty tree. Targets the icechunk fast-path
    specifically: icechunk's `Session::list_nodes(parent)` returns an
    empty iterator for non-existent paths (vs vanilla's
    `Group::async_open` which already raises). Without the post-walk
    presence check, a typo like ``group="VCP/sweep_0"`` (intended
    ``"VCP-12/sweep_0"``) would silently succeed with empty data.
    Globs are exempt — empty match is valid for them.
    """
    with pytest.raises(KeyError, match="not found"):
        xr.open_datatree(
            str(multilevel_icechunk_repo),
            engine="rustytree",
            group="/nonexistent/path",
        )


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


# ---- include_ancestor_coords (literal-group ancestor merge) ----


def test_literal_group_default_promotes_ancestor_coords(
    multilevel_zarr_store: Path,
) -> None:
    """Literal `group=/volume_a/sweep_0` with the default flag should
    surface ancestor variables on the new root: `temp` (from
    `/volume_a`) as a data_var, and `x` (from `/`, dim-coord since it's
    named after its only dim) as a coord. Without the flag the new
    root would carry only the subgroup's own `dbz`.
    """
    dt = xr.open_datatree(
        str(multilevel_zarr_store),
        engine="rustytree",
        group="/volume_a/sweep_0",
    )
    root_ds = dt.dataset
    # Subgroup's own variable.
    assert "dbz" in root_ds.data_vars
    # Promoted from `/volume_a`.
    assert "temp" in root_ds.data_vars
    # Promoted from `/`. Dim-named arrays become coords.
    assert "x" in root_ds.coords


def test_literal_group_opt_out_orphans_subtree(
    multilevel_zarr_store: Path,
) -> None:
    """`include_ancestor_coords=False` reverts to the orphaned-subtree
    contract (matches the upstream prototype's default and the
    pre-flag rustytree behavior)."""
    dt = xr.open_datatree(
        str(multilevel_zarr_store),
        engine="rustytree",
        group="/volume_a/sweep_0",
        include_ancestor_coords=False,
    )
    root_ds = dt.dataset
    assert "dbz" in root_ds.data_vars
    assert "temp" not in root_ds.data_vars
    assert "x" not in root_ds.coords


def test_literal_group_root_is_noop(multilevel_zarr_store: Path) -> None:
    """`group="/"` (or `group=None`) has no ancestors. The flag must
    not alter the resulting tree — same path set and same root
    dataset as a no-flag full-tree open."""
    with_flag = xr.open_datatree(
        str(multilevel_zarr_store),
        engine="rustytree",
        group="/",
        include_ancestor_coords=True,
    )
    no_group = xr.open_datatree(
        str(multilevel_zarr_store), engine="rustytree"
    )
    assert {n.path for n in with_flag.subtree} == {
        n.path for n in no_group.subtree
    }
    xr.testing.assert_identical(with_flag.dataset, no_group.dataset)


def test_glob_group_flag_is_noop(multilevel_zarr_store: Path) -> None:
    """The flag only fires for literal non-root groups. Glob mode
    already keeps ancestors as nodes via `_filter_by_glob`, so passing
    `include_ancestor_coords=True` with a glob must produce the same
    tree as `False`. Pins the no-op contract."""
    on = xr.open_datatree(
        str(multilevel_zarr_store),
        engine="rustytree",
        group="*/sweep_0",
        include_ancestor_coords=True,
    )
    off = xr.open_datatree(
        str(multilevel_zarr_store),
        engine="rustytree",
        group="*/sweep_0",
        include_ancestor_coords=False,
    )
    assert {n.path for n in on.subtree} == {n.path for n in off.subtree}
    for path in (n.path for n in on.subtree):
        xr.testing.assert_identical(on[path].dataset, off[path].dataset)


def test_literal_group_promotes_ancestor_coords_icechunk(
    multilevel_icechunk_repo: Path,
) -> None:
    """The icechunk path is a separate Rust walk
    (`walk_icechunk_session_snapshot`); the ancestor merge must work
    there too. Per-ancestor `self.open_dataset` calls each take the
    `recursive=False` icechunk fast-path."""
    dt = xr.open_datatree(
        str(multilevel_icechunk_repo),
        engine="rustytree",
        group="/volume_a/sweep_0",
    )
    root_ds = dt.dataset
    assert "dbz" in root_ds.data_vars
    assert "temp" in root_ds.data_vars
    assert "x" in root_ds.coords


@pytest.mark.parametrize(
    "group,sanity_coord",
    [
        ("/volume_a", "x"),                   # depth 1 → 1 ancestor (`/`)
        ("/volume_a/sweep_0", "x"),           # depth 2 → 2 ancestors
    ],
)
def test_icechunk_session_serialised_once_for_ancestor_merge(
    multilevel_icechunk_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    group: str,
    sanity_coord: str,
) -> None:
    """`include_ancestor_coords=True` against an icechunk Session must
    serialise the session snapshot exactly once, regardless of group
    depth — ancestor opens reuse the already-serialised `source` via
    `_to_rust_source`'s `bytes` short-circuit. Counter-test pins the
    contract so a future refactor that re-introduces per-ancestor
    encodes surfaces here.
    """
    import icechunk

    storage = icechunk.local_filesystem_storage(str(multilevel_icechunk_repo))
    repo = icechunk.Repository.open(storage)
    session = repo.readonly_session("main")

    session_cls = type(session.store._store.session)
    real_as_bytes = session_cls.as_bytes
    calls = 0

    def counting_as_bytes(self) -> bytes:
        nonlocal calls
        calls += 1
        return real_as_bytes(self)

    monkeypatch.setattr(session_cls, "as_bytes", counting_as_bytes)

    dt = xr.open_datatree(
        session.store,
        engine="rustytree",
        group=group,
    )
    # Sanity: the ancestor merge fired.
    assert sanity_coord in dt.dataset.coords
    assert calls == 1, (
        f"as_bytes() called {calls} times for group={group!r}; expected 1."
    )


def test_subtree_via_group_kwarg_default_unchanged_paths(
    multilevel_zarr_store: Path,
) -> None:
    """Default `True` is a content change, not a shape change. The
    existing `test_subtree_via_group_kwarg` path-set assertion must
    still hold; this test pins it explicitly with the flag named so
    the contract is searchable."""
    sub = xr.open_datatree(
        str(multilevel_zarr_store),
        engine="rustytree",
        group="/volume_a",
        include_ancestor_coords=True,
    )
    assert {n.path for n in sub.subtree} == {"/", "/sweep_0", "/sweep_1"}


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
