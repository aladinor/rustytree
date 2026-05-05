"""Recursive walk against a local Zarr v3 fixture.

The fixture is single-level (root + 2 arrays, no child groups), so
the recursive walk surfaces just `{"/": ...}`. Multi-level recursion is
exercised in `test_icechunk.py` against the icechunk fixture which has
sweep groups.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from rustytree._rustytree import open_datatree


def _metadata_only(tree: dict) -> dict:
    """Return a copy of `tree` with each var's ``handle`` stripped.

    The handle is a `ZarrsArrayHandle` PyO3 object and uses identity
    comparison, so two structurally-equal trees from separate
    ``open_datatree`` calls would compare unequal at the var level even
    though their *metadata* matches. Tests that assert behavioural
    equivalence across kwargs use this to compare metadata only.
    """
    out: dict[str, Any] = {}
    for path, node in tree.items():
        out[path] = {
            "path": node["path"],
            "attrs": node["attrs"],
            "vars": [{k: v for k, v in var.items() if k != "handle"} for var in node["vars"]],
        }
    return out


def test_open_returns_dict_keyed_by_path(tiny_zarr_store: Path) -> None:
    tree = open_datatree(str(tiny_zarr_store))
    assert isinstance(tree, dict)
    assert "/" in tree
    root = tree["/"]
    assert set(root) == {"path", "attrs", "vars"}
    assert root["path"] == "/"


def test_root_attrs_round_trip(tiny_zarr_store: Path) -> None:
    tree = open_datatree(str(tiny_zarr_store))
    assert tree["/"]["attrs"] == {"title": "tiny"}


def test_vars_listed_complete(tiny_zarr_store: Path) -> None:
    tree = open_datatree(str(tiny_zarr_store))
    names = {var["name"] for var in tree["/"]["vars"]}
    assert names == {"temp", "mask"}


def test_var_metadata_shape_and_dims(tiny_zarr_store: Path) -> None:
    tree = open_datatree(str(tiny_zarr_store))
    by_name = {var["name"]: var for var in tree["/"]["vars"]}

    temp = by_name["temp"]
    assert temp["dims"] == ["lat", "lon"]
    assert temp["shape"] == [4, 3]
    assert temp["attrs"] == {"units": "K"}
    assert temp["dtype"].lower().startswith("float64") or "f8" in temp["dtype"]

    mask = by_name["mask"]
    assert mask["dims"] == ["lat", "lon"]
    assert mask["shape"] == [4, 3]
    assert mask["attrs"] == {}


def test_explicit_group_root_is_default(tiny_zarr_store: Path) -> None:
    a = open_datatree(str(tiny_zarr_store))
    b = open_datatree(str(tiny_zarr_store), group="/")
    assert _metadata_only(a) == _metadata_only(b)


def test_file_url_scheme_accepted(tiny_zarr_store: Path) -> None:
    tree = open_datatree(f"file://{tiny_zarr_store}")
    assert "/" in tree


def test_recursive_walk_finds_all_groups(multilevel_zarr_store: Path) -> None:
    tree = open_datatree(str(multilevel_zarr_store))
    assert set(tree) == {"/", "/volume_a", "/volume_a/sweep_0", "/volume_a/sweep_1"}


def test_recursive_walk_each_node_has_right_vars(multilevel_zarr_store: Path) -> None:
    tree = open_datatree(str(multilevel_zarr_store))

    assert {v["name"] for v in tree["/"]["vars"]} == {"x"}
    assert {v["name"] for v in tree["/volume_a"]["vars"]} == {"temp"}
    assert {v["name"] for v in tree["/volume_a/sweep_0"]["vars"]} == {"dbz"}
    assert {v["name"] for v in tree["/volume_a/sweep_1"]["vars"]} == {"dbz"}


def test_recursive_walk_attrs_round_trip(multilevel_zarr_store: Path) -> None:
    tree = open_datatree(str(multilevel_zarr_store))
    assert tree["/"]["attrs"] == {"title": "multilevel"}
    assert tree["/volume_a"]["attrs"] == {"id": "A"}
    assert tree["/volume_a/sweep_0"]["attrs"] == {"angle": 0.5}
    assert tree["/volume_a/sweep_1"]["attrs"] == {"angle": 1.5}


def test_recursive_walk_subtree_rooted_at_group(multilevel_zarr_store: Path) -> None:
    # Walking from /volume_a returns volume_a + its 2 sweeps; root and x
    # don't appear.
    subtree = open_datatree(str(multilevel_zarr_store), group="/volume_a")
    assert set(subtree) == {"/volume_a", "/volume_a/sweep_0", "/volume_a/sweep_1"}


def test_max_concurrency_kwarg_accepted(tiny_zarr_store: Path) -> None:
    # Smoke: tree opens and matches the default-walk result regardless of
    # the cap (the work fits inside any reasonable bound for a 1-group
    # tree). Verifies the kwarg is plumbed through without affecting
    # correctness.
    a = _metadata_only(open_datatree(str(tiny_zarr_store)))
    b = _metadata_only(open_datatree(str(tiny_zarr_store), max_concurrency=1))
    c = _metadata_only(open_datatree(str(tiny_zarr_store), max_concurrency=128))
    assert a == b == c


def test_max_concurrency_one_does_not_deadlock_on_deep_tree(
    multilevel_zarr_store: Path,
) -> None:
    # Regression: holding the semaphore permit across recursion would
    # deadlock for any max_concurrency < tree depth. The 3-level
    # multilevel fixture catches it; result must still be the full tree.
    tree = open_datatree(str(multilevel_zarr_store), max_concurrency=1)
    assert set(tree) == {"/", "/volume_a", "/volume_a/sweep_0", "/volume_a/sweep_1"}


def test_missing_path_raises_oserror(tmp_path: Path) -> None:
    missing = tmp_path / "nope.zarr"
    with pytest.raises(OSError):
        open_datatree(str(missing))


def test_unsupported_scheme_rejected_clearly() -> None:
    with pytest.raises(ValueError, match="unsupported URL scheme"):
        open_datatree("gs://bucket/store.zarr")


def test_s3_unknown_storage_option_rejected() -> None:
    with pytest.raises(ValueError, match="unknown s3 storage option"):
        open_datatree(
            "s3://example-bucket/store.zarr",
            storage_options={"regin": "us-east-1"},
        )


def test_s3_invalid_bool_option_rejected() -> None:
    with pytest.raises(ValueError, match="expects a boolean"):
        open_datatree(
            "s3://example-bucket/store.zarr",
            storage_options={"anon": "maybe"},
        )
