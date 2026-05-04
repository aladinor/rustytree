"""Single-group walk against a known local Zarr v3 fixture."""

from __future__ import annotations

from pathlib import Path

import pytest

from rustytree._rustytree import open_datatree


def test_open_root_returns_node_dict(tiny_zarr_store: Path) -> None:
    node = open_datatree(str(tiny_zarr_store))

    assert isinstance(node, dict)
    assert set(node) == {"path", "attrs", "vars"}
    assert node["path"] == "/"


def test_root_attrs_round_trip(tiny_zarr_store: Path) -> None:
    node = open_datatree(str(tiny_zarr_store))
    assert node["attrs"] == {"title": "tiny"}


def test_vars_listed_alphabetically_or_at_least_complete(tiny_zarr_store: Path) -> None:
    node = open_datatree(str(tiny_zarr_store))
    names = {var["name"] for var in node["vars"]}
    assert names == {"temp", "mask"}


def test_var_metadata_shape_and_dims(tiny_zarr_store: Path) -> None:
    node = open_datatree(str(tiny_zarr_store))
    by_name = {var["name"]: var for var in node["vars"]}

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
    assert a == b


def test_file_url_scheme_accepted(tiny_zarr_store: Path) -> None:
    node = open_datatree(f"file://{tiny_zarr_store}")
    assert node["path"] == "/"


def test_unsupported_scheme_rejected_clearly() -> None:
    with pytest.raises(NotImplementedError, match="local-filesystem"):
        open_datatree("s3://bucket/store.zarr")


def test_missing_path_raises_oserror(tmp_path: Path) -> None:
    missing = tmp_path / "nope.zarr"
    with pytest.raises(OSError):
        open_datatree(str(missing))
